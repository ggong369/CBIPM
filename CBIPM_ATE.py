import numpy as np
import pandas as pd
import random
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from bartpy.sklearnmodel import SklearnModel
from progress.bar import ChargingBar as Bar
import sys, os
from datetime import datetime

from functions import * 

import argparse
parser = argparse.ArgumentParser(description='PyTorch Covariate Balancing IPM for the ATE')

### for dataset 
parser.add_argument('--data_dir', default='', help='directory for data load and save result')
parser.add_argument('--n', type=int, default= 1000, help='the number of samples')
parser.add_argument('--sigma', type=int, default= 1, help='standard deviation for noise')
parser.add_argument('--ps', type=str, default= 'nonlinear', choices=['linear', 'nonlinear'], help='logit of ps is linear or not')
parser.add_argument('--seed', type=int, default= 1000, help='the number of seeds')

### for w modeling
parser.add_argument('--model', type=str, default= 'Nonpara', choices=['Linear', 'Nonpara'], help='Model for weights')

### for IPM
parser.add_argument('--IPM', type=str, default= 'MMD', choices=['WASS', 'SIPM', 'MMD'], help='IPM class')

### for training
parser.add_argument('--gpu', default=0, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--epoch', type=int, default= 1000)
parser.add_argument('--lr', default=0.03, type=float, help='initial learning rate for w')

parser.add_argument('--optim_adv', type=str, default= 'Adam', choices=['SGD', 'Adam'], help='optimizer for adv')
parser.add_argument('--epoch_adv', type=int, default= 5)
parser.add_argument('--lr_adv', default=0.3, type=float, help='initial learning rate for sipm')

# Only for WASS distance
parser.add_argument('--tau', type=float, default= 0.3, help='tau for gradient penalty of WASS')
parser.add_argument('--R', type=int, default= 100, help='number of samples for interpolate points')
parser.add_argument('--cutoff', type=float, default= 0.1, help='cutoff for WASS')

# Only for SIPM
parser.add_argument('--n_SIPM', type=int, default= 100, help='the number of sigmoid functions')

# for small overlap
parser.add_argument('--X', type=int, default= 1, help='X')

args = parser.parse_args()
print(args)

torch.cuda.set_device(args.gpu)

def main():    
    
    ATE_0_list_GLM = []
    ATE_OLS_list_GLM = []
    ATE_BART_list_GLM = []
    
    ATE_0_list = []
    ATE_OLS_list = []
    ATE_BART_list = []
    
    data_directory = args.data_dir + '/datasets/n_' + str(args.n) + '_ps' + args.ps + '_sigma_' + str(args.sigma)
    out_directory = args.data_dir + "/results/n_" + str(args.n) + '_ps' + args.ps + '_sigma_' + str(args.sigma)
    
    if args.X != 1:
        data_directory += '_X' + str(args.X)
        out_directory += '_X' + str(args.X)
    
    if not os.path.isdir(out_directory):
        mkdir_p(out_directory)
    
    bar = Bar('{:>6}'.format('Gage Bar '), max=args.seed)
    for seed in range(args.seed):

        start_time = datetime.now()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        
############################data upload###############################
        df = pd.read_csv(data_directory + '/seed_' + str(seed+1) + '.csv')
    
        X = np.array(df)[:,:4]
        std = StandardScaler()
        X_std = std.fit_transform(X)
        n, d= X.shape

        treatment = np.array(df)[:,4]
        idx = np.where(treatment==1)[0]

        y_0 = np.array(df)[:,5]
        y_OLS = np.array(df)[:,6]
        y_BART = np.array(df)[:,7]
            
        X_all = torch.tensor(X_std).float().cuda()
        X_c = torch.tensor(np.delete(X_std,idx,axis=0)).float().cuda()
        X_t = torch.tensor(X_std[idx,:]).float().cuda()
        n_0, n_1 = len(X_c), len(X_t)
        
################ For simple comparison ##########################        
        clf = LogisticRegression(random_state=seed).fit(X_std, treatment)
        glm_ps = clf.predict_proba(X_std)[:,1]

        v = 1 / glm_ps[idx]
        v_GLM = v / np.sum(v)        
        
        w = 1 / (1 - np.delete(glm_ps,idx))
        w_GLM = w / np.sum(w)

        ATE_0_GLM = y_0[idx] @ v_GLM  - np.delete(y_0,idx) @ w_GLM
        ATE_0_list_GLM.append(ATE_0_GLM)
        
        ATE_OLS_GLM = y_OLS[idx] @ v_GLM - np.delete(y_OLS,idx) @ w_GLM
        ATE_OLS_list_GLM.append(ATE_OLS_GLM)
        
        ATE_BART_GLM = y_BART[idx] @ v_GLM - np.delete(y_BART,idx) @ w_GLM
        ATE_BART_list_GLM.append(ATE_BART_GLM)

##################################################################    
        if args.model=='Nonpara':
            v = wNonpara_(n_1).cuda()
            w = wNonpara_(n_0).cuda()            
        elif args.model=='Linear':
            v = wlinear_ATE_(d,n,n_1).cuda()
            w = wlinear_ATE_(d,n,n_0).cuda()
            
        if args.IPM == "WASS":
            adv_v =  WASS_(d).cuda()
            adv_w =  WASS_(d).cuda()
            clipper = weightConstraint(args.cutoff)
        elif args.IPM == "SIPM":
            adv_v = para_sigmoid_(d,args.n_SIPM).cuda()
            adv_w = para_sigmoid_(d,args.n_SIPM).cuda()
        elif args.IPM == "MMD":
            adv_v = None
            adv_w = None            
            
        optimizer_v = optim.Adam(v.parameters(), lr=args.lr)
        optimizer_w = optim.Adam(w.parameters(), lr=args.lr)

        if args.IPM != "MMD":
            if args.optim_adv == "SGD":
                optimizer_adv_v = optim.SGD(adv_v.parameters(), lr=args.lr_adv, momentum=0.9, weight_decay=5e-4)
                optimizer_adv_w = optim.SGD(adv_w.parameters(), lr=args.lr_adv, momentum=0.9, weight_decay=5e-4)
            elif args.optim_adv == "Adam":    
                optimizer_adv_v = optim.Adam(adv_v.parameters(), lr=args.lr_adv)
                optimizer_adv_w = optim.Adam(adv_w.parameters(), lr=args.lr_adv)

        for epoch in range(args.epoch):
            if args.IPM != "MMD":   
                for epoch_adv in range(args.epoch_adv):
                    loss_adv = -loss_adv_(args.IPM, X_t, X_all, v, adv_v)
                    optimizer_adv_v.zero_grad()
                    loss_adv.backward()
                    optimizer_adv_v.step()
                    if args.IPM == "WASS":
                        adv_v.apply(clipper)
                    
            loss = loss_(args.IPM, X_t, X_all, v, adv_v)
            optimizer_v.zero_grad()
            loss.backward()
            optimizer_v.step()            
        v_IPM = v.weight(X_t).detach().cpu().numpy()
            
        for epoch in range(args.epoch):
            if args.IPM != "MMD":   
                for epoch_adv in range(args.epoch_adv):
                    loss_adv = -loss_adv_(args.IPM, X_c, X_all, w, adv_w)
                    optimizer_adv_w.zero_grad()
                    loss_adv.backward()
                    optimizer_adv_w.step()
                    if args.IPM == "WASS":
                        adv_w.apply(clipper)
                    
            loss = loss_(args.IPM, X_c, X_all, w, adv_w)
            optimizer_w.zero_grad()
            loss.backward()
            optimizer_w.step()
        w_IPM = w.weight(X_c).detach().cpu().numpy()       

        ATE_0 = y_0[idx] @ v_IPM - np.delete(y_0,idx) @ w_IPM
        ATE_0_list.append(ATE_0)
        
        ATE_OLS = y_OLS[idx] @ v_IPM - np.delete(y_OLS,idx) @ w_IPM
        ATE_OLS_list.append(ATE_OLS)
        
        ATE_BART = y_BART[idx] @ v_IPM - np.delete(y_BART,idx) @ w_IPM
        ATE_BART_list.append(ATE_BART)
    
        end_time = datetime.now()
        time_delta = round((end_time - start_time).total_seconds())     
        
        bar.suffix  = ' Seed : {se} | IPM,GLM | w/o : ({bias1}/{rmse1}), ({bias4}/{rmse4}) |  OLS : ({bias2}/{rmse2}), ({bias5}/{rmse5}) | BART : ({bias3}/{rmse3}), ({bias6}/{rmse6}) | time : {tf}s | remain : {eet}m)'.format(
                        se = seed,            
                        bias1=np.round(np.stack(ATE_0_list).mean(),3),
                        bias2=np.round(np.stack(ATE_OLS_list).mean(),3),
                        bias3=np.round(np.stack(ATE_BART_list).mean(),3),
                        bias4=np.round(np.stack(ATE_0_list_GLM).mean(),3),
                        bias5=np.round(np.stack(ATE_OLS_list_GLM).mean(),3),
                        bias6=np.round(np.stack(ATE_BART_list_GLM).mean(),3),            
                        rmse1=np.round(np.sqrt((np.stack(ATE_0_list)**2).mean()),3),
                        rmse2=np.round(np.sqrt((np.stack(ATE_OLS_list)**2).mean()),3),
                        rmse3=np.round(np.sqrt((np.stack(ATE_BART_list)**2).mean()),3),
                        rmse4=np.round(np.sqrt((np.stack(ATE_0_list_GLM)**2).mean()),3),
                        rmse5=np.round(np.sqrt((np.stack(ATE_OLS_list_GLM)**2).mean()),3),
                        rmse6=np.round(np.sqrt((np.stack(ATE_BART_list_GLM)**2).mean()),3),            
                        tf = time_delta,
                        eet = round(time_delta * (args.seed - seed - 1) / 60)
                        )
        
        bar.next()
    bar.finish() 
        
    print("IPM w/o augmentation (Bias,RMSE) : ", np.round(np.stack(ATE_0_list).mean(),3), np.round(np.sqrt((np.stack(ATE_0_list)**2).mean()),3))
    print("IPM OLS augmentation (Bias,RMSE) : ", np.round(np.stack(ATE_OLS_list).mean(),3), np.round(np.sqrt((np.stack(ATE_OLS_list)**2).mean()),3))
    print("IPM BART augmentation (Bias,RMSE) : ", np.round(np.stack(ATE_BART_list).mean(),3), np.round(np.sqrt((np.stack(ATE_BART_list)**2).mean()),3))
    
    pd.DataFrame(ATE_0_list).to_csv(out_directory + "/ATE_model_" + args.model + "_IPM_" + args.IPM + ".csv" , index=False)
    
        
if __name__ == '__main__':
    main()
