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
from generator import *

import argparse
parser = argparse.ArgumentParser(description='PyTorch Covariate Balancing IPM for the ATT')

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

args = parser.parse_args()
print(args)

torch.cuda.set_device(args.gpu)

def main():    
    
    ATT_0_list_GLM = []
    ATT_OLS_list_GLM = []
    ATT_BART_list_GLM = []
    
    ATT_0_list = []
    ATT_OLS_list = []
    ATT_BART_list = []
    
    data_directory = args.data_dir + '/datasets/n_' + str(args.n) + '_ps' + args.ps + '_sigma_' + str(args.sigma)
    out_directory = args.data_dir + "/results/n_" + str(args.n) + '_ps' + args.ps + '_sigma_' + str(args.sigma)
    
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
          
        X_c = torch.tensor(np.delete(X_std,idx,axis=0)).float().cuda()
        X_t = torch.tensor(X_std[idx,:]).float().cuda()
        n_0, n_1 = len(X_c), len(X_t)
        
################ For simple comparison ##########################        
        clf = LogisticRegression(random_state=seed).fit(X_std, treatment)
        glm_ps = clf.predict_proba(X_std)[:,1]  
        w = np.delete(glm_ps,idx) / (1 - np.delete(glm_ps,idx))
        w_GLM = w / np.sum(w)

        ATT_0_GLM = y_0[idx].mean() - np.delete(y_0,idx) @ w_GLM
        ATT_0_list_GLM.append(ATT_0_GLM)
        
        ATT_OLS_GLM = y_OLS[idx].mean() - np.delete(y_OLS,idx) @ w_GLM
        ATT_OLS_list_GLM.append(ATT_OLS_GLM)
        
        ATT_BART_GLM = y_BART[idx].mean() - np.delete(y_BART,idx) @ w_GLM
        ATT_BART_list_GLM.append(ATT_BART_GLM)

##################################################################        
        if args.model=='Nonpara':
            w = wNonpara_(n_0).cuda()
        elif args.model=='Linear':
            w = wlinear_(d).cuda()
            
        if args.IPM == "WASS":
            adv =  WASS_(d).cuda()
            clipper = weightConstraint(args.cutoff)
        elif args.IPM == "SIPM":
            adv = para_sigmoid_(d,args.n_SIPM).cuda()
        elif args.IPM == "MMD":
            adv = None            
            
        optimizer = optim.Adam(w.parameters(), lr=args.lr)

        if args.IPM != "MMD":
            if args.optim_adv == "SGD":
                optimizer_adv = optim.SGD(adv.parameters(), lr=args.lr_adv, momentum=0.9, weight_decay=5e-4)
            elif args.optim_adv == "Adam":    
                optimizer_adv = optim.Adam(adv.parameters(), lr=args.lr_adv)

        for epoch in range(args.epoch):
            if args.IPM != "MMD":   
                for epoch_adv in range(args.epoch_adv):
                    loss_adv = -loss_adv_(args.IPM, X_c, X_t, w, adv)
                    optimizer_adv.zero_grad()
                    loss_adv.backward()
                    optimizer_adv.step()
                    if args.IPM == "WASS":
                        adv.apply(clipper)
                    
            loss = loss_(args.IPM, X_c, X_t, w, adv)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        w_IPM = w.weight(X_c).detach().cpu().numpy()            

        ATT_0 = y_0[idx].mean() - np.delete(y_0,idx) @ w_IPM
        ATT_0_list.append(ATT_0)
        
        ATT_OLS = y_OLS[idx].mean() - np.delete(y_OLS,idx) @ w_IPM
        ATT_OLS_list.append(ATT_OLS)
        
        ATT_BART = y_BART[idx].mean() - np.delete(y_BART,idx) @ w_IPM
        ATT_BART_list.append(ATT_BART)
    
        end_time = datetime.now()
        time_delta = round((end_time - start_time).total_seconds())     
        
        bar.suffix  = ' Seed : {se} | IPM,GLM | w/o : ({bias1}/{rmse1}), ({bias4}/{rmse4}) |  OLS : ({bias2}/{rmse2}), ({bias5}/{rmse5}) | BART : ({bias3}/{rmse3}), ({bias6}/{rmse6}) | time : {tf}s | remain : {eet}m)'.format(
                        se = seed,            
                        bias1=np.round(np.stack(ATT_0_list).mean(),3),
                        bias2=np.round(np.stack(ATT_OLS_list).mean(),3),
                        bias3=np.round(np.stack(ATT_BART_list).mean(),3),
                        bias4=np.round(np.stack(ATT_0_list_GLM).mean(),3),
                        bias5=np.round(np.stack(ATT_OLS_list_GLM).mean(),3),
                        bias6=np.round(np.stack(ATT_BART_list_GLM).mean(),3),            
                        rmse1=np.round(np.sqrt((np.stack(ATT_0_list)**2).mean()),3),
                        rmse2=np.round(np.sqrt((np.stack(ATT_OLS_list)**2).mean()),3),
                        rmse3=np.round(np.sqrt((np.stack(ATT_BART_list)**2).mean()),3),
                        rmse4=np.round(np.sqrt((np.stack(ATT_0_list_GLM)**2).mean()),3),
                        rmse5=np.round(np.sqrt((np.stack(ATT_OLS_list_GLM)**2).mean()),3),
                        rmse6=np.round(np.sqrt((np.stack(ATT_BART_list_GLM)**2).mean()),3),            
                        tf = time_delta,
                        eet = round(time_delta * (args.seed - seed - 1) / 60)
                        )        
        
        bar.next()
    bar.finish()  
        
    print("IPM w/o augmentation (Bias,RMSE) : ", np.round(np.stack(ATT_0_list).mean(),3), np.round(np.sqrt((np.stack(ATT_0_list)**2).mean()),3))
    print("IPM OLS augmentation (Bias,RMSE) : ", np.round(np.stack(ATT_OLS_list).mean(),3), np.round(np.sqrt((np.stack(ATT_OLS_list)**2).mean()),3))
    print("IPM BART augmentation (Bias,RMSE) : ", np.round(np.stack(ATT_BART_list).mean(),3), np.round(np.sqrt((np.stack(ATT_BART_list)**2).mean()),3))
    
    pd.DataFrame(ATT_0_list).to_csv(out_directory + "/model_" + args.model + "_IPM_" + args.IPM + ".csv" , index=False)
            

if __name__ == '__main__':
    main()
