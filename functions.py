import numpy as np
import random
import torch
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import sys, os
import torch.nn.functional as F

################################### w ####################################  
class wlinear_(nn.Module):
    def __init__(self, d):
        super(wlinear_, self).__init__()

        self.layer = nn.Linear(d, 1, bias=False)
    
    def weight(self, x):        
        x = nn.Softmax(0)(self.layer(x).squeeze())
        return x    
        
class wNonpara_(nn.Module):
    def __init__(self, n):
        super(wNonpara_, self).__init__()
                
        self.vec = nn.Parameter(torch.ones(n))
    
    def weight(self, x):
        x = nn.Softmax(0)(self.vec)
        return x
    
class wlinear_ATE_(nn.Module):
    def __init__(self, d, n, n_m):
        super(wlinear_ATE_, self).__init__()

        self.layer = nn.Linear(d, 1, bias=False)
        self.n = n
        self.n_m = n_m
    
    def weight(self, x):        
        x = nn.Softmax(0)(self.layer(x).squeeze())
        return (1/self.n + x*(1 - self.n_m/self.n))   
    
###################################IPM####################################    
    
# Wassesterin IPM
class WASS_(nn.Module):
    def __init__(self, input_dim, L=1, p=100, slope = 0.1):
        super(WASS_, self).__init__()

        self.input_dim = input_dim
        self.L = L
        self.p = p
        self.act= nn.LeakyReLU(slope)
                
        self.layers = self._make_layer()
        
    def _make_layer(self):        
        layers = []
        layers.append(nn.Linear(self.input_dim, self.p))
        layers.append(self.act)
        
        for l in range(self.L-1):       
            layers.append(nn.Linear(self.p, self.p))
            layers.append(self.act)
            
        layers.append(nn.Linear(self.p, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):              
        x = self.layers(x).squeeze()
        return x 
    
# SIPM
class para_sigmoid_(nn.Module):
    def __init__(self, input_dim, n_SIPM):
        super(para_sigmoid_, self).__init__()
                
        self.linear = nn.Linear(input_dim,n_SIPM)
    
    def forward(self, x):              
        x = self.linear(x)
        x = nn.Sigmoid()(x.squeeze())
        return x


########################## loss function for gradient ascent#########################################
def loss_adv_(IPM, X_c, X_t, w, adv, R=100, tau=0.3):
    
    if IPM == "WASS":
        return loss_adv_WASS_(X_c, X_t, w, adv, R, tau)
    elif IPM == "SIPM":
        return loss_adv_SIPM_(X_c, X_t, w, adv)
    else:
        return("IPM method error")
    
def loss_adv_WASS_(X_c, X_t, w, adv, R, tau):
    n_0, n_1 = len(X_c), len(X_t)
    
    adv_t = adv(X_t)
    adv_c = adv(X_c)
    
    weight_t = (torch.ones(n_1)/n_1).cuda()
    weight_c = w.weight(X_c) 
    
    out = torch.sum(weight_t * adv_t) - torch.sum(weight_c * adv_c)
    out = out**2   
    out -= gradient_penalty_(X_c, X_t, adv, R, tau)
    
    return out
    
def loss_adv_SIPM_(X_c, X_t, w, adv):
    n_0, n_1 = len(X_c), len(X_t)
    
    adv_t = adv(X_t)
    adv_c = adv(X_c)
    
    weight_t = (torch.ones(n_1)/n_1).cuda()
    weight_c = w.weight(X_c) 

    if adv_t.dim()==1:
        out = torch.sum(weight_t * adv_t) - torch.sum(weight_c * adv_c)
    else:        
        out = torch.sum(weight_t.reshape(-1,1) * adv_t, 0) - torch.sum(weight_c.reshape(-1,1) * adv_c, 0)
    out = out**2
    
    return torch.sum(out)

def gradient_penalty_(X_c, X_t, adv, R, tau):
    X_c_sample = X_c[random.choices(range(len(X_c)),k=R)]
    X_t_sample = X_t[random.choices(range(len(X_t)),k=R)]
    alpha = torch.rand(R, 1).cuda()
    interpolates = (X_c_sample * alpha + X_t_sample * (1-alpha)).requires_grad_(True)
    
    d_interpolates = adv(interpolates).view(R, 1).cuda()
    fake = torch.autograd.Variable(torch.Tensor(R, 1).fill_(1.0), requires_grad=False).cuda()
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
    return tau * gradient_penalty 

class weightConstraint(object):
    def __init__(self, cutoff):
        self.cutoff = cutoff
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w=w.clamp(-self.cutoff,self.cutoff)
            module.weight.data=w


########################## loss function for gradient descent################################

def loss_(IPM, X_c, X_t, w, adv, gamma=10):
    
    if IPM == "WASS":
        return loss_WASS_(X_c, X_t, w, adv)
    elif IPM == "SIPM":
        return loss_SIPM_(X_c, X_t, w, adv)
    elif IPM == "MMD":
        return loss_MMD_(X_c, X_t, w, gamma)
    else:
        return("IPM method error")    
    
def loss_WASS_(X_c, X_t, w, adv):
    n_0, n_1 = len(X_c), len(X_t)
    
    adv_t = adv(X_t)
    adv_c = adv(X_c)
    
    weight_t = (torch.ones(n_1)/n_1).cuda()
    weight_c = w.weight(X_c) 
    
    out = torch.sum(weight_t * adv_t) - torch.sum(weight_c * adv_c)

    return out**2

def loss_SIPM_(X_c, X_t, w, adv):
    n_0, n_1 = len(X_c), len(X_t)
    
    adv_t = adv(X_t)
    adv_c = adv(X_c)
    
    weight_t = (torch.ones(n_1)/n_1).cuda()
    weight_c = w.weight(X_c) 

    if adv_t.dim()==1:
        out = torch.sum(weight_t * adv_t) - torch.sum(weight_c * adv_c)
    else:        
        out = torch.sum(weight_t.reshape(-1,1) * adv_t, 0) - torch.sum(weight_c.reshape(-1,1) * adv_c, 0)
    
    return torch.max(out**2)

def loss_MMD_(X_c, X_t, w, gamma):       
    weight_c = w.weight(X_c)
    weight_c = len(X_c)*(weight_c.reshape(-1,1))
    mmd = []  
    
    if gamma == -1:
        gamma_list = [1.0, 3.0, 10.0]    
      
        for gamma in gamma_list:
            KXX = Gaussian_kernel_matrix_(X_c, X_c, gamma) * (weight_c @ weight_c.T)
            KXY = Gaussian_kernel_matrix_(X_c, X_t, gamma) * weight_c
            KYY = Gaussian_kernel_matrix_(X_t, X_t, gamma)
            mmd.append(KXX.mean() - 2 * KXY.mean() + KYY.mean())
            
    else:
        KXX = Gaussian_kernel_matrix_(X_c, X_c, gamma) * (weight_c @ weight_c.T)
        KXY = Gaussian_kernel_matrix_(X_c, X_t, gamma) * weight_c
        KYY = Gaussian_kernel_matrix_(X_t, X_t, gamma)
        mmd.append(KXX.mean() - 2 * KXY.mean() + KYY.mean())
        
    return torch.mean(torch.stack(mmd))

def Gaussian_kernel_matrix_(X_c, X_t, gamma):
    matrix = -torch.cdist(X_c, X_t, p=2)**2
    matrix /= (2.0 * gamma**2)
    matrix = torch.exp(matrix)
    return matrix

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise