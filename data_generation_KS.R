library(BART); library(CBPS)
setwd('')

nn = c(200, 1000)
sigma.list = c(1.0, 2.0)
seed_end = 1000
ps.method = "nonlinear"  # or "linear"
const = 2    # or Do Not define 

#############
expit <- function(x){
  return(1/(1 + exp(-1 * x)))
}

make_X <- function(Z){
  X1 = exp(Z[,1]/2)
  X2 = Z[,2]/(1+exp(Z[,1])) + 10
  X3 = (Z[,1]*Z[,3]/25 + 0.6)**3
  X4 = (Z[,2]+Z[,4]+20)**2
  X = cbind(X1, X2, X3, X4)
  return(X)
}

weight_by_GLM = function(treatment, X.model, idx, ATT){
  glm_ps = glm(treatment ~ X.model, family = binomial(link = "logit"))$fitted.values
  
  w = treatment
  if(ATT == 0){
    w[idx] = sum1(1 / glm_ps[idx])
    w[!idx] = sum1(1 / (1 - glm_ps[!idx]))
  }
  else if(ATT == 1){
    w = sum1(glm_ps[!idx] / (1 - glm_ps[!idx]))
  }
  
  return(w)
}

sum1 = function(object){
  so = sum(object)
  object / so
}

#############
for(h in 1:length(sigma.list)){
  for(k in 1:length(nn)){
    
    if(exists("const")) {
      data_out = paste0('n_',nn[k],'_ps',ps.method,'_sigma_',sigma.list[h], '_X', const)
    } else {
      data_out = paste0('n_',nn[k],'_ps',ps.method,'_sigma_',sigma.list[h])
    }
    
    dir.create(data_out)
    n = nn[k]
    sigma = sigma.list[h]
    
    for (seed in 1:seed_end){
      set.seed(seed)
      Z = rnorm(n = n*4, mean = 0, sd = 1)
      Z = matrix(Z, ncol = 4)
      
      if(ps.method == "nonlinear"){
        
        coeff_pi = c(-1.0, 0.5, -0.25, -0.1)
        coeff_y = c(27.4, 13.7, 13.7, 13.7)
        mu = 210.0
        
        if(exists("const")) {
          true_ps = expit((Z %*% coeff_pi) * const)
        } else {
          true_ps = expit(Z %*% coeff_pi)
        }
        
        u = runif(n)
        treatment = ifelse(true_ps > u, 1, 0)
        idx = (treatment == 1)
        
        y = Z %*% coeff_y + mu + rnorm(n = n, mean = 0, sd = sigma)
        
        X = make_X(Z)
        X_std = scale(X)
        
      } else if (ps.method == "linear") {
        
        coeff_pi = c(1, -0.5, -2, -0.01)
        nu = 8
        coeff_y = c(27.4, 13.7, 13.7, 13.7)
        mu = 210.0
        
        X = make_X(Z)
        
        true_ps = expit(X %*% coeff_pi + nu)
        u = runif(n)
        treatment = ifelse(true_ps > u, 1, 0)
        idx = (treatment == 1)
        
        y = Z %*% coeff_y + mu + rnorm(n = n, mean = 0, sd = sigma)
        
        X_std = scale(X)
      }
      
      df.idx = data.frame(X_std, y)[!idx,] 
      m0 = lm(y ~ ., data = df.idx)
      new.d = data.frame(X_std)
      y_aug_lm = y - predict.lm(m0, newdata = new.d)
      
      y_aug_BART = y - wbart(
        x.train = X_std[!idx, , drop = FALSE],
        y.train = y[!idx],
        x.test = X_std
      )$yhat.test.mean

      data = cbind(X, treatment, y, y_aug_lm, y_aug_BART)
      colnames(data) = c('X1','X2','X3','X4', 'treatment', 'y', 'aug-OLS', 'aug-BART')
      
      write.csv(data, paste0(data_out,'/seed_',seed,'.csv'), row.names = F)
    }
  }
}


