setwd('')

n = 200
seed_end = 1000

data_out = paste0('n_',n,'_ps_nonlinear_hetero')
dir.create(data_out)

for(seed in 1:seed_end){
  
  set.seed(seed)
  X = runif(n = n*6, min = -2, max = 2)
  X = matrix(X, ncol = 6)
  x1 = X[,1]; x2 = X[,2]; x3 = X[,3]; x4 = X[,4]; x5 = X[,5]; x6 = X[,6]
  
  mu1 = ( (sin(max(x1,x2,x3)) + (max(x3,x4,x5))^2) / (2+(x1+x5)^2) ) + 4 * (x1^3) * sin(3*x3) * (1+exp(x4-0.5*x3)) + 
    x3^2 + 2*sin(x4)*(x5)^2 - 3

  t_prime = rnorm(n = n, mean = mu1, sd = sqrt(0.5))
  
  expit <- function(x){
    return(1/(1 + exp(-1 * x)))
  }
  
  t = expit(t_prime)
  u = runif(n)
  treatment = ifelse(t > u, 1, 0)
  idx = (treatment == 1)

  mu_2 = function(treat) {2 * (x1-2)^2 + 5*cos(2*x5)*treat + ((1/(x2^2+1))*(max(x1,x6)^3 / (1+2*x3^2))*sin(x2)) + 3*(2*x4-1)^2 }
  mu2 = mu_2(treatment)
  y = rnorm(n = n, mean = mu2, sd = sqrt(0.1))
  
  data = cbind(X, treatment, y)
  colnames(data) = c('X1','X2','X3','X4', 'X5', 'X6', 'treatment', 'y')
  
  write.csv(data, paste0(data_out,'/seed_',seed,'.csv'), row.names = F)
}

