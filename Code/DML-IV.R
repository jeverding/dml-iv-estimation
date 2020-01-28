# ========================================================================================================== #
# ========================================================================================================== # 
#
# Double Machine Learning for IV estimation 
#
# This script implements the double machine learning (DML) approach to conduct causal inference. The script 
# defines functions which allow to sequentially execute the different DML steps and estimate instrumenal 
# variables regressions eventually. 
#
# To Do: 
# - Implement behnchmarking of different ML methods for each partialling out step. (Incl. automatic 
# selection of best performing ML method to compute residuals.) 
# - Estimation of various standard errors, e.g. wild cluster robust 
# - Implement support for choosing regression weights 
#
# ========================================================================================================== #
# ========================================================================================================== #
# Remove all objects from the workspace
rm( list=ls() )

library(plyr)
library(dplyr)
library(hdm)
library(glmnet)
library(nnet)
library(randomForest)
library(rpart)
library(rpart.plot)
library(gbm)
library(foreign)
library(readstata13)
library(sandwich)
library(AER)
library(clusterSEs)


# Setup ------------------------------------------------------------------------------------------------------
# Set working directory using main project path 
main_dir <- getwd() 
code_dir <- file.path(main_dir,"Code")
data_dir <- file.path(main_dir,"Data")
output_dir <- file.path(main_dir,"Output")
# Set seed 
seed.set <- 180911 


# Functions --------------------------------------------------------------------------------------------------
# Clustering standard errors
# Traditional approach. Not needed for wild cluster bootstrap inf. 
clust.se <- function(est.model, cluster){
  G <- length(unique(cluster))
  N <- length(cluster)
  # reweight the variance-covariance matrix by groups (clusters) and sample size, using the sandwich package 
  dfa <- (G/(G - 1)) * ((N - 1)/(N - est.model$rank))
  u <- apply(estfun(est.model),2,
             function(x) tapply(x, cluster, sum))
  vcovCL <- dfa*sandwich(est.model, meat=crossprod(u)/N)
  coeftest(est.model, vcovCL) 
}

# Partialling out (first part of DML algorithm) 
Partial.out <- function(y,x){
  # Setting up the table  the ranking
  table.mse<- matrix(0, nrow = 6, ncol = 2)              
  #rownames(table.mse)<- c("OLS", "Ridge",  "Elastic Net (alpha = 0.2)", "Elastic Net (alpha = 0.4)", "Elastic Net (alpha = 0.6)" ,"Elastic Net (alpha = 0.8)","Lasso" , "Random Forest", "Gradient Boosting", "Neural Nets")
  rownames(table.mse)<- c("Ridge",  "Elastic Net (alpha = 0.2)", "Elastic Net (alpha = 0.4)", "Elastic Net (alpha = 0.6)" ,"Elastic Net (alpha = 0.8)","Lasso")
  colnames(table.mse)<- c("MSE", "Optimal Tuning Parameter")               
  
  # Sample splitting 80-20
  train <- sample(1:nrow(x),8*dim(x)[1]/10) 
  # Split data in 80% training and 20 % test data. Leaving out the 1 vector because glmnet will automatically estimate the intercept.  
  x.train <- x[train,]
  y.train <- y[train]
  x.test <- x[-train,] 
  y.test <- y[-train] 
  
  #### OLS #####
  #Lin.mod <- lm(y.train~., data = data.frame(y.train, x.train[,-1]))
  # Calculating the MSE of the prediction error (removing the Intercept of x.test)
  # Issuse -> Dummies high probability of Perfect multicollinearity. No reliable Prediction.
  #table.mse[1,1] <- mean((predict.lm(Lin.mod, as.data.frame(x.test[,-1])) - y.test)^2)
  #table.mse[1,2] <- "-"
  
  #### Lasso Elastic Net Ridge ####
  # Set step size for alpha 
  r <- 5
  for (i in seq(0,1, 1/r)) {
    # 5-fold CV to estimate tuning paramter with the lowest prediction error (could also use e.g. 10 folds)
    cv.out <- cv.glmnet(x.train, y.train, family = "gaussian", nfolds = 10, alpha = i)
    # Select lambda (here: 1se instead of min.) 
    bestlam <- cv.out$lambda.1se
    # Get prediction error using the test data
    pred <- predict(cv.out, type = 'response', 
                    s = bestlam, newx = x.test)
    table.mse[r*i+1,1] <- mean((pred - y.test)^2) 
    table.mse[r*i+1,2] <- bestlam
    print(paste("Model using alpha =", i, "fitted."))
  }
  
  paste(rownames(table.mse)[which.min(table.mse[,1])], "performs best with a MSE of", table.mse[which.min(table.mse[,1]),1], "and a hyperparameter of",table.mse[which.min(table.mse[,1]),2])
  
  # Benchmarking: Select best method based on OOB MSE 
  opt.lambda <- table.mse[which.min(table.mse[,1]), 2]
  if (rownames(table.mse)[which.min(table.mse[,1])] == "Elastic Net (alpha = 0.2)") {
    best.mod <-  glmnet(x, y, 
                        alpha = 0.2, lambda = opt.lambda)
    y.hat <- predict(best.mod, type = 'response', s = opt.lambda, newx = x)
  }
  if (rownames(table.mse)[which.min(table.mse[,1])] == "Elastic Net (alpha = 0.4)") {
    best.mod <-  glmnet(x, y, 
                        alpha = 0.4, lambda = opt.lambda)
    y.hat <- predict(best.mod, type = 'response', s = opt.lambda, newx = x)
  }
  if (rownames(table.mse)[which.min(table.mse[,1])] == "Elastic Net (alpha = 0.6)") {
    best.mod <-  glmnet(x, y, 
                        alpha = 0.6, lambda = opt.lambda)
    y.hat <- predict(best.mod, type = 'response', s = opt.lambda, newx = x)
  }
  if (rownames(table.mse)[which.min(table.mse[,1])] == "Elastic Net (alpha = 0.8)") {
    best.mod <-  glmnet(x, y, 
                        alpha = 0.8, lambda = opt.lambda)
    y.hat <- predict(best.mod, type = 'response', s = opt.lambda, newx = x)
  }
  if (rownames(table.mse)[which.min(table.mse[,1])] == "Lasso") {
    best.mod <-  glmnet(x, y, 
                        alpha = 1, lambda = opt.lambda)
    y.hat <- predict(best.mod, type = 'response', s = opt.lambda, newx = x)
  }
  if (rownames(table.mse)[which.min(table.mse[,1])] == "Ridge") {
    best.mod <-  glmnet(x, y, 
                        alpha = 0, lambda = opt.lambda)
    y.hat <- predict(best.mod, type = 'response', s = opt.lambda, newx = x)
  }
  ytil <- (y - y.hat) 
  return(list("ytil" = ytil, 
              "table.mse" = table.mse))
}

# DML for PLIVM
DML2.for.PLIVM <- function(x, y, d, z, yreg, dreg, zreg, nfold=2) {
  # this implements DML2 algorithm. Moments are estimated via DML, randomly split data into folds before 
  # estimating pooled estimate of theta 
  nobs <- nrow(x)
  foldid <- rep.int(1:nfold,times = ceiling(nobs/nfold))[sample.int(nobs)]
  I <- split(1:nobs, foldid)
  # code up residualized objects to fill
  ytil <- dtil <- ztil<- rep(NA, nobs)
  # get cross-fitted residuals
  cat("fold: ")
  for(b in 1:length(I)){
    # take a fold out 
    dfit <- dreg(x[-I[[b]],], d[-I[[b]]])  
    zfit <- zreg(x[-I[[b]],], z[-I[[b]]])  
    yfit <- yreg(x[-I[[b]],], y[-I[[b]]])  
    # predict out folds 
    dhat <- predict(dfit, x[I[[b]],], type="response")  
    zhat <- predict(zfit, x[I[[b]],], type="response")  
    yhat <- predict(yfit, x[I[[b]],], type="response")  
    # save residuals
    dtil[I[[b]]] <- (d[I[[b]]] - dhat) 
    ztil[I[[b]]] <- (z[I[[b]]] - zhat) 
    ytil[I[[b]]] <- (y[I[[b]]] - yhat) 
    cat(b," ")
  }
  ivfit= tsls(y=ytil,d=dtil, x=NULL, z=ztil, intercept=FALSE)
  # save estimation parameters (i.e. regression coefficients and standard errors)
  coef.est <-  ivfit$coef 
  se <-  ivfit$se 
  cat(sprintf("\ncoef (se) = %g (%g)\n", coef.est , se))
  return( list(coef.est =coef.est , se=se, dtil=dtil, ytil=ytil, ztil=ztil) )
}

# Start ==================================================================================================== # 
# Load data --------------------------------------------------------------------------------------------------
### To Do: Maybe change syntax for import here 
data.share <- read.dta13(file.path(data_dir,"share_rel6-1-1_data3.dta")) 

# Some additional pre-processing (select relevant variables, drop missings, recode variables)
var.select <- c("eurodcat", "chyrseduc", "t_compschool", "sex", "chsex", "alone", "int_year", "chmarried", "chdivorce", "chwidow", "ch200km", "chclose", "chlescontct", "chmuccontct", "ch007_", "ch014_", "ch016_", "married", "divorce", "widow", "partnerinhh", "ep005_", "agemonth", "hhsize", "yrseduc", "chnchild", "chbyear", "country")
data.share <- 
  data.share %>% 
  select(var.select) %>% 
  na.omit() %>% 
  mutate(eurodcat = as.numeric(eurodcat))

# Assign outcome y, treatment d, and instrumental variable z  
y <- as.matrix(data.share[,"eurodcat"]) 
d <- as.matrix(data.share[,"chyrseduc"]) 
z <- as.matrix(data.share[,"t_compschool"]) 

# ++++
# test: ivreg for cluster robust inference (to do: wrap in function)
data.share$y <- data.share[,"eurodcat"] 
data.share$d <- data.share[,"chyrseduc"] 
data.share$z <- data.share[,"t_compschool"] 
data.share$cluster <- as.numeric(factor(data.share$country)) 
test.ivfit <- ivreg(formula = y ~ d | z, 
                    data = data.share)
summ.test.ivfit <- summary(test.ivfit)
summ.test.ivfit #$coefficients[2,3]
test.ivfit.clust <- cluster.wild.ivreg(test.ivfit, 
                                       dat = data.share, 
                                       cluster = ~ cluster, 
                                       ci.level = 0.95, 
                                       boot.reps = 1000, 
                                       seed = seed.set)
test.ivfit.clust
# ++++

## define level of clustering standard errors 
## (not needed when using clusterSEs methods like wild cluster bootstrap)
#cluster.level <- as.numeric(factor(data.share$country)) 

# Implement machine learning methods to get residuals --------------------------------------------------------
# Code up model for regularized regression methods 
# Use this model for testing only (so that code runs faster)
x <- model.matrix(~(factor(country) + factor(chbyear) + factor(int_year)), # + sex + chsex + factor(ch016_) + poly(agemonth,2) + poly(hhsize,2) + poly(yrseduc,2))^2, 
                  data=data.share)
# Basic model for actual preliminary analyses 
x.prelim <- model.matrix(~(sex + chsex + alone + factor(int_year) + chmarried + chdivorce + chwidow + ch200km + chclose + chlescontct + chmuccontct + factor(ch007_) + factor(ch014_) + factor(ch016_) + married + divorce + widow + partnerinhh + factor(ep005_) + poly(agemonth,4) + poly(hhsize,4) + poly(yrseduc,4) + poly(chnchild,4) + factor(chbyear) + factor(country))^2, 
                         data=data.share)

# 1) Lasso 
yreg <- function(x,y){ rlasso(x, y) } 
dreg <- function(x,d){ rlasso(x, d) } 
zreg <- function(x,z){ rlasso(x, z) } 

# Run DML algorithm 
DML2.lasso <- DML2.for.PLIVM(x, y, d, z, yreg, dreg, zreg, nfold = 2)
