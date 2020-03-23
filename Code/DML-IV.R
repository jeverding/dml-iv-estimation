# ========================================================================================================== #
# ========================================================================================================== # 
#
# Double Machine Learning for IV estimation 
#
# This script implements the double machine learning (DML) approach to conduct causal inference. The script 
# defines functions which allow to sequentially execute the different DML steps and estimate instrumental 
# variables regressions eventually. Supports weighted regressions and wild cluster bootstrapped inference. 
#
# To Do: 
# - Export DML IV WCB results 
# - Run heterogeneity analyses 
#
# ========================================================================================================== #
# ========================================================================================================== #
# Remove all objects from the workspace
rm( list=ls() )

library(plyr)
library(dplyr)
library(Matrix)
library(hdm)
library(glmnet)
library(nnet)
library(ranger)
library(rpart)
library(rpart.plot)
library(xgboost)
library(foreign)
library(haven)
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
set.seed(seed.set)


# Functions --------------------------------------------------------------------------------------------------
# Clustering standard errors (traditional approach. Not needed for wild cluster bootstrap inf.) 
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
partial.out <- function(y, x, nfold = 5){
  # Setting up the table 
  columns <- c("MSE" 
               , "lambda", "alpha" # For elastic net 
               #, "mtry", "ntree" # For random forest (ranger) 
               #, "iterations", "learn_rate", "max_depth", "gamma", "frac_subsample", "colsample_bytree" # For gradient boosting (xgb) 
               )  
  table.mse <- data.frame(matrix(nrow=0, ncol= length(columns)))
  colnames(table.mse) <- columns
  
  # Split data in 80% training and 20% test data
  set.seed(seed.set)
  train <- sample(x = 1:nrow(x), size = nrow(x)*0.8) 
  x.train <- Matrix(x[train,], sparse = TRUE) 
  x.test <- Matrix(x[-train,], sparse = TRUE) 
  y.train <- y[train] 
  y.test <- y[-train] 
  
  
  #### OLS #####
  #Lin.mod <- lm(y.train~., data = data.frame(y.train, x.train[,-1]))
  # Calculating the MSE of the prediction error (removing the Intercept of x.test)
  #table.mse[1,1] <- mean((predict.lm(Lin.mod, as.data.frame(x.test[,-1])) - y.test)^2)
  #table.mse[1,2] <- "-"
  
  #### Elastic Net (incl. Lasso, Ridge) ####
  # Set step size for alpha 
  r <- 0.1
  for (i in seq(0, 1, r)) {
    # 10-fold CV to estimate tuning paramter with the lowest prediction error 
    set.seed(seed.set)
    cv.out <- cv.glmnet(x.train, y.train, family = "gaussian", nfolds = 10, alpha = i)
    # Select lambda (here: 1se instead of min.) 
    bestlam <- cv.out$lambda.1se
    # Get prediction error using the test data
    pred <- predict(cv.out, type = 'response', 
                    s = bestlam, newx = x.test)
    
    # add NA row and fill sequentially. 
    table.mse[dim(table.mse)[1]+1,] <- rep(NA, length(columns))
    if(i==0) {
      rownames(table.mse)[dim(table.mse)[1]]  <- "Ridge"
    } else if(i==1) {
      rownames(table.mse)[dim(table.mse)[1]]  <- "Lasso"
    } else {
      rownames(table.mse)[dim(table.mse)[1]]  <- paste0("Elastic Net (alpha=",i, ")")
    }
    table.mse$MSE[dim(table.mse)[1]] <- mean((pred - y.test)^2)
    table.mse$lambda[dim(table.mse)[1]] <- bestlam
    table.mse$alpha[dim(table.mse)[1]] <- i
    print(paste("glmnet using alpha =", i, "fitted."))
  }
  
  # Benchmarking: Select best model based on OOB MSE 
  # Prepare data for (nfold) cross-fitting 
  set.seed(seed.set)
  foldid <- rep.int(1:nfold, times = ceiling(nrow(x)/nfold))[sample.int(nrow(x))]
  I <- split(1:nrow(x), foldid)
  # Code up object to store residuals 
  til <- rep(NA, nrow(x))
  # Compute cross-fitted residuals 
  for(b in 1:length(I)){
    print(paste0("Compute residuals: Fold ", b, "/", length(I)))
    set.seed(seed.set)
    opt.alpha <- table.mse$alpha[which.min(table.mse$MSE)]
    opt.lambda <- table.mse$lambda[which.min(table.mse$MSE)]
    best.mod <- glmnet(x[-I[[b]],], y[-I[[b]]], 
                       alpha = opt.alpha, 
                       lambda = opt.lambda)
    hat <- predict(best.mod, type = 'response', 
                   s = opt.lambda, 
                   newx = x[I[[b]],])
    til[I[[b]]] <- (y[I[[b]]] - hat) 
    }
  return(list("til" = til, 
              "table.mse" = table.mse))
}

# Start ==================================================================================================== # 
# Load data --------------------------------------------------------------------------------------------------
data.share <- read_dta(file.path(data_dir,"share_rel6-1-1_data3.dta")) 

# Some additional pre-processing (select relevant variables, drop missings, recode variables)
ctrend_1 <- paste0("trend1cntry_", 1:length(unique(data.share$country))) 
ctrend_2 <- paste0("trend2cntry_", 1:length(unique(data.share$country))) 
ctrend_3 <- paste0("trend3cntry_", 1:length(unique(data.share$country))) 
var.select <- c("eurodcat", "chyrseduc", "t_compschool", 
                "country", "chbyear", "sex", "chsex", "int_year", "agemonth", "yrseduc", 
                ctrend_1, ctrend_2, ctrend_3, 
                "normchbyear", "w_ch") 

data.share <- 
  data.share %>% 
  select(var.select) %>% 
  na.omit() %>% 
  filter(normchbyear %in% c(-10:-1,1:10)) %>% 
  mutate(eurodcat = as.numeric(eurodcat))

# Main sample ================================================================================================
# Implement machine learning methods to get residuals --------------------------------------------------------
# Define outcome y, treatment d, and instrumental variable z  
y <- as.matrix(data.share[,"eurodcat"]) 
d <- as.matrix(data.share[,"chyrseduc"]) 
z <- as.matrix(data.share[,"t_compschool"]) 

# Code up formula for all models 
# Use this model for testing only (so that code runs faster)
#x <- model.matrix(~(-1 + factor(country) + factor(chbyear) + factor(int_year)), 
#                  data=data.share)
# Basic model for actual preliminary analyses 
x.formula <- as.formula(paste0("~(-1 + factor(country) + factor(chbyear) + factor(sex) + factor(chsex) + factor(int_year) + poly(agemonth,2) + poly(yrseduc,2) + ", 
                               paste(ctrend_1, collapse = " + "), " + ", 
                               paste(ctrend_2, collapse = " + "), " + ", 
                               paste(ctrend_3, collapse = " + "), 
                               ")^2"))
x <- model.matrix(x.formula, 
                         data=data.share) 

# Start: partialling out -------------------------------------------------------------------------------------
# (also check running time and save results) 
start_time <- Sys.time()
ytil <- partial.out(y, x)
y_end_time <- Sys.time() 
y_end_time - start_time 
write.csv2(as.data.frame(ytil$til),
           file=file.path(output_dir,'y_til.csv'),
           row.names=FALSE)
write.csv2(as.data.frame(ytil$table.mse),
           file=file.path(output_dir,'y_tablemse.csv'),
           row.names=FALSE)
dtil <- partial.out(d, x)
d_end_time <- Sys.time() 
d_end_time - start_time 
write.csv2(as.data.frame(dtil$til),
           file=file.path(output_dir,'d_til.csv'),
           row.names=FALSE)
write.csv2(as.data.frame(dtil$table.mse),
           file=file.path(output_dir,'d_tablemse.csv'),
           row.names=FALSE)
ztil <- partial.out(z, x) 
z_end_time <- Sys.time() 
z_end_time - start_time 
write.csv2(as.data.frame(ztil$til),
           file=file.path(output_dir,'z_til.csv'),
           row.names=FALSE)
write.csv2(as.data.frame(ztil$table.mse),
           file=file.path(output_dir,'z_tablemse.csv'),
           row.names=FALSE)

# Start: Inference -------------------------------------------------------------------------------------------
# IV regression using residuals along with wild cluster bootstrap inference 
# code up dataframe data.til, containing all residuals from partialling out 
data.til <- data.frame(y = as.numeric(ytil$til), 
                       d = as.numeric(dtil$til), 
                       z = as.numeric(ztil$til), 
                       cluster = as.numeric(factor(data.share$country)), 
                       wght = data.share$w_ch)
ivfit <- ivreg(formula = y ~ d | z, 
               weights = wght, 
               data = data.til)
summ.ivfit <- summary(ivfit)
summ.ivfit #$coefficients[2,3]
set.seed(seed.set)
ivfit.clust <- cluster.wild.ivreg(ivfit, 
                                  dat = data.til, 
                                  cluster = ~ cluster, 
                                  ci.level = 0.95, 
                                  boot.reps = 1000, 
                                  seed = seed.set)
ivfit.clust 

# Heterogeneity by gender ====================================================================================
# To Do: Estimate DML IV effects for fathers/mothers, parents of sons/daughters 
