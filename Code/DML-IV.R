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
# - Check data type of outcome and adjust fam.glmnet; binary: binomial; else: gaussian 
# - Implement RF using ranger 
# - Implement GBM for partialling out / model selection 
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
library(ranger)
library(rpart)
library(rpart.plot)
library(gbm)
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
partial.out <- function(y,x){
  # Setting up the table 
  columns <- c("MSE", "lambda", "alpha", "mtry", "ntree")  
  table.mse <- data.frame(matrix(nrow=0, ncol= length(columns)))
  colnames(table.mse) <- columns
  
  # Split data in 80% training and 20 % test data 
  train <- sample(x = 1:nrow(x), size = dim(x)[1]*0.8) 
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
  r <- 0.2
  for (i in seq(0, 1, r)) {
    # 5-fold CV to estimate tuning paramter with the lowest prediction error (could also use e.g. 10 folds)
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
    print(paste("Model using alpha =", i, "fitted."))
  }
  
  #### Random Forest ####
  ntree.set <- 5000 
  # Use different mtrys, eventually select best RF model 
  mtry.seq <- seq(from = 2, to = floor(sqrt(ncol(x.train)))*2, by = 2)
  if (!floor(sqrt(ncol(x.train))) %in% mtry.seq) {
    mtry.seq <- sort(c(mtry.seq, 
                       floor(sqrt(ncol(x.train)))
                       )) 
  } 
  for (i in 1:length(mtry.seq)) {
    rf <- randomForest(x.train, y.train, 
                       ntree = ntree.set, 
                       mtry = mtry.seq[i]) 
    # TO DO: implement RF using ranger 
    #rf <- ranger(x = x.train, y = y.train, 
    #             num.trees = ntree.set, 
    #             mtry = mtry.seq[i]) 
    pred <- predict(rf, newdata = x.test, type="response") 
    
    # new NA row for random forest, fill again sequentially 
    table.mse[dim(table.mse)[1]+1,] <- rep(NA,length(columns)) 
    rownames(table.mse)[dim(table.mse)[1]]  <- paste0("Random Forest, no. ", i, "/", length(mtry.seq)) 
    table.mse$MSE[dim(table.mse)[1]] <- mean((pred - y.test)^2) 
    table.mse$mtry[dim(table.mse)[1]] <- mtry.seq[i] 
    table.mse$ntree[dim(table.mse)[1]]<- ntree.set 
  }
  
  # Benchmarking: Select best method based on OOB MSE 
  # (Identify best method directly using method-specific tuning parameters): 
  if (!is.na(table.mse$lambda[which.min(table.mse$MSE)])) { 
    opt.alpha <- table.mse$alpha[which.min(table.mse$MSE)]
    opt.lambda <- table.mse$lambda[which.min(table.mse$MSE)]
    best.mod <- glmnet(x, y, 
                       alpha = opt.alpha, 
                       lambda = opt.lambda)
    y.hat <- predict(best.mod, type = 'response', 
                     s = opt.lambda, 
                     newx = x)
  }
  if (!is.na(table.mse$mtry[which.min(table.mse$MSE)])) { 
    opt.mtry <- table.mse$mtry[which.min(table.mse$MSE)] 
    opt.ntree <- table.mse$ntree[which.min(table.mse$MSE)] 
    best.mod <- randomForest(y~., data.frame(y, x), 
                             ntree = opt.ntree,
                             mtry = opt.mtry, 
                             importance = TRUE)
    y.hat <- predict(best.mod, newdata = data.frame(y, x)) 
  }
  
  ytil <- (y - y.hat) 
  return(list("til" = ytil, 
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

# Implement machine learning methods to get residuals --------------------------------------------------------
# Define outcome y, treatment d, and instrumental variable z  
y <- as.matrix(data.share[,"eurodcat"]) 
d <- as.matrix(data.share[,"chyrseduc"]) 
z <- as.matrix(data.share[,"t_compschool"]) 

# Code up formula for all models 
# Use this model for testing only (so that code runs faster)
x <- model.matrix(~(factor(country) + factor(chbyear) + factor(int_year)), 
                  data=data.share)
# Basic model for actual preliminary analyses 
x.formula <- as.formula(paste0("~(-1 + factor(country) + factor(chbyear) + factor(sex) + factor(chsex) + factor(int_year) + poly(agemonth,2) + poly(yrseduc,2) + ", 
                               paste(ctrend_1, collapse = " + "), " + ", 
                               paste(ctrend_2, collapse = " + "), " + ", 
                               paste(ctrend_3, collapse = " + "), 
                               ")"))
x.prelim <- model.matrix(x.formula, 
                         data=data.share)

ytil <- partial.out(y, x)
dtil <- partial.out(d, x)
ztil <- partial.out(z, x)

# Start: Inference -------------------------------------------------------------------------------------------
# IV regression using residuals along with wild cluster bootstrap inference 
# code up dataframe data.til, containing all residuals from partialling out 
data.til <- data.frame(y = ytil$til, 
                       d = dtil$til, 
                       z = ztil$til, 
                       cluster = as.numeric(factor(data.share$country)), 
                       wght = data.share$w_ch)
ivfit <- ivreg(formula = y ~ d | z, 
               weights = wght, 
               data = data.til)
summ.ivfit <- summary(ivfit)
summ.ivfit #$coefficients[2,3]
ivfit.clust <- cluster.wild.ivreg(ivfit, 
                                  dat = data.til, 
                                  cluster = ~ cluster, 
                                  ci.level = 0.95, 
                                  boot.reps = 1000, 
                                  seed = seed.set)
ivfit.clust 



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
## define level of clustering standard errors 
## (not needed when using clusterSEs methods like wild cluster bootstrap)
#cluster.level <- as.numeric(factor(data.share$country)) 
# ++++
