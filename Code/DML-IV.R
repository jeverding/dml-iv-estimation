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
# - Estimation of various standard errors/confidence intervals for inference, e.g. wild cluster robust 
# - Implement support for choosing regression weights 
# - Check data type of outcome and adjust fam.glmnet; binary: binomial; else: gaussian 
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
set.seed(seed.set)


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
  # Setting up the table 
  columns <- c("MSE", "lambda", "alpha", "mtry", "ntree")  
  table.mse <- data.frame(matrix(nrow=0, ncol= length(columns)))
  colnames(table.mse) <- columns
  
  # Sample splitting 80-20
  train <- sample(x = 1:nrow(x), size = dim(x)[1]*0.8) 
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
  # Tuning part using rfcv from randomForest 
  rf.cv <- rfcv(x.train, y.train, cv.fold = 10, tree= ntree.set, scale = "log", step= 0.5)
  # fitting the random forest using the best mtry
  rf <- randomForest(x.train, y.train, ntree = n.tree, mtry = rf.cv$n.var[which.min(rf.cv$error.cv)])
  pred <- predict(rf, newdata = x.test, type="response")
  
  # new NA row for random forest, fill again sequentially 
  table.mse[dim(table.mse)[1]+1,] <- rep(NA,length(columns)) 
  rownames(table.mse)[dim(table.mse)[1]]  <- "Random Forest" 
  table.mse$MSE[dim(table.mse)[1]] <- mean((pred - y.test)^2) 
  table.mse$mtry[dim(table.mse)[1]] <- rf.cv$n.var[which.min(rf.cv$error.cv)] 
  table.mse$ntree[dim(table.mse)[1]]<- ntree.set 

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
    y.hat <- predict(best.mod, newdata = x) #data.frame(y, x)) 
  }
  
  ytil <- (y - y.hat) 
  return(list("til" = ytil, 
              "table.mse" = table.mse))
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

ytil <- Partial.out(y, x)
dtil <- Partial.out(d, x)
ztil <- Partial.out(z, x)
