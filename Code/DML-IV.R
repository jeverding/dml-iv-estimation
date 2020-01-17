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


# Setup ------------------------------------------------------------------------------------------------------
# Set working directory using main project path 
main_dir <- getwd() 
code_dir <- file.path(main_dir,"Code")
data_dir <- file.path(main_dir,"Data")
output_dir <- file.path(main_dir,"Output")
# Set seed 
seed.set <- 180911 


# Functions --------------------------------------------------------------------------------------------------
# DML for PLIVM
DML2.for.PLIVM <- function(x, d, z, y, dreg, yreg, zreg, nfold=2) {
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
data.share <- read.dta13(file.path(data_dir,"share_rel6-1-1_data3.dta")) #data.share <- read.dta13("C:\\Users\\Jakob.Everding\\Desktop\\170512_Offline Arbeit\\05_Spillovers and Health of Unemployed\\03_Stata\\01_Datasets\\03_SHARE\\share_rel6-1-1_data3.dta")

# Some remaining pre-processing (select relevant variables, drop missings, recode variables)
var.select <- c("eurodcat", "bmi", "chyrseduc", "t_compschool", "sex", "chsex", "alone", "wwar", "int_year", "chmarried", "chdivorce", "chwidow", "ch200km", "chclose", "chlescontct", "chmuccontct", "ch007_", "ch014_", "ch016_", "married", "divorce", "widow", "partnerinhh", "ep005_", "agemonth", "hhsize", "yrseduc", "chnchild", "chchbyear")
data.share <- 
  data.share %>% 
  select(var.select) %>% 
  na.omit() %>% 
  mutate(eurodcat = as.numeric(eurodcat))

# Assign outcome y, treatment d, and instrumental variable z  
y= as.matrix(data.share[,"eurodcat"]) 
d= as.matrix(data.share[,"chyrseduc"]) 
z= as.matrix(data.share[,"t_compschool"]) 

# Implement machine learning methods to get residuals --------------------------------------------------------
# Code up model for regularized regression methods 
x <- model.matrix(~(sex + chsex + alone + wwar + factor(int_year) + chmarried + chdivorce + chwidow + ch200km + chclose + chlescontct + chmuccontct + factor(ch007_) + factor(ch014_) + factor(ch016_) + married + divorce + widow + partnerinhh + factor(ep005_) + poly(agemonth,4) + poly(hhsize,4) + poly(yrseduc,4) + poly(chnchild,4) + chchbyear)^2, 
                  data=data.share)

# 1) Lasso 
dreg <- function(x,d){ rlasso(x, d) } 
yreg <- function(x,y){ rlasso(x, y) } 
zreg <- function(x,z){ rlasso(x, z) } 

# Run DML algorithm 
DML2.lasso <- DML2.for.PLIVM(x, d, z, y, dreg, yreg, zreg, nfold=2)
