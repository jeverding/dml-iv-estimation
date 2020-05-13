# ========================================================================================================== #
# ========================================================================================================== # 
#
# Wild cluster bootstrap IV estimation 
#
# This script implements the wild cluster bootstrap procedure to conduct causal inference. The script 
# defines functions which allow to sequentially execute the different wcb steps and estimate instrumental 
# variables regressions. Supports weighted regressions. 
#
# ========================================================================================================== #
# ========================================================================================================== #
# Remove all objects from the workspace
rm( list=ls() )

library(plyr)
library(dplyr)
library(xtable)
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
iv.wcb <- function(data = data.share, 
                   y = data$eurodcat, 
                   d = data$chyrseduc, 
                   z = data$t_compschool, 
                   wght = data$w_ch, 
                   cluster = as.numeric(factor(data$country)), 
                   funcform = form.convcontrols) {
  model.data <- as.data.frame(model.matrix(funcform, 
                                           data=data)) 
  #y:
  y.lm <- lm(formula = y~., 
             data = model.data,
             weights = wght)
  y.lm.til <- y - predict(y.lm) 
  #d:
  d.lm <- lm(formula = d~., 
             data = model.data,
             weights = wght)
  d.lm.til <- d - predict(d.lm) 
  #z:
  z.lm <- lm(formula = z~., 
             data = model.data,
             weights = wght)
  z.lm.til <- z - predict(z.lm) 
  # iv
  simple.data.til <- data.frame(y = y.lm.til, 
                                d = d.lm.til, 
                                z = z.lm.til, 
                                cluster = cluster, 
                                wght = wght)
  simple.ivfit <- ivreg(formula = y ~ d | z, 
                        weights = wght, 
                        data = simple.data.til)
  simple.summ.ivfit <- summary(simple.ivfit)
  simple.ivfit.clust.90 <- cluster.wild.ivreg(simple.ivfit, 
                                              dat = simple.data.til, 
                                              cluster = ~ cluster, 
                                              ci.level = 0.90, 
                                              boot.reps = 1000, 
                                              impose.null = FALSE, 
                                              seed = seed.set)
  simple.ivfit.clust.95 <- cluster.wild.ivreg(simple.ivfit, 
                                              dat = simple.data.til, 
                                              cluster = ~ cluster, 
                                              ci.level = 0.95, 
                                              boot.reps = 1000, 
                                              impose.null = FALSE, 
                                              seed = seed.set)
  simple.ivfit.clust.99 <- cluster.wild.ivreg(simple.ivfit, 
                                              dat = simple.data.til, 
                                              cluster = ~ cluster, 
                                              ci.level = 0.99, 
                                              boot.reps = 1000, 
                                              impose.null = FALSE, 
                                              seed = seed.set)
  simple.ivfit.clust$ci[2,c(1:2)]
  return(list("coeff" = simple.summ.ivfit$coefficients[2,1], 
              "ci.90.low" = simple.ivfit.clust.90$ci[2,1], 
              "ci.90.high" = simple.ivfit.clust.90$ci[2,2], 
              "ci.95.low" = simple.ivfit.clust.95$ci[2,1], 
              "ci.95.high" = simple.ivfit.clust.95$ci[2,2], 
              "ci.99.low" = simple.ivfit.clust.99$ci[2,1], 
              "ci.99.high" = simple.ivfit.clust.99$ci[2,2], 
              "obs" = nrow(simple.data.til)))
}

# Prepare table 
make.tab.wcb <- function(table.name = table.iv.wcb, outcome1 = eurodcat.iv, outcome2 = eurodc3.iv, outcome3 = deprsad.iv) {
  # panel A: main sample 
  # coefficients 
  table.name[nrow(table.name)+1,1] <- format(round(as.numeric(outcome1$coeff), 3), nsmall=3, 
                                             big.mark = ",", decimal.mark = ".", 
                                             digits = 3)
  table.name[nrow(table.name),2] <- format(round(as.numeric(outcome2$coeff), 3), nsmall=3, 
                                           big.mark = ",", decimal.mark = ".", 
                                           digits = 3)
  table.name[nrow(table.name),3] <- format(round(as.numeric(outcome3$coeff), 3), nsmall=3, 
                                           big.mark = ",", decimal.mark = ".", 
                                           digits = 3)
  # ci: 90% 
  table.name[nrow(table.name)+1,1] <- paste0("[", 
                                             format(round(as.numeric(outcome1$ci.90.low), 3), nsmall=3, 
                                                    big.mark = ",", decimal.mark = ".", 
                                                    digits = 3), 
                                             ", ", 
                                             format(round(as.numeric(outcome1$ci.90.high), 3), nsmall=3, 
                                                    big.mark = ",", decimal.mark = ".", 
                                                    digits = 3), 
                                             "]")
  table.name[nrow(table.name),2] <- paste0("[", 
                                           format(round(as.numeric(outcome2$ci.90.low), 3), nsmall=3, 
                                                  big.mark = ",", decimal.mark = ".", 
                                                  digits = 3), 
                                           ", ", 
                                           format(round(as.numeric(outcome2$ci.90.high), 3), nsmall=3, 
                                                  big.mark = ",", decimal.mark = ".", 
                                                  digits = 3), 
                                           "]")
  table.name[nrow(table.name),3] <- paste0("[", 
                                           format(round(as.numeric(outcome3$ci.90.low), 3), nsmall=3, 
                                                  big.mark = ",", decimal.mark = ".", 
                                                  digits = 3), 
                                           ", ", 
                                           format(round(as.numeric(outcome3$ci.90.high), 3), nsmall=3, 
                                                  big.mark = ",", decimal.mark = ".", 
                                                  digits = 3), 
                                           "]")
  # ci: 95% 
  table.name[nrow(table.name)+1,1] <- paste0("[[", 
                                             format(round(as.numeric(outcome1$ci.95.low), 3), nsmall=3, 
                                                    big.mark = ",", decimal.mark = ".", 
                                                    digits = 3), 
                                             ", ", 
                                             format(round(as.numeric(outcome1$ci.95.high), 3), nsmall=3, 
                                                    big.mark = ",", decimal.mark = ".", 
                                                    digits = 3), 
                                             "]]")
  table.name[nrow(table.name),2] <- paste0("[[", 
                                           format(round(as.numeric(outcome2$ci.95.low), 3), nsmall=3, 
                                                  big.mark = ",", decimal.mark = ".", 
                                                  digits = 3), 
                                           ", ", 
                                           format(round(as.numeric(outcome2$ci.95.high), 3), nsmall=3, 
                                                  big.mark = ",", decimal.mark = ".", 
                                                  digits = 3), 
                                           "]]")
  table.name[nrow(table.name),3] <- paste0("[[", 
                                           format(round(as.numeric(outcome3$ci.95.low), 3), nsmall=3, 
                                                  big.mark = ",", decimal.mark = ".", 
                                                  digits = 3), 
                                           ", ", 
                                           format(round(as.numeric(outcome3$ci.95.high), 3), nsmall=3, 
                                                  big.mark = ",", decimal.mark = ".", 
                                                  digits = 3), 
                                           "]]")
  # ci: 99% 
  eurodc3.iv
  table.name[nrow(table.name)+1,1] <- paste0("[[[", 
                                             format(round(as.numeric(outcome1$ci.99.low), 3), nsmall=3, 
                                                    big.mark = ",", decimal.mark = ".", 
                                                    digits = 3), 
                                             ", ", 
                                             format(round(as.numeric(outcome1$ci.99.high), 3), nsmall=3, 
                                                    big.mark = ",", decimal.mark = ".", 
                                                    digits = 3), 
                                             "]]]")
  table.name[nrow(table.name),2] <- paste0("[[[", 
                                           format(round(as.numeric(outcome2$ci.99.low), 3), nsmall=3, 
                                                  big.mark = ",", decimal.mark = ".", 
                                                  digits = 3), 
                                           ", ", 
                                           format(round(as.numeric(outcome2$ci.99.high), 3), nsmall=3, 
                                                  big.mark = ",", decimal.mark = ".", 
                                                  digits = 3), 
                                           "]]]")
  table.name[nrow(table.name),3] <- paste0("[[[", 
                                           format(round(as.numeric(outcome3$ci.99.low), 3), nsmall=3, 
                                                  big.mark = ",", decimal.mark = ".", 
                                                  digits = 3), 
                                           ", ", 
                                           format(round(as.numeric(outcome3$ci.99.high), 3), nsmall=3, 
                                                  big.mark = ",", decimal.mark = ".", 
                                                  digits = 3), 
                                           "]]]")
  # add obs.: 
  table.name[nrow(table.name)+1,1] <- format(round(as.numeric(outcome1$obs), 3), 
                                             big.mark = ",", decimal.mark = ".", 
                                             digits = 5)
  table.name[nrow(table.name),2] <- format(round(as.numeric(outcome2$obs), 3), 
                                           big.mark = ",", decimal.mark = ".", 
                                           digits = 5)
  table.name[nrow(table.name),3] <- format(round(as.numeric(outcome3$obs), 3), 
                                           big.mark = ",", decimal.mark = ".", 
                                           digits = 5)
  return(table.name) 
}


# Start ==================================================================================================== # 
# Load data --------------------------------------------------------------------------------------------------
data.share <- read_dta(file.path(data_dir,"share_rel6-1-1_data3.dta")) 

# Some additional pre-processing (select relevant variables, drop missings, recode variables)
ctrend_1 <- paste0("trend1cntry_", 1:length(unique(data.share$country))) 
ctrend_2 <- paste0("trend2cntry_", 1:length(unique(data.share$country))) 
ctrend_3 <- paste0("trend3cntry_", 1:length(unique(data.share$country))) 
var.select <- c("deprsad", "eurodc3", "eurodcat", 
                "chyrseduc", "t_compschool", 
                "country", "chbyear", "sex", "chsex", "int_year", "agemonth", "yrseduc", 
                ctrend_1, ctrend_2, ctrend_3, 
                "normchbyear", "w_ch") 

data.share <- 
  data.share %>% 
  select(var.select) %>% 
  na.omit() %>% 
  filter(normchbyear %in% c(-10:-1,1:10), 
         !is.na(eurodcat)) %>% 
  mutate(deprsad = as.numeric(deprsad), 
         eurodc3 = as.numeric(eurodc3), 
         eurodcat = as.numeric(eurodcat)) 

# Set up model for controls 
form.convcontrols <- as.formula(paste0("~(-1 + factor(country) + factor(chbyear) + sex + chsex + factor(int_year) + agemonth + yrseduc + ", 
                                       paste(ctrend_2, collapse = " + "), 
                                       ")"))

# Code up subsamples 
data.share.fath <- 
  data.share %>% 
  filter(sex==0) 
data.share.moth <- 
  data.share %>% 
  filter(sex==1) 
data.share.sons <- 
  data.share %>% 
  filter(chsex==0) 
data.share.daug <- 
  data.share %>% 
  filter(chsex==1) 

# Run model 
# main sample 
eurodcat.iv <- iv.wcb(data = data.share, y = data.share$eurodcat)
eurodc3.iv <- iv.wcb(data = data.share, y = data.share$eurodc3)
deprsad.iv <- iv.wcb(data = data.share, y = data.share$deprsad)
# Fathers 
eurodcat.iv.fath <- iv.wcb(data = data.share.fath, y = data.share.fath$eurodcat)
eurodc3.iv.fath <- iv.wcb(data = data.share.fath, y = data.share.fath$eurodc3)
deprsad.iv.fath <- iv.wcb(data = data.share.fath, y = data.share.fath$deprsad)
# Mothers 
eurodcat.iv.moth <- iv.wcb(data = data.share.moth, y = data.share.moth$eurodcat)
eurodc3.iv.moth <- iv.wcb(data = data.share.moth, y = data.share.moth$eurodc3)
deprsad.iv.moth <- iv.wcb(data = data.share.moth, y = data.share.moth$deprsad)
# Sons 
eurodcat.iv.sons <- iv.wcb(data = data.share.sons, y = data.share.sons$eurodcat)
eurodc3.iv.sons <- iv.wcb(data = data.share.sons, y = data.share.sons$eurodc3)
deprsad.iv.sons <- iv.wcb(data = data.share.sons, y = data.share.sons$deprsad)
# Daughters 
eurodcat.iv.daug <- iv.wcb(data = data.share.daug, y = data.share.daug$eurodcat)
eurodc3.iv.daug <- iv.wcb(data = data.share.daug, y = data.share.daug$eurodc3)
deprsad.iv.daug <- iv.wcb(data = data.share.daug, y = data.share.daug$deprsad)


# Make table 
table.iv.wcb <- data.frame(matrix(nrow = 0, ncol = 3))
colnames(table.iv.wcb) <- c("EURO-D >= 5", "EURO-D >= 3", "Feeling depressed")
table.iv.wcb <- make.tab.wcb(outcome1 = eurodcat.iv, outcome2 = eurodc3.iv, outcome3 = deprsad.iv)
table.iv.wcb <- make.tab.wcb(outcome1 = eurodcat.iv.fath, outcome2 = eurodc3.iv.fath, outcome3 = deprsad.iv.fath)
table.iv.wcb <- make.tab.wcb(outcome1 = eurodcat.iv.moth, outcome2 = eurodc3.iv.moth, outcome3 = deprsad.iv.moth)
table.iv.wcb <- make.tab.wcb(outcome1 = eurodcat.iv.sons, outcome2 = eurodc3.iv.sons, outcome3 = deprsad.iv.sons)
table.iv.wcb <- make.tab.wcb(outcome1 = eurodcat.iv.daug, outcome2 = eurodc3.iv.daug, outcome3 = deprsad.iv.daug)


# Export Table 
print(xtable(table.iv.wcb, 
             caption = "Sensitivity analyses: Wild cluster bootstrap"),
      file = file.path(output_dir,"Table_wcb.tex"))
