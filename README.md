# Double-Machine-Learning-and-IV-Estimation
This dissertation project combines methods from **machine learning** and **econometrics** to conduct **causal inference**. 

Specifically, this code implements Chernozhukov et al.'s (2017) Double Machine Learning (DML) technique and instrumental variables (IV) regression to analyze heterogeneity in the parental health returns to children's education. 

This implementation of the DML approach utilizes the partialling out concept based on the Frisch-Waugh-Lovell (FWL) theorem, also known as decomposition theorem. 

## Machine Learning 
The code allows to compare a range of different machine learning methods, including regularized regression (Lasso, Ridge, Elastic Net: glmnet), random forest (ranger), and gradient boosting (XGBoost). 

## Econometrics 
After learning the nuisance parameters for bias reduction, this code implements IV estimation for causal inference. 
The code allows to compute heteroskedasticity robust and clustered standard errors and additionally applies the wild cluster bootstrap procedure (Cameron et al. 2008). 


# References 
Cameron, A. C., J. B. Gelbach, and D. L. Miller. 2008. [Bootstrap-Based Improvements for Inference with Clustered Errors.](https://www.mitpressjournals.org/doi/pdf/10.1162/rest.90.3.414) *The Review of Economics and Statistics*, 90(3), 414-427.

Chen, T., and C. Guestrin. 2016. [XGBoost: A Scalable Tree Boosting System.](https://arxiv.org/abs/1603.02754) *22nd SIGKDD Conference on Knowledge Discovery and Data Mining*.

Chernozhukov, V., D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and J. Robins. 2018. [Double/debiased machine learning for treatment and structural parameters.](https://onlinelibrary.wiley.com/doi/full/10.1111/ectj.12097) *The Econometrics Journal*, 21(1), C1-C68. 

Friedman, J., T. Hastie, and R. Tibshirani. 2008. [Regularization Paths for Generalized Linear Models via Coordinate Descent.](https://www.jstatsoft.org/article/view/v033i01) *Journal of Statistical Software*, 33(1), 1-22. 

Frisch, R., and F. V. Waugh. 1933. [Partial time regressions as compared with individual trends.](http://www.jstor.org/stable/1907330) *Econometrica*, 1, 387-401.

Wright, M. N., and A. Ziegler. 2017. [ranger: A fast implementation of random forests for high dimensional data in C++ and R.](https://doi.org/10.18637/jss.v077.i01) *Journal of Statistical Software*, 77, 1-17.
