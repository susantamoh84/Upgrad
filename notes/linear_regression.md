# Simple Linear Regression

  - Regression: The output variable to be predicted is a continuous variable 
  - Classification: The output variable to be predicted is a categorical variable
  - Clustering: No pre-defined notion of label allocated to groups/clusters formed
  
  - Supervised learning methods
    - Past data with labels is used for building the model
    - Regression and classification algorithms fall under this category
    - The past data is divided into training and testing data sets for building the model
    
  - Unsupervised learning methods
    - No pre-defined labels are assigned to past data
    - Clustering algorithms fall under this category
    
  - Regression Line
    - The independent variable is also known as the predictor variable. And the dependent variables are also known as the output variables
    - Scatter plot - relationship between the predictor and output variables
    
    - Algorithim
      - Find the residuals and RSS for any given line through the scatter plot
      - Then you found the equation of the best-fit line by minimising the RSS and found the optimal value of β₀ and β₁
      - In mathematical terms, the algorithim is trying minimize ( sometimes maximize ) the cost function
      - gradient descent is the methodology by which the cost function is optimized

  - ESS - explained sum of squares  - SUM[i] (yhat - ymean)^2 ; yhat = ith prediction of the response variable
  - TSS - total sum of squares      - SUM[i] (yi - ymean)^2   ; yi = ith actual value of the response variable
  - RSS - residual sum of squares   - SUM[i] (yi - yhat)^2
  - TSS = ESS + RSS
  - Rsquare = ESS/TSS ( explained sum of squares / total sum of squares )
  -         = 1 - RSS/TSS
  
  - RSE ( Residual Square Error )  = SQRT( RSS/dF ) ; dF = n-2 ; n --->  number of data points
  
  - For Test Set:
    - R = cor( test$Output, test$Actual )
    - Rsquare = cor ^2
    
# Multiple Linear Regression

  - Regression using multiple independent variables
  - Model Building Steps Sample:
    - Build a primary model "model_1" taking into consideration all the independent variables
    - Analyse the summary of "model_1".
    - Remove the insignificant variable on the basis of the p-value. The p-value should be <0.05 for a variable to be significant.
    - Build the final model with the remaining variables.
    
  - Dummy Variables:
    - Turning Categorical variables into list of dummy variables ( having 0/1 values ) ---> best way to handle the categorical variables
    - Rsquare always increases with increase in number of variables
    - Adjusted Rsquare decreases with number of variables if those variables are not significant
    - Adjusted Rsquare is a better metric of model performance

  - Multi-colinearity
    - Colinearity between multiple independent variable makes it difficult of assess the effect of the individual predictors
    - A variable with a high VIF means it can be largely explained by other independent variables
    - Thus, you have to check and remove variables with a high VIF after checking for p-values
    - It doesn't affect the Adjusted R-square of the model, only coeefficients are exaggerated or reduced.
    
  - Variable Selection Method
    - stepAIC
    
  - Prediction vs Projection
    - linear regression is the process of interpolation the results
    - projection or extrapolation may not be accurate
    
