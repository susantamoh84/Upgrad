# Univariate Logistic Regression

  - Binary classification
    - output variable has 2 outcomes

  - Sigmoid function
    - Probability = 1/(1+exp(-(b0+b1x)))

  - Likelihood function
    - (1-P1)(1-P2)(1-P3)(1-P4)(1-P6)P5P7P8P9P10
      - 1-P1 for those points where label is 0
      - P5 for those points where label is 1.
      - Logistic regression tries to maximize the Likelihood function

  - Building a logistic regression model in R
    - glm ( general linear model )

  - Odds and log odds
    - For sigmoid P = 1/(1+exp(-(b0+b1x)))
    -             Ln(P/(1-p))  = b0+b1x
    - Log of odds has linear relationship with x

# Multi-variate Logistic Regression

  - Recall that, for continuous variables, the scale command is used to standardise 
  - What the scale command basically does is — it converts values to the z-scores.
  - In glm if the coeeficients of variable is NA, those can be removed.
    - Step wise variable selection: 
      - stepAIC is used to find the relevant variables
    - Backward variable selection:
      - using VIF, p-values the variable selections can be performed
    
# Model Evaluation

  - Model performance metrics:
  - Accuracy
    - Is measured the correct prediction % in the test data
    - Confusion matrix = predicted vs actual yes/no

  - Sensitivity and Specificity

    - True Negatives (TN) are actual negatives, correctly predicted as negatives
    - False Negatives (FN) are actual positives, incorrectly predicted as negatives
    - True Positives (TP) are actual positives, correctly predicted as positives
    - False Positives (FP) are actual negatives, incorrectly predicted as positives
    -  | Actual      | Predicted
    -  |             |   No           |     Yes         |
    -  | No          | TRUE Negative  | FALSE Positives |      
    -  | Yes         | FALSE Negative | TRUE Positives  |
      
    - Sensitivity = Proportions of Yes values are correct
    -             = Number of Yes's correctly predicted / Total Number of actual Yes's
    - Sensitivity = True Positive Rate
    -             = TP/(TP+FN)
    - Specificity = Proportions of No values are correct
    -             = Number of No's correctly predicted / Total Number of actual No's      
    - Specificity = True Negative Rate
    -             = TN/(TN+FP)
    - False Positive Rate = FP/(FP+TN)
    -                     = 1 - True Negative Rate
    - False Negative Rate = FN/(TP+FN)
    -                     = 1 - True Positive Rate
    - Accuracy = (TP+FN)/(TN+FP+FN+TP)
    
    - Based on the cut-off the accuracy, sensitivity, specificity values fluctuate 
      - There is a single optimum cut-off point where all the above 3 values obtain maximum value without sacrificing the value of the other 2
  
    - Discriminative Power of a model - accuracy, sensitivity, specificity
    
  - Gain and Lift
    - Lift = Gain from current model / Gain from random model
    - Gain
      - Divide the data into 10 deciles sorting by their probabilities ( desc )
      - In each decile find the following
        - Cumulative total responses
        - Cumulative total positive responses
        - Gain = Cumulative % of the total positive responses till now 
        -      = Cumulative total positive responses upto that decile / Total positive responses

  - KS statistic 
    - Similar tp Gain chart find the Cumulative % of the total negative responses till now
    - Then find KS stats = Cum % positive - Cum % negative in each decile
    - A good model is one for which the KS statistic:
      - is equal to 40% or more
      - lies in the top deciles, i.e. 1st, 2nd, 3rd or 4th

# Logistic Regression - Industrial Demo

  - there are major errors you should be on the lookout for while selecting a sample. These include: 
    - Cyclical or seasonal fluctuations in the business that need to be taken care of while building the samples. E.g. Diwali sales, economic ups and downs, etc.
    - The sample should be representative of the population on which the model will be applied in the future.
    - For rare events samples, the sample should be balanced before it is used for modelling.
    
  - Segmentation:
    - Can help build different models for different segments
    - students vs salaried
    
  - Variable Transformations
    - Dummary variables - makes model more stable ( compared to treating it as a numerical variable )
      - disadvantage is data clumping - a lot of records having similar values
    - Weight Of Evidence
      - WOE = ln( Good in bucket/Total Good) - ln( Bad in bucket/Total Bad)
      -     = ln( % Good in bucket / % Bad in the bucket )
      - WOE has to be monotonic
      - If not then coarse binning - merging buckets takes places till it becomes monotonic
      - IV = WOE * ( % of Good in bucket - % bad in bucket )
        - Predictive power of the model
      
# Industrial Demo - Model Evaluation

  - ROC Curve = % of Bad (x) vs % of Good (y)
  - Gini = Area Under the ROC Curve 
  - Validations
    - IN-sample validation : Train & Test data
    - Out-of-time validations
    - K-fold cross validations
  - Obviously, a good model will be stable. A model is considered stable if it has:
    - Performance Stability: Results of in-sample validation approximately match those of out-of-time validation
    - Variable Stability: The sample used for model building hasn't changed too much and has the same general characteristics
  - Tracking Performance Overtime
    - Recalibration - first time performance drops ---> recalibration minor correction
    - Rebuilding - after first time the performance drops
