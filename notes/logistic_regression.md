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
    
  - Gain and Lift

  - KS statistic 
