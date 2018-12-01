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
