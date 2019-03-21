# SVM

  - SVMs are linear models that require numeric attributes. 
  - In case the attributes are non-numeric, you need to convert them to a numeric form in the data preparation stage.
  
  - Hyperplane:
    - A line that is used to classify one class from another is also called a hyperplane. 
    - In fact, it is the model you're trying to build  
    - A positive value (blue points in the plot above) would mean that the set of values of the features is in one class; 
    - however, a negative value (red points in the plot above) would imply it belongs to the other class. 
    - A value of zero would imply that the point lies on the line (hyperplane) because any point on the line will satisfy the equation: 
      - W0 + W1*X1 + W2*X2 = 0
