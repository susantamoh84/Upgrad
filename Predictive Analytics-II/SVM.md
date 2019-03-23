# SVM

  - SVMs are linear models that require numeric attributes. 
  - In case the attributes are non-numeric, you need to convert them to a numeric form in the data preparation stage.
  
# Hyperplane:
  - A line that is used to classify one class from another is also called a hyperplane. 
  - In fact, it is the model you're trying to build  
  - A positive value (blue points in the plot above) would mean that the set of values of the features is in one class; 
  - however, a negative value (red points in the plot above) would imply it belongs to the other class. 
  - A value of zero would imply that the point lies on the line (hyperplane) because any point on the line will satisfy the equation: 
    - W0 + W1*X1 + W2*X2 = 0

# Maximal Margin Classifier:
  - This is the hyperplace having maximum margin from both of the sides. Following are the 2 constrains for maximizing the margin:
    - Standardization of coefficients such that Sum(Wi^2) = 1
    - Li * (Wi*Yi) >= M
      - label 1, -1
      - Wi = coefficients of attributes
      - Yi = data points of all attributes in each row
        
# The Soft Margin Classifier
  - To summarise, support vectors are the points that lie close to the hyperplane. In fact, they are the only points that are used in constructing the hyperplane.
  - The hyperplane that allows certain points to be deliberately misclassified is also called the Support Vector Classifier
  - Advantages:
    - Not-sensitive to the slightest movement of the nearby points
    - Can separate non-linearly separable point
    - Hard-margin classifer doesn't allow any mis-classification whereas the soft-margin classifier allows mis-classification.

# The Slack Variable

  - Each data point has a slack value associated to it, according to where the point is located.
  - The value of slack lies between 0 and +infinity.
  - Lower values of slack are better than higher values (slack = 0 implies a correct classification, but slack > 1 implies an incorrect classification, whereas slack within 0 and 1 classifies correctly but violates the margin).
  
# Cost of Misclassification

  - misclassification can be controlled by the value of cost or C. 
  - If C is large, the slack variables (epsilons( ϵ)) can be large, i.e. you allow a larger number of data points to be misclassified or violate the margin; 
  - and if C is small, you force the individual slack variables to be small, i.e. you do not allow many data points to fall on the wrong side of the margin or the hyperplane
