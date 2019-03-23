# Decision Trees

  - A decision tree splits the data into multiple sets. Then, each of these sets is further split into subsets to arrive at a decision.
  - if a test splits the data into more than two partitions, this is called a multiway decision tree
  
  - In regression problems, a decision tree splits the data into multiple subsets. 
  - The difference between decision tree classification and decision tree regression is that in regression, 
  each leaf represents a linear regression model, as opposed to a class label.
  
  - Decision trees are easy to interpret; you can always go back and identify the various factors leading to a decision
  - A decision tree requires you to perform tests on attributes in order to split the data into multiple partitions
  - In classification, each data point in a leaf has a class label associated with it
  - There are some cases where a linear regression model cannot be used to make predictions, such as when you want to divide the 
  data set into multiple subsets because each subset has an independent trend corresponding to it. 
  There, you use a decision tree model to make predictions because a tree regression model has the capability of splitting the data 
  into multiple sets and assigning a linear regression model to each set independently.
  
# Algorithms for Decision Tree Construction

  - Always try to generate partitions that result in homogeneous data points. 
  - For classification tasks, a data set is completely homogeneous if it contains only a single class label.
  
# Gini Index

  - Gini Index(gender) = (fraction of total observations in male-node)*Gini index of male-node + (fraction of total observations in female-node)*Gini index of female-node.
    - Gini index of male-node = (Positive-label-male/Total-male)^2 + (Negative-label-male/Total-male)^2
    - Gini index of feamale-node = (Positive-label-female/Total-female)^2 + (Negative-label-female/Total-female)^2
    
  - Gini Index for attribute split is considered better if it provides higher Gini-Index compared to other attributes
  
# Entropy & Information Gain

  - Entropy of a dataset varies from values 0 to 1
  - If a dataset is completely homogenous then Entrpy = 0 ( no-disorder )
  - Entropy ε[D] = −∑(i=1 to k)Pilog2Pi
    - Probability of finding a point with the label i
  
  - Information Gain = How much entropy has decreased between the parent set and the paritions obtained after splitting
  - Let's consider an example. 
    - You have four data points out of which two belong to the class label '1', and the other two belong to the class label '2'. 
    - You split the points such that 
      - the left partition has two data points belonging to label '1', 
      - the right partition has the other two data points that belong to label '2'. 
    - Now let's assume that you split on some attribute called 'A'.
    - Entropy of original/parent data set is ε[D] = −[(2/4)log2(2/4) + (2/4)log2(2/4)] = 1.0.
    - Entropy of the partitions after splitting is ε[DA] = −[(0.5)log2(2/2) + (0.5)log2(2/2)] = 0
    - Information gain after splitting is Gain[D,A] = ε[D] - ε[DA] = 1.0
    
    
