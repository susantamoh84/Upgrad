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
    
# R-square

  - Splitting is done such that R-square of the partitions obtained is greater than the R-square of the original dataset.
  - This is used for regression in decision trees
  
# Summary

  - So, the following steps are involved in decision tree construction:

  - 1. A decision tree first decides on an attribute to split on.
  - 2. To select this attribute, it measures the homogeneity of the nodes before and after the split.
  - 3. There are various ways in which you can measure the homogeneity.
  - 4. You have the Gini index and information gain for classification; you also have R2
  - 5. for the regression, to measure the homogeneity.
  - 6. The attribute that results in a maximum homogeneous data set is then selected for splitting.
  - 7. Then, this whole cycle is repeated till you get a sufficiently homogeneous data set.

# Truncation & Pruning

  - Advantages:
    - Prediction made by decision trees are easily interpretable
    - It doesn't assume anything about the dataset and can handle any kind of data - numeric, categorical, boolean etc
    - Doesn't require normalization as it compares values within the attribute
    - Gives us idea of the relative importance of the explanatory attributes which are required for prediction
    
  - Dis-advantages:
    - Tend to overfit the data
    - Tend to be very unstable which is the implication of overfitting
    
# Tree Truncation

  - Truncation: Stop the tree while it is still growing, so that it may not end up with very few datapoints.
  - Pruning: Let the tree grow to any complexity and then cut the branches of the tree in a bottom-up fashion starting from the leaves.
    - Practically its more common to use pruning to avoid overfitting of trees.
  
  - Various ways to truncate the tree:
    - 1. Minsplit - This is the minimum size of partition for splitting. If you set the minsplit = 30, any node with less than 30 observations will not be split further.
    - 2. Cp (complexity parameter) - This is the minimum percentage reduction in the error after a split, i.e. the error (which is ‘1 - accuracy’) should reduce by at least the cp; otherwise, the split will not be attempted. 
      - For example, if you set the cp = 0.01 (the cp is a fraction), a node will be split only if the error after the split reduces by at least 1%. Thus, the splits that only marginally improve the model (say, reduce the error only by 0.01%) will not be attempted.
    - 3. Maxdepth - This is the maximum depth to which a tree can grow.
    - 4. Minbucket - This is the minimum number of observations a terminal node (i.e. leaf) should have
    
# Tree Pruning

  - You check the performance of a pruned tree on a validation set. 
  - If the accuracy of the pruned tree is higher than the accuracy of the original tree (on the validation set), then you keep that branch chopped
  
  
