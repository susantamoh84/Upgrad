# Principles Of Model Selection

  - Occam's razor
    - Occam's razor is perhaps the most important thumb rule in machine learning, and incredibly 'simple' at the same time. When in dilemma, choose the simpler model.
      - A simpler model is usually more generic than a complex model. This becomes important because generic models are bound to perform better on unseen datasets.
      - A simpler model requires less training data points. This becomes extremely important because in many cases one has to work with limited data points.
      - A simple model is more robust and does not change significantly if the training data points undergo small changes.
      - A simple model may make more errors in the training phase but it is bound to outperform complex models when it sees new data. This happens because of overfitting.
      
# Bias-Variance Tradeoff

  - We considered the example of a model memorizing the entire training dataset. If you change the dataset a little, this model will need to change drastically. The model is, therefore, unstable and sensitive to changes in training data, and this is called high variance.
  - Bias quantifies how accurate the model is likely to be on future (test) data. Extremely simple models are likely to fail in predicting complex real world phenomena. Simplicity has its own disadvantages.
  
  - Although, in practice, we often cannot have a low bias and low variance model. As the model complexity goes up, the bias reduces while the variance increases, hence the trade-off.
  
# Regularization

  - Regularization is the process of deliberately simplifying models to achieve the correct balance between keeping the model simple and yet not too naive. Recall that there are a few objective ways of measuring simplicity - choice of simpler functions, lesser number of model parameters, using lower degree polynomials, etc.
  

# Hyper-parameters

  - To summarize the concept of hyperparameters:
    - Hyperparameters are used to 'fine-tune' or regularize the model so as to keep it optimally complex
    - The learning algorithm is given the hyperparameters as an 'input' and returns the model parameters as the output
    - Hyperparameters are not a part of the final model output 
    
# Validation Data 

  - Since hyperparameters need some unseen data to tune the model, validation set is used
  It prevents the learning algorithm to 'peek' at the test data while tuning the hyperparameters 
  A severe and practically frequent limitation of this approach is that data is often not abundant

# Cross Validation

  - It is a statistical technique which enables us to make extremely efficient use of available data
  It divides the data into several pieces, or 'folds', and uses each piece as test data one at a time
