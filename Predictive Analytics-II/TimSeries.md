# TimeSeries

  - So, the two most important differences between time series and regression are:
  
    - Time series have a strong temporal (time-based) dependence — each of these data sets essentially consists of a series of time stamped 
    observations, i.e., each observation is tied to a specific time instance. Thus, unlike regression, the order of the data is 
    important in a time series.
    
    - In a time series, you are not concerned with the causal relationship between the response and explanatory variable. 
    The cause behind the changes in the response variable is very much a black box.
    
  - Components
    - Predictable:
      - Local:
      - Global:
        - Trend
        - Seasonality
    - Un-predictable
      - Noise
      
  - Additive Model: Addition of all the above components
  - Multiplicative model: Multiplication of all the above components

- Additive vs Multiplicative:

  - When the magnitude of the seasonal pattern in the data increases with an increase in data values, and decreases with a decrease in the data values, the multiplicative model may be a better choice.

  - When the magnitude of the seasonal pattern in the data does not directly correlate with the value of the series, the additive model may be a better choice.
  
# TimeSeries Modelling

  - In general, the different steps of the time series modelling process can be summarised as follows:
    - Visualise the time series.
    - Recognise the trend and seasonal component.
    - Apply regression to model the trend and the seasonality.
    - Remove the trend and seasonal component from the series. What remains is the stationary part: a combination of the autoregressive, and white noise.
    - Model this stationary time series.
    - Combine the forecast of this model with the trend and seasonal component.
    - Find the residual series by subtracting the forecasted value from the actual observed value.
    - Check if the residual series is pure white noise.
    
  - Staionarity
    - Strongly Stationary: 
      - White noise: white noise is basically a series of values that are all independent. Typically, for noise, you can assume that all these values come from a Gaussian distribution with a zero mean.
      - if the series is white noise, the values in it will belong to a normal distribution. Hence, you can test if the series’ values belong to a normal distribution or not. For this, you’ve learnt two tests:
        - Histogram test
        - Q-Q plot test
        
    - there are a few more popular tests that can be used to understand whether a series is white noise or not. Some of them are listed here:
      - Ljung-Box (Portmanteau) test
      - Turning point test
      - Difference sign test
      - Runs test
      - Rank test
      - ACF and PACF
      
  - The only two series that are stationary in the strongest sense are 
    - White noise
    - Constant function
    
  - Weak stationarity would imply that the values of this time series depend on past values in a fixed manner. 
  - So there is local predictability of some kind that is present.
  
  - We have two basic types of time series models:
    
    - Autoregressive (AR) model
    - Moving-average model
    
    - you look for the cutoff value in the PACF plot for the most optimal p in the AR(p) model
    - the ACF plot for q in the MA(q) process
    - For the ARMA(p,q) process, you need to find the cutoff lag values from both the ACF and the PACF plots
    
    - there are two formal tests for checking the stationarity of a time series, both with opposite null and alternate hypotheses.
      - ADF test: Null hypothesis assumes that the series is not stationary
      - KPSS test: Null hypothesis assumes that the series is stationary
      
  - Finding the Best-Fit ARMA Model
    - try different ARMA (p,q) combinations and compare them, based on parameters such as log likelihood, AIC, AICc, and BIC
    - Note that in all these measures, the common pattern is
      - The higher the likelihood, the lower the measure, and
      - The higher the model complexity, the higher the measure
    - Pick a model which has high likelihood and low complexity
    
  - Model Evaluation:
    - MAPE: mean absolute percentage error
    - It is similar to R-squared but still slightly different. Some pros and cons of using the MAPE are
      - R-squared punishes big deviations very strictly compared to the MAPE.
      - The MAPE will not perform well if one of the actual data points is equal to zero.
      - It is difficult to interpret as a percentage.
      - The MAPE is known to favour models that consistently predict lower values. This may introduce bias.
      
