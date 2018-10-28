housing <- read.csv("Housing.csv")
View(housing)
str(housing)

# Here we follow the same steps as we did in the previous case.

# convert factors with 2 levels to numerical variables
levels(housing$mainroad)<-c(1,0)
housing$mainroad<- as.numeric(levels(housing$mainroad))[housing$mainroad]

levels(housing$guestroom)<-c(1,0)
housing$guestroom <- as.numeric(levels(housing$guestroom))[housing$guestroom]

levels(housing$basement)<-c(1,0)
housing$basement <- as.numeric(levels(housing$basement))[housing$basement]

levels(housing$hotwaterheating)<-c(1,0) 
housing$hotwaterheating <- as.numeric(levels(housing$hotwaterheating))[housing$hotwaterheating]

levels(housing$airconditioning)<-c(1,0)
housing$airconditioning <- as.numeric(levels(housing$airconditioning))[housing$airconditioning]

levels(housing$prefarea)<-c(1,0)
housing$prefarea <- as.numeric(levels(housing$prefarea))[housing$prefarea]

 
# Create the dummy variable for furnishingstatus variable
dummy_1 <- data.frame(model.matrix( ~furnishingstatus, data = housing))
View(dummy_1)
dummy_1 <- dummy_1[,-1]

# Combine the dummy variables and the numeric columns of housing dataset, in a new dataset called housing_1
housing_1 <- cbind(housing[,-13], dummy_1)


# create derived metrics
# create a new metric area per unit bedrooms
housing_1$areaperbedroom <- housing_1$area/housing_1$bedrooms
# create a new metric bathrooms per bedrooms
housing_1$bbratio <- housing_1$bathrooms/housing_1$bedrooms


# separate training and testing data
set.seed(100)
trainindices= sample(1:nrow(housing_1), 0.7*nrow(housing_1))
train = housing_1[trainindices,]
test = housing_1[-trainindices,]


# Build model 1 containing all variables
model_1 <-lm(price~.,data=train)
summary(model_1)
#######

# Now, lets see how to use stepAIC

# In stepAIC function, we pass our first model i.e model_1 and 
# direction is ser as both, because in stepwise,  both the forward selection 
# of variables and backward elimination of variables happen simultaneously 


# Lets load the library in which stepAIC function exists
install.packages("MASS")
library(MASS)

# We have a total of 15 variables considered into the model 
#Now let;s run the code. 

step <- stepAIC(model_1, direction="both")
#Great, so many iterations have been done through the stepwise command. 
# now we need to know our model equation so lets write the Step command here. 

step
# stepAIC makes multiple calls while checking which variables to keep
# The last call that step makes, contains only the variables it considers to be important in the model. 
# some insignifican variables have been removed. 
# Now store the last model equation of stepwise method into an object called model_2
# You can notice that stepAIC removed variables - Bedrooms, furnishingstatussemi.furnished, bbratio, 

# Let's execute this model here, 
model_2 <- lm(formula = price ~ area + bathrooms + stories + mainroad + 
                guestroom + basement + hotwaterheating + airconditioning + 
                parking + prefarea + furnishingstatusunfurnished + areaperbedroom, 
              data = train)
# Let us look at the summary of the model
summary(model_2)


## Let us check for multicollinearity 
# If the VIF is above 2 or 5 as the business goal says, you would remove 
# the variables if they are statistically insignificant
vif(model_2)

# area has a VIF of 4.20 and areaperbedroom has a VIF of 4.09
# Let us check now their p values
# You can see that are variable has a very low p value, while areaperbedroom has a VIF of 0.14,
# and thus, we can remove areaperbedroom variable and run the model again

model_3 <- lm(formula = price ~ area + bathrooms + stories + mainroad + 
                guestroom + basement + hotwaterheating + airconditioning + 
                parking + prefarea + furnishingstatusunfurnished, 
              data = train)

# Let us look at the VIFs now
vif(model_3)
# All variables have a VIF below 2
# Let us check now for variables which have high p values
summary(model_3)

# Although, all variables have a p value below 0.05, the number of variables is still too large.
# You could continue removing the variables till the significance level is 0.001
# Try the rest yourself and compare the results of the two, this one and the previous model.

model_4 <- lm(formula = price ~ area + bathrooms + stories + mainroad + 
                guestroom + hotwaterheating + airconditioning + 
                parking + prefarea + furnishingstatusunfurnished, 
              data = train)
summary(model_4)

# Remove mainroad


model_4 <- lm(formula = price ~ area + bathrooms + stories + 
                guestroom + hotwaterheating + airconditioning + 
                parking + prefarea + furnishingstatusunfurnished, 
              data = train)
summary(model_4)

# Remove hotwaterheating
model_5 <- lm(formula = price ~ area + bathrooms + stories + 
                guestroom + airconditioning + 
                parking + prefarea + furnishingstatusunfurnished, 
              data = train)
summary(model_5)

# predicting the results in test dataset
Predict_1 <- predict(model_5,test[,-1])
test$test_price <- Predict_1

# Now, we need to test the r square between actual and predicted sales. 
r <- cor(test$price,test$test_price)
rsquared <- cor(test$price,test$test_price)^2
rsquared