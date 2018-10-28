#Multiple Linear Regression
advertisement<-read.csv("advertising.csv")

#Examine the data
View(advertisement)

# Check the structure of "advertisement"
str(advertisement)

# Now first build the linear model using lm() and store it into a object "model_1"
model_1 <- lm(Sales~.,data=advertisement)

#Now, check the summary of your first model i.e model_1
summary(model_1)

# Compare simple model with Newspaper with the MLR model:
# Create a Simple Linear Regression model where Sales is the independent variable and Newspaper the only independent variable
news_model <- lm(Sales~Newspaper, advertisement)

# Check the summary of both the models
summary(news_model)
summary(model_1)

# Look at the correlations between the independent variables.
corrs = cor(advertisement[, -4])
View(round(corrs, 2))

# Remove the Newspaper variable, as it was insignificant in the multiple variable model
# Store the new linear model having TV and Radio marketing as the two significant variables into the object "model_2" .
model_2 <- lm(Sales~.-Newspaper,data=advertisement)

# Great! Check the summary of model_2
summary(model_2)

