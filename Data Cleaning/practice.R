# reading the file after manually removing the header rows in Excel
customer <- read.csv("customer.csv")

View(customer)


## ------------------ Cleaning customer df -----------------
# 1. look for duplicate values
duplicated(customer$customerID)
sum(duplicated(customer$customerID)) # 2 duplicate IDs here


# see which rows have duplicate IDs duplicate IDs
customer[which(duplicated(customer$customerID) == T), ]
customer[ which(customer$customerID %in% c("9237-HQITU", "6575-SUVOI")),]

# these seem to be repeated rows, let's delete the duplicate ones
customer <- customer[-which(duplicated(customer$customerID) == T), ]
sum(duplicated(customer$customerID)) # removed duplicate rows

# 2. Missing values
sum(is.na(customer)) # none
sapply(customer, function(x) length(which(x == ""))) # checking for blank "" values; there are none

# 3. Check individual columns
# Customer ID is already treated
# Gender
str(customer)
summary(customer$gender) # uppercase and lower problem

# let's convert everything to lower case
customer$gender <- factor(tolower(customer$gender))
summary(customer$gender)

# Senior citizen
summary(customer$SeniorCitizen)
# convert to factor
customer$SeniorCitizen <- factor(customer$SeniorCitizen)
summary(customer$SeniorCitizen)

# Partner
summary(customer$Partner) #same problem
customer$Partner <- factor(tolower(customer$Partner))
summary(customer$Partner)

# Dependents
summary(customer$Dependents) #seems okay

write.csv(customer, "customer_clean.csv")

## ------------------ Cleaning internet df -----------------

internet <- read.csv("internet.csv")
View(internet)
str(internet)

# 1. Missing column name



# First, lets change the variable name based on the observation. Google shows that dsl is a router. it
# means the column represent a internet service: thus replace it with "internet_ser"
colnames(internet)[which(colnames(internet)=="X")] <- "internet_ser"


#summarize this variable first 
summary(internet$internet_ser)

## Codesource :http://stackoverflow.com/questions/6081439/changing-column-names-of-a-data-frame-in-r


# 2. Duplicate values
sum(duplicated(internet$customerID)) # no duplicate values

# 3. Missing values
sum(is.na(internet)) #no NA



# 4.check for blanks
sapply(internet, function(x) length(which(x == ""))) # checking for blank "" values
# there are blank values - we should just replace them by NAs 
internet$MultipleLines[which(internet$MultipleLines == "")] <- NA
internet$internet_ser[which(internet$internet_ser == "")] <- NA
internet$OnlineSecurity[which(internet$OnlineSecurity == "")] <- NA
internet$OnlineBackup[which(internet$OnlineBackup == "")] <- NA
internet$DeviceProtection[which(internet$DeviceProtection == "")] <- NA
internet$TechSupport[which(internet$TechSupport == "")] <- NA
internet$StreamingTV[which(internet$StreamingTV == "")] <- NA
internet$StreamingMovies[which(internet$StreamingMovies == "")] <- NA

sapply(internet, function(x) length(which(x == ""))) 
sapply(internet, function(x) length(which(is.na(x))))
       
# 3. Check individual columns
summary(internet)

# Cust ID seems okay
summary(internet$MultipleLines) # No and No phone service are not the same
# readjusting the levels

?levels
internet$MultipleLines <- as.character(internet$MultipleLines)
internet$MultipleLines <- factor(internet$MultipleLines, levels=c("No", "Yes", NA))


# Online Security
summary(internet$OnlineSecurity)

# Online backup
summary(internet$OnlineBackup)

# device protection
summary(internet$DeviceProtection)

# tech support
summary(internet$TechSupport)

# streaming TV
summary(internet$StreamingTV)

# streaming Movies
summary(internet$StreamingMovies)

summary(internet) # looks okay now


## ------------------ Cleaning churn df (after removing the header & summary rows )-----------------
churn <- read.csv("churn.csv")
View(churn)

# deduplicate
sum(duplicated(churn$customerID)) # 2 duplicate values

# see which rows have duplicate IDs 
churn[which(duplicated(churn$customerID) == T), ]
churn[which(churn$customerID %in% c("6234-RAAPL", "4629-NRXKX")),]

# Two customer Ids are repeated, but one is not an identical repetition (both rows have different
# values) - hence, we delete those 2 rows with ID 4629-NRXKX
churn <- churn[-which(churn$customerID == "4629-NRXKX"),]

# and remove the row with dupliacte ID 6234-RAAPL
churn <- churn[-which(duplicated(churn$customerID) == T),]

churn[which(duplicated(churn$customerID) == T), ] # no duplicates now

# 2. Missing values
sum(is.na(churn)) # none
sapply(churn, function(x) length(which(x == ""))) # checking for blank "" values; there are none

# replace blank rows by NA
churn$PhoneService[which(churn$PhoneService == "")] <- NA
churn$Contract[which(churn$Contract == "")] <- NA
churn$PaperlessBilling[which(churn$PaperlessBilling == "")] <- NA
churn$PaymentMethod[which(churn$PaymentMethod == "")] <- NA
churn$Churn[which(churn$Churn == "")] <- NA

sapply(churn, function(x) length(which(x == ""))) # checking for blank "" values; there are none now

sum(is.na(churn))

# 3. Check individual columns
summary(churn)

# contract
summary(churn$Contract)
churn$Contract <- factor(tolower(churn$Contract))

# Paymet method
summary(churn$PaymentMethod)
churn$PaymentMethod <- factor(tolower(churn$PaymentMethod))

churn$PaymentMethod[which(churn$PaymentMethod == "bank trans")] <- "bank transfer (automatic)"


# Monthly Charges should not have the $ at the end
library(stringr)
churn$MonthlyCharges <- str_replace_all(churn$MonthlyCharges, "[$]", "")
churn$MonthlyCharges <- as.numeric(churn$MonthlyCharges)

summary(churn$TotalCharges)
sum(is.na(churn$TotalCharges))
which(churn$TotalCharges == "")

churn$TotalCharges <- str_replace_all(churn$TotalCharges, "[$]", "")
churn$TotalCharges <- round(as.numeric(churn$TotalCharges), 2) # introduces NAs! why?


summary(churn)
