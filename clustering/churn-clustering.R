churn_data_complete <- read.csv("~/Upgrad/DataScience/Course3/Module4/churn_data_complete.csv")
churn_data <- churn_data_complete[, c("change_mou", "drop_vce_Mean", "custcare_Mean", "avgmou" )]
churn_data <- na.omit(churn_data)

box <- boxplot.stats(churn_data$change_mou)
out <- box$out

churn_data1 <- churn_data[ !churn_data$change_mou %in% out, ]

churn_data <- churn_data1

box <- boxplot.stats(churn_data$drop_vce_Mean)
out <- box$out

churn_data1 <- churn_data[ !churn_data$drop_vce_Mean %in% out, ]

churn_data <- churn_data1

box <- boxplot.stats(churn_data$custcare_Mean)
out <- box$out

churn_data1 <- churn_data[ !churn_data$custcare_Mean %in% out, ]

churn_data <- churn_data1

box <- boxplot.stats(churn_data$avgmou)
out <- box$out

churn_data1 <- churn_data[ !churn_data$avgmou %in% out, ]

churn_data <- churn_data1

churn_data$change_mou <- scale(churn_data$change_mou)
churn_data$drop_vce_Mean <- scale(churn_data$drop_vce_Mean)
churn_data$custcare_Mean <- scale(churn_data$custcare_Mean)
churn_data$avgmou <- scale(churn_data$avgmou)

r_sq<- rnorm(20)

for (number in 1:20){
  clus <- kmeans(churn_data, centers = number, nstart = 50)
  r_sq[number]<- clus$betweenss/clus$totss
}

plot(r_sq)

churn_data_dist<- dist(churn_data)

churn_data_hclust<- hclust(churn_data_dist, method="complete")
plot(churn_data_hclust)

rect.hclust(churn_data_hclust, k=4, border="red")

clusterCut <- cutree(churn_data_hclust, k=4)

library(cluster)
churn_data_pam = pam(churn_data_dist,4)
plot(churn_data_pam)
