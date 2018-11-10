cars = read.delim('cars.tab',stringsAsFactors=FALSE)
head(cars)

cars.use = cars[,-c(1,2)]
colSums(is.na(cars.use))

medians = apply(cars.use,2,median)
mads = apply(cars.use,2,mad)
cars.use = scale(cars.use,center=medians,scale=mads)
View(cars.use)
summary(cars.use)

cars.dist = dist(cars.use)

cars.hclust = hclust(cars.dist)

groups.3 = cutree(cars.hclust,3)

table(groups.3)

counts = sapply(2:6,function(ncl)table(cutree(cars.hclust,ncl)))
names(counts) = 2:6
counts

cars$Car[groups.3 == 1]

sapply(unique(groups.3),function(g)cars$Car[groups.3 == g])

groups.4 = cutree(cars.hclust,4)
sapply(unique(groups.4),function(g)cars$Car[groups.4 == g])

table(groups.3,cars$Country)

aggregate(cars.use,list(groups.3),median)
aggregate(cars[,-c(1,2)],list(groups.3),median)

a3 = aggregate(cars[,-c(1,2)],list(groups.3),median)
data.frame(Cluster=a3[,1],Freq=as.vector(table(groups.3)),a3[,-1])

library(cluster)
cars.pam = pam(cars.dist,3)

cars$Car[groups.3 != cars.pam$clustering]

plot(cars.pam)

plot(silhouette(cutree(cars.hclust,4),cars.dist))

#Average Silhouette Width:
#Range of SC	Interpretation
#0.71-1.0	A strong structure has been found
#0.51-0.70	A reasonable structure has been found
#0.26-0.50	The structure is weak and could be artificial
#< 0.25	No substantial structure has been found
