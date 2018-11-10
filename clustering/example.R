View(species)

#Kmeans clustering
r_sq <- clus$betweenss/clus$totss
cluster_1 <- kmeans(species, 3, nstart = 20)

ggplot(species, aes(Petal.Length, Petal.Width, color = factor(cluster_1$cluster))) + geom_point() 

#Agglomerative clustering
cluster_2 <- hclust(dist(species), method="single")
plot(cluster_2)

#form rectangles around the 3 clusters
rect.hclust(cluster_2, 3, border="red")

#Cut the dendogram at height to make 3 clusters
cluster<-cutree(cluster_2, 3)

species <- cbind(species, cluster)

ggplot(species, aes(Petal.Length, Petal.Width, color=factor(species$cluster))) + geom_point() 
#the third cluster has only 1 point - not appropriate clustering here

#Use complete linkage
cluster_3 <- hclust(dist(species), method="complete")
plot(cluster_3)

rect.hclust(cluster_3, 3, border="red")

clusterid<-cutree(cluster_3, 3)

species <- cbind(species, clusterid)

ggplot(species, aes(Petal.Length, Petal.Width, color=factor(species$clusterid))) + geom_point()

#K-means accuracy
table(cluster_1$cluster, species_1$Species)

   setosa versicolor virginica
  1      0         48         4
  2      0          2        46
  3     50          0         0

#Complete linkage accuracy
table(clusterid, species_1$Species)

clusterid setosa versicolor virginica
        1     50          0         0
        2      0         21        50
        3      0         29         0
        

