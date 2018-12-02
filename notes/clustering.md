# K-means Clustering

  -  intra-segment homogeneity and inter-segment heterogeneity
  - mainly 3 types of segmentation are used for customer segmentation:
    - Behavioural segmentation: Segmentation is based on the actual patterns displayed by the consumer
    - Attitudinal segmentation: Segmentation is based on the beliefs or the intents of people, which may not translate into similar action
    - Demographic segmentation: Segmentation is based on the person’s profile and uses information such as age, gender, residence locality, income, etc.
    
  - The steps in the K-Means algorithm
    - Find the centroid: the formula (coordinates) of the centroid will be:
      - (x1+x2…..+xn / n, y1+y2…..+yn / n).
    - Find the distance of the different points to different centroids
      - assign the min distance as the cluster index
    - Continue re-assigning the cluster indexes till there is no change in cluster indexes
    - Algorithim convergence highly depends ( biased ) on the starting cluster centers

    - The clustering process is very sensitive to outliers.
    - Since the distance metric used in the clustering process is the Euclidean distance, all the attributes to be brought to the same scale. This can be achieved through standardisation.
    - The K-Means algorithm does not work with categorical data.
    - The process may not converge in the given number of iterations. You should always check for convergence.
      - Algorithim is non-deterministic

  - Data Preparation:
  
    - RFM segmentation: In RFM analysis, you look at the recency, frequency and the monetary scores of all the customers for segmentation.
      - Recency: It measures how recently you visited the store or made a purchase
      - Frequency: It measures the frequency of the transactions the customers made
      - Monetary: It measures how much the customer spent on purchases he/she made
        - Good amount of data is required atleast 1 years
    - RPI segmentation
      - Relationship - old / new customer
      - Persona - buy certain kind of stuff: gift giver etc
      - Intent - Based on your browsing pattern what is your intent at that particular time

    - CDJ segmentation ( Customer Decsion Journey )
      - Example; at different stages of customer journey the customer may decide to churn/leave
      
    - Segmentation Question always comes in the most structured fashion
      - But segmentation is just answer to one of the possible question
      - Unstructured question ------> Always go for segmentation
      
    - kmeans() function stores the output of the algorithm in a list of 9 objects.
      - cluster: This stores the cluster IDs for each data point
      - centers: This gives the location of the cluster centres
      - totss: This measures the total spread in the data by calculating the total sum of squares of distance
      - withinss: This is a measure of the within-cluster sum of squares, one component per cluster
      - tot.withinss: This gives the total within-cluster sum of squares, i.e. sum(withinss)
      - betweenss: The inter-cluster sum of squares of the distance, i.e. totss-tot.withinss
      - size: The number of points in each cluster
      - iter: The number of (outer) iterations
      - ifault: integer indicator of a possible algorithm problem – for experts
      
    - different ways to decide upon the required number of clusters that you want.

      - The business understanding or the requirement can guide you, or rather constrain you

      - You can use the elbow method to arrive at a range of the values of K for which you can get optimal clusters, i.e. clusters with the maximum inter-cluster variance and minimum intra-cluster variance. For this, you can use any of the 2 metrics:

        - R-sq value which is the ratio of (betweenss/totss)
        - Pseudo F statistic, which is given by (between-cluster-sum-of-squares / (c-1)) / (within-cluster-sum-of-squares / (n-c)) where c is the number of clusters and n is the total number of data points
        

# Hierarchical clustering
  - Algorithim
    - Given a set of N items to be clustered, the steps in hierarchical clustering are:
    - Calculate the NxN distance (similarity) matrix, which calculates the distance of each data point from the other
    - Each item is first assigned to its own cluster, i.e. N clusters are formed
    - The clusters which are closest to each other are merged to form a single cluster
    - The same step of computing the distance and merging the closest clusters is repeated till all the points become part of a single cluster
    
  - Interpreting the dendrogram
    - Aglomoreative clustering
    - Divisive clustering
    
  - Types of linkages
    - Single Linkage: Here, the distance between 2 clusters is defined as the shortest distance between points in the two clusters
    - Complete Linkage: Here, the distance between 2 clusters is defined as the maximum distance between any 2 points in the clusters
    - Average Linkage: Here, the distance between 2 clusters is defined as the average distance between every point of one cluster to every other point of the other cluster.

  - Depending on how much clusters need to be kept, the dendogram can be cut a suitable height
