# Hierarchical clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')

set.seed(123)

X <- dataset[4:5]

# Denrogram
dendrogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')

plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Eucliedean Distances')

# Fitting
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(tree = hc, k = 5)

# Visualising the clusters
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of clients'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')