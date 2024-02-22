#install.packages("kernlab")
#install.packages("uwot")
library(mclust)
library(kernlab)
library(ggplot2)
library(uwot)
library(dbscan)
###########################################################################
#  S-sets (s_3)
##########################################################################
setwd("D:/Third Year/Data Science II/Option 2 Clustering/S-sets")
getwd()

data <- read.table("s3.txt")

set.seed(6)
#Gaussian Mixture Model (GMM)
model <- Mclust(data, G = 15)


# Plotting the results
plot(model, what="classification")

# Extracting the centroids
centroids <- model$parameters$mean
print(centroids)

#predicted labels
model_label <- model$classification
length(model_label)

#Metric
#Adjusted Rand Index (ARI)
setwd("D:/Third Year/Data Science II/Option 2 Clustering/S-sets/Ground truth centroids and partitions")
getwd()

label <- read.table("s3-label.txt")
label_v <- label[, 1]
length(label_v)


ari_value1 <- adjustedRandIndex(label_v, model_label)

# Print the results
#cat("Adjusted Rand Index (ARI):", ari_value, "\n")

############
S-sets (s_4)
#############
setwd("D:/Third Year/Data Science II/Option 2 Clustering/S-sets")
getwd()
#update.packages()

data1 <- read.table("s4.txt")

set.seed(3)
#Gaussian Mixture Model (GMM)
model1 <- Mclust(data1, G = 15)

#Summary of the model
#summary(model1)

# Plotting the results
plot(model1, what="classification")

# Extracting the centroids
centroids1 <- model1$parameters$mean

# Printing the centroids
print(centroids1)

model_label1 <- model1$classification
length(model_label1)

#Metric
setwd("D:/Third Year/Data Science II/Option 2 Clustering/S-sets/Ground truth centroids and partitions")
getwd()

label1 <- read.table("s4-label.txt")
label_v1 <- label1[, 1]
length(label_v1)

ari_value2 <- adjustedRandIndex(label_v1, model_label1)

# Print the results
#cat("Adjusted Rand Index (ARI):", ari_value1, "\n")
##############################################################

                         #A-sets
#############################################################

setwd("D:/Third Year/Data Science II/Option 2 Clustering/A-sets")
getwd()
#update.packages()

dat <- read.table("a3.txt")

set.seed(1)
#Gaussian Mixture Model (GMM)
mod <- Mclust(dat, G = 50)

#Summary of the model
#summary(model1)

# Plotting the results
plot(mod, what="classification")

# Extracting the centroids
centr <- mod$parameters$mean

# Printing the centroids
print(centr)

mod_label <- mod$classification
length(mod_label)


data <- mod$dat                           
data <- data.frame(V1 = dat[,1], V2 = dat[,2], Cluster = as.factor(mod_label))

#Nunber of clusters
num_clusters <- length(unique(data$Cluster))
colors <- grDevices::rainbow(num_clusters)

# Create the plot
ggplot(data, aes(x = V1, y = V2, color = Cluster)) +
  geom_point(alpha = 0.6, size = 1.5) +  # Adjust alpha and size as needed
  scale_color_manual(values = colors) +
  theme_minimal() +
  ggtitle("Classification Results for A-sets")

#Metric
setwd("D:/Third Year/Data Science II/Option 2 Clustering/A-sets/Ground truth centroids")
getwd()

lab <- read.table("a3-Ground truth partitions.txt")
lab_v1 <- lab[, 1]
length(label_v1)

ari_value3 <- adjustedRandIndex(lab_v1, mod_label)

# Print the results
#cat("Adjusted Rand Index (ARI):", ari_value1, "\n")

#######################################################################

                           # Birch-sets (b_1)
#######################################################################
setwd("D:/Third Year/Data Science II/Option 2 Clustering/Birch-sets")
getwd()

data2 <- read.table("birch1.txt")

set.seed(123)  
k <- 100  
kmeans_result <- kmeans(data2, centers = k) 

data_df <- as.data.frame(data2)
data_df$cluster <- factor(kmeans_result$cluster)

# Plot
ggplot(data_df, aes(x = V1, y = V2, color = cluster)) +
  geom_point(alpha = 0.6) +
  scale_color_viridis_d() +  # Use a nice color palette
  theme_minimal() +
  labs(title = "K-Means Clustering Results", x = "Dimension 1", y = "Dimension 2")

#Metric
setwd("D:/Third Year/Data Science II/Option 2 Clustering/Birch-sets/ground truth centroids TXT and partitions PA")
getwd()

lab <- read.table("b1-gt.p.txt")
lab_v2 <- lab[, 1]
length(lab_v2)

ari_value4 <- adjustedRandIndex(lab_v2, kmeans_result$cluster)

# Print the results
#cat("Adjusted Rand Index (ARI):", ari_value2, "\n")
##########################################################################

                               # b_2

#########################################################################
setwd("D:/Third Year/Data Science II/Option 2 Clustering/Birch-sets")
getwd()


data3 <- read.table("birch2.txt")

set.seed(1)
#Gaussian Mixture Model (GMM)
model <- Mclust(data3, G = 100)


# Plotting the results
plot(model, what="classification")

# Extracting the centroids
centroids <- model$parameters$mean

#predicted label
model_label3 <- model$classification
length(model_label3)

#Metric
#Adjusted Rand Index (ARI)
setwd("D:/Third Year/Data Science II/Option 2 Clustering/Birch-sets/ground truth centroids TXT and partitions PA")
getwd()

label <- read.table("b2-gt.p.txt")
label_v <- label[, 1]
length(label_v)


ari_value5 <- adjustedRandIndex(label_v, model_label3)

# Print the results
#cat("Adjusted Rand Index (ARI):", ari_value, "\n")

#####################################################################

                              #b_3
#####################################################################

setwd("D:/Third Year/Data Science II/Option 2 Clustering/Birch-sets")
getwd()


data4 <- read.table("birch3.txt")


set.seed(1)
#Gaussian Mixture Model (GMM)
model <- Mclust(data4, G = 100)


# Plotting the results
plot(model, what="classification")


# # Apply DBSCAN
# 
# eps_val <- 0.5  
# minPts_val <- 10  
# 
# dbscan_result <- dbscan(data4, eps = eps_val, minPts = minPts_val)
# 
# # Check the number of clusters found (excluding noise)
# num_clusters_found <- max(dbscan_result$cluster)
# cat("Number of clusters found:", num_clusters_found, "\n")
# 
# 
# 
# 
# # Assuming 'data' contains your coordinates and 'dbscan_result' is the result from dbscan()
# plot(data4[,1], data4[,2], col = dbscan_result$cluster + 1L, pch = 20, asp = 1,
#      xlab = "Feature 1", ylab = "Feature 2", main = "DBSCAN Clustering")
# 
# # Optionally, add a legend if needed
# legend("topright", legend = c("Noise", paste("Cluster", unique(dbscan_result$cluster[dbscan_result$cluster > 0]))),
#        col = 1 + unique(dbscan_result$cluster), pch = 20)
# 
# 
# 
# #Metric
# #Adjusted Rand Index (ARI)
# setwd("D:/Third Year/Data Science II/Option 2 Clustering/Birch-sets/ground truth centroids TXT and partitions PA")
# getwd()
# 
# label4 <- read.table("")
# label4_v <- label4[, 1]
# length(label4_v)
# 
# 
# ari_value6 <- adjustedRandIndex(label4_v, model4_label3)

# Print the results
#cat("Adjusted Rand Index (ARI):", ari_value, "\n")

#########################################################

                       # G2_set
########################################################
setwd("D:/Third Year/Data Science II/Option 2 Clustering/G2 sets")
getwd()


datag2 <- read.table("g2-1024-100.txt")

# Perform UMAP
umap_result <- umap(datag2, n_neighbors = 15, min_dist = 0.1, n_components = 2)

# Plot the UMAP
plot(umap_result[,1], umap_result[,2], asp = 1, xlab = "UMAP 1", ylab = "UMAP 2", main = "UMAP Dimensionality Reduction")


#set.seed(1)
#Gaussian Mixture Model (GMM)
modelg2 <- Mclust(umap_result, G = 2)


# Plotting the results
plot(modelg2, what="classification")

# Extracting the centroids
centroids <- modelg2$parameters$mean

#predicted label
model_labelg2 <- modelg2$classification
length(model_labelg2)

#Metric
#Adjusted Rand Index (ARI)
#setwd("D:/Third Year/Data Science II/Option 2 Clustering/Birch-sets/ground truth centroids TXT and partitions PA")
#getwd()

labelg2 <- read.table("g2-1024-100-round.txt")
label_v <- labelg2[, 1]
length(label_v)


ari_value7 <- adjustedRandIndex(label_v, model_labelg2)

# Print the results
#cat("Adjusted Rand Index (ARI):", ari_value, "\n")

###################################################################

                       #Dim_Set (Dim032)
####################################################################

setwd("D:/Third Year/Data Science II/Option 2 Clustering/DIM-sets")
getwd()
datad <- read.table("dim032.txt")

# Perform UMAP
umapd <- umap(datad, n_neighbors = 15, min_dist = 0.1, n_components = 2)

#set.seed(1)
#Gaussian Mixture Model (GMM)
modeld <- Mclust(umapd, G = 16)


# Plotting the results
plot(modeld, what="classification")

# Extracting the centroids
centroids <- modeld$parameters$mean

# Printing the centroids
print(centroids)


model_labeld <- modeld$classification
length(model_labeld)

#Metric
#Adjusted Rand Index (ARI)
setwd("D:/Third Year/Data Science II/Option 2 Clustering/DIM-sets/Ground truth centroids and partition")
getwd()

labeld <- read.table("dim032.p.txt")
label_v <- labeld[, 1]
length(label_v)


ari_value8 <- adjustedRandIndex(label_v, model_labeld)

# Print the results
#cat("Adjusted Rand Index (ARI):", ari_value, "\n")

#########################################################
                        #Dim1024
#########################################################

setwd("D:/Third Year/Data Science II/Option 2 Clustering/DIM-sets")
getwd()
datadd <- read.table("dim1024.txt")

# Perform UMAP
umapdd <- umap(datadd, n_neighbors = 15, min_dist = 0.1, n_components = 2)


#set.seed(1)
#Gaussian Mixture Model (GMM)
modeldd <- Mclust(umapdd, G = 16)


# Plotting the results
plot(modeldd, what="classification")

# Extracting the centroids
centroids <- modeldd$parameters$mean

model_labeldd <- modeldd$classification
length(model_labeldd)

#Metric
#Adjusted Rand Index (ARI)
setwd("D:/Third Year/Data Science II/Option 2 Clustering/DIM-sets/Ground truth centroids and partition")
getwd()

labeldd <- read.table("dim1024.p.txt")
label_v <- labeldd[, 1]
length(label_v)


ari_value9 <- adjustedRandIndex(label_v, model_labeldd)

# Print the results
#cat("Adjusted Rand Index (ARI):", ari_value, "\n")


############################################################
                  # Unbalance
##########################################################


setwd("D:/Third Year/Data Science II/Option 2 Clustering/Unbalance-sets")
getwd()
datau <- read.table("unbalance.txt")

set.seed(2)
#Gaussian Mixture Model (GMM)
modelu <- Mclust(datau, G = 8)

# Plotting the results
plot(modelu, what="classification")

# Extracting the centroids
centroids <- modelu$parameters$mean

# Printing the centroids
print(centroids)


model_labelu <- modelu$classification
length(model_labelu)

#Metric
#Adjusted Rand Index (ARI)
setwd("D:/Third Year/Data Science II/Option 2 Clustering/Unbalance-sets/Ground truth centroids and partitions")
getwd()

labelu <- read.table("unbalance-gt.p.txt")
label_v <- labelu[, 1]
length(label_v)


ari_value10 <- adjustedRandIndex(label_v, model_labelu)

# Print the results
#cat("Adjusted Rand Index (ARI):", ari_value, "\n")



ari_value1
ari_value2
ari_value3
ari_value4
ari_value5
ari_value7
ari_value8
ari_value9
ari_value10



