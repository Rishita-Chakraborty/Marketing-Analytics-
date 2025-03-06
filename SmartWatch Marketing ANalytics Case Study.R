# ---------------------------------------------
# Load necessary packages
# ---------------------------------------------
# Install any packages if you haven't already using install.packages("packageName")
library(readxl)       # For reading Excel files
library(tidyverse)    # For data manipulation and visualization
library(ggplot2)      # For plotting
library(corrplot)     # For correlation plots
library(factoextra)   # For clustering visualization

data <- SmartWatch_Data_File
# ---------------------------------------------
# 1. Exploratory Data Analysis (EDA)
# ---------------------------------------------

# Inspect the first few rows of the data
head(data)

# Check the structure (variable types) and summary statistics
str(data)
summary(data)

# Check for missing values in each column
colSums(is.na(data))

# Univariate Analysis:
# Histogram for "ConstCom" (Constant Communication)
ggplot(data, aes(x = ConstCom)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Constant Communication Ratings",
       x = "Rating (1 to 7)", y = "Frequency")

# Boxplot for "Age"
ggplot(data, aes(y = Age)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Boxplot of Age", y = "Age (years)")

# Bar plot for categorical variable "AmznP" (Amazon Prime account)
ggplot(data, aes(x = factor(AmznP))) +
  geom_bar(fill = "orange") +
  labs(title = "Distribution of Amazon Prime Account Holders",
       x = "Amazon Prime (1 = Yes, 0 = No)", y = "Count")

# ---------------------------------------------
# 2. Clustering Analysis for Market Segmentation
# ---------------------------------------------
# We will use the product attribute variables for clustering:
# ConstCom, TimelyInf, TaskMgm, DeviceSt, Wellness, Athlete, Style

# Select the product attribute columns
product_attributes <- data %>% 
  select(ConstCom, TimelyInf, TaskMgm, DeviceSt, Wellness, Athlete, Style)

# Scale the data (important for k-means clustering)
product_attributes_scaled <- scale(product_attributes)

# Determine the optimal number of clusters using the Elbow Method
wss <- vector()
for (i in 1:10) {
  set.seed(123)  # For reproducibility
  kmeans_model <- kmeans(product_attributes_scaled, centers = i, nstart = 25)
  wss[i] <- kmeans_model$tot.withinss
}

# Plot the within-cluster sum of squares (WSS) against the number of clusters
plot(1:10, wss, type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of Clusters", ylab = "Total within-clusters sum of squares",
     main = "Elbow Method: Optimal Number of Clusters")

# Based on the Elbow plot, choose an optimal number of clusters (e.g., 3 clusters)
set.seed(123)
kmeans_result <- kmeans(product_attributes_scaled, centers = 3, nstart = 25)

# Visualize the clusters using factoextra
fviz_cluster(kmeans_result, data = product_attributes_scaled,
             ellipse.type = "convex", geom = "point",
             stand = FALSE, show.clust.cent = TRUE,
             main = "Cluster Plot of Smartwatch Customer Segments")

# Attach the cluster labels to the original data
data$cluster <- kmeans_result$cluster

# ---------------------------------------------
# 3. Summarizing the Segments
# ---------------------------------------------
# Summarize the product attributes by cluster
cluster_summary <- data %>%
  group_by(cluster) %>%
  summarise(across(c(ConstCom, TimelyInf, TaskMgm, DeviceSt, Wellness, Athlete, Style), 
                   list(mean = ~mean(. , na.rm = TRUE), 
                        sd = ~sd(. , na.rm = TRUE))))
print(cluster_summary)

# Summarize demographic information by cluster (e.g., Age, Female, AmznP, Degree, Income)
demographics_summary <- data %>%
  group_by(cluster) %>%
  summarise(Mean_Age = mean(Age, na.rm = TRUE),
            Female_Percentage = mean(Female, na.rm = TRUE) * 10

# 4. Generate Summary Statistics per Cluster
library(dplyr)

cluster_attributes <- data %>%
  group_by(cluster) %>%
  summarise(
    n = n(),  # number of respondents in each cluster
    mean_ConstCom = mean(ConstCom, na.rm = TRUE),
    mean_TimelyInf = mean(TimelyInf, na.rm = TRUE),
    mean_TaskMgm   = mean(TaskMgm, na.rm = TRUE),
    mean_DeviceSt  = mean(DeviceSt, na.rm = TRUE),
    mean_Wellness  = mean(Wellness, na.rm = TRUE),
    mean_Athlete   = mean(Athlete, na.rm = TRUE),
    mean_Style     = mean(Style, na.rm = TRUE)
  )

print(cluster_attributes)

# 5. Generate Demographic Profiles
demographics_summary <- data %>%
  group_by(cluster) %>%
  summarise(
    Mean_Age = mean(Age, na.rm = TRUE),
    Pct_Female = mean(Female, na.rm = TRUE) * 100,
    Pct_AmznP  = mean(AmznP, na.rm = TRUE) * 100
    # For Degree and Income (if coded as integers),
    # you might want to look at the distribution or the most common category:
    # Degree_Mode = names(sort(table(Degree[cluster == unique(cluster)]), decreasing = TRUE))[1],
    # Income_Mode = names(sort(table(Income[cluster == unique(cluster)]), decreasing = TRUE))[1]
  )

print(demographics_summary)



            