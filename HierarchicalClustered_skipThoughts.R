# The purpose of this script is to analyse the skip-thought vectors and provide a hierarchical clustering
# of the dimension which is then used to reorder the dimension for filters used in a CNN model.

setwd("C:/Users/1/James/Research/Projects/DTW_Similarity_Project/R/")

# libraries
library(data.table)
library(fastcluster)
library(jmotif)
library(geometry)

root <-"C:/Users/1/James/Research/Projects/DTW_Similarity_Project/Datasets/Sentence Similarity/dataset-sts/data/sts/sick2014/"
trainpath <- "SICK_train.txt" ; testpath <- "SICK_test_annotated.txt"
training <- read.csv(paste(root,trainpath, sep=""),sep="\t",fill = T,header = T)
testing <- read.csv(paste(root,testpath, sep=  ""), sep = "\t", fill = T,header=T)

# Data preprocessing
sentenceA_train_vector <- fread("sentenceA_train_vectors.out")
sentenceB_train_vector <- fread("sentenceB_train_vectors.out")
sentenceA_test_vector <- fread("sentenceA_test_vectors.out")
sentenceB_test_vector <- fread("sentenceB_test_vectors.out")

# hierarchical clustering 
distfunc <- function(x) as.dist((1-cor(t(x)))/2)

sentenceA_train_clust <- hclust(distfunc(as.matrix(sentenceA_train_vector)))
sentenceB_train_clust <- hclust(distfunc(as.matrix(sentenceB_train_vector)))
sentenceA_test_clust <- hclust(distfunc(as.matrix(sentenceA_test_vector)))
sentenceB_test_clust <- hclust(distfunc(as.matrix(sentenceB_test_vector)))
sentence_train_cos <- sapply(1:dim(sentenceA_train_vector)[1],function(i) cosine_dist(rbind(as.numeric(sentenceA_train_vector[i,]),as.numeric(sentenceB_train_vector[i,]))))
sentence_test_cos <- sapply(1:dim(sentenceA_test_vector)[1],function(i) cosine_dist(rbind(as.numeric(sentenceA_test_vector[i,]),as.numeric(sentenceB_test_vector[i,]))))
sentence_train_dot <- sapply(1:dim(sentenceA_train_vector)[1],function(i) dot(as.numeric(sentenceA_train_vector[i,]),as.numeric(sentenceB_train_vector[i,])))
sentence_test_dot <- sapply(1:dim(sentenceA_test_vector)[1],function(i) dot(as.numeric(sentenceA_test_vector[i,]),as.numeric(sentenceB_test_vector[i,])))

train_feature.1 <- abs(sentenceA_train_vector-sentenceB_train_vector)**2
test_feature.1 <- abs(sentenceA_test_vector-sentenceB_test_vector)**2


y_train_rel <- training$relatedness_score
y_test_rel <- testing$relatedness_score

y_train_judge <- training$entailment_judgment
y_test_judge <- testing$entailment_judgment

plot(sentenceA_train_clust)
windows()
plot(sentenceB_train_clust)
windows()
plot(sentenceA_trest_clust)
windows()
plot(sentenceB_test_clust)


sentenceA_train_groups <- cutree(sentenceA_train_clust, h=0.38)
sentenceB_train_groups <- cutree(sentenceB_train_clust, h=0.38)
sentenceA_test_groups <- cutree(sentenceA_test_clust, h=0.38)
sentenceB_test_groups <- cutree(sentenceB_test_clust, h=0.38)
