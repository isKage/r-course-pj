getwd()
rm(list = ls())
ls()
n <- 1500
N <- 2 * n
q <- 10

library(cluster)
library(ggplot2)
library(magick)
library(png)
library(Matrix)
library(irlba)

# load img
read_one_png <- function(file_path) {
  img <- image_read(file_path)
  img <- image_scale(img, "224x224!")
  img_data <- as.numeric(image_data(img, channels = "gray"))
  img_matrix <- img_data[, , 1]
  img_vector <- as.vector(img_matrix)
  return(img_vector)
}

# transform into img and save
to_img <- function(v, file_path) {
  img <- matrix(v, nrow = 224)
  img_norm <- (img - min(img)) / (max(img) - min(img))
  writePNG(img_norm, file_path)
}

# set the path to the file
normal_dir <- "data/CT/normal"
cancer_dir <- "data/CT/cancer"
normal_file_path <- list.files(normal_dir, full.names = TRUE) # 1500个
cancer_file_path <- list.files(cancer_dir, full.names = TRUE) # 1500个

# all data index
normal_idx <- 1:n
cancer_idx <- 1:n

# all data path
data_file_path <- c(normal_file_path[normal_idx], cancer_file_path[cancer_idx])

# all data matrix
X <- t(sapply(data_file_path, read_one_png))
dim(X) # n x p = 3000 x 50176

# standarized
X_centered <- scale(X, center = TRUE, scale = FALSE)
center_mean <- attr(X_centered, "scaled:center")

# PCA
res_pca <- prcomp_irlba(X_centered, n = q)
pca_rotation <- res_pca$rotation
dim(pca_rotation) # p x q = 50176 x 10

# X after PCA
X_pca <- X_centered %*% pca_rotation
dim(X_pca) # n x q = 3000 x 10

# begin cluster
kmeans_res <- kmeans(X_pca, centers = 2, nstart = 25)
cluster_res <- kmeans_res$cluster
pred_label <- as.vector(cluster_res)

# real label: 1 -> normal, 2 -> cancer
true_label_train <- rep(1:2, each = n)

# ACU, as it is not predict the label, but for cluster
accuracy_train <- max(
  mean(pred_label == true_label_train),
  mean(3 - pred_label == true_label_train)
)
cat("ACU: ", accuracy_train, "\n")
# ACU: 0.6356667
