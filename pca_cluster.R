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

# 读取和预处理函数保持不变
read_one_png <- function(file_path) {
  img <- image_read(file_path)
  img <- image_scale(img, "224x224!")
  img_data <- as.numeric(image_data(img, channels = "gray"))
  img_matrix <- img_data[, , 1]
  img_vector <- as.vector(img_matrix)
  return(img_vector)
}

to_img <- function(v, file_path) {
  img <- matrix(v, nrow = 224)
  img_norm <- (img - min(img)) / (max(img) - min(img))
  writePNG(img_norm, file_path)
}

# 路径设置
normal_dir <- "data/CT/normal"
cancer_dir <- "data/CT/cancer"
normal_file_path <- list.files(normal_dir, full.names = TRUE) # 1500个
cancer_file_path <- list.files(cancer_dir, full.names = TRUE) # 1500个

# --- 修改1: 随机抽取训练集和测试集 ---
set.seed(12)
# 从正常样本中随机选1000训练，剩余500中选100测试
train_normal_idx <- sample(length(normal_file_path), n)
test_normal_idx <- sample(setdiff(1:1500, train_normal_idx), 100)
# 从癌症样本中随机选1000训练，剩余500中选100测试
train_cancer_idx <- sample(length(cancer_file_path), n)
test_cancer_idx <- sample(setdiff(1:1500, train_cancer_idx), 100)

# 构建训练集和测试集路径
train_file_path <- c(normal_file_path[train_normal_idx], cancer_file_path[train_cancer_idx])
test_file_path <- c(normal_file_path[test_normal_idx], cancer_file_path[test_cancer_idx])

# --- 修改2: 加载训练数据并计算PCA ---
# 训练数据矩阵 (2000 x 50176)
X_train <- t(sapply(train_file_path, read_one_png))
dim(X_train)

# 中心化并保存均值
X_centered <- scale(X_train, center = TRUE, scale = FALSE)
center_mean <- attr(X_centered, "scaled:center")

# 计算PCA
res_pca <- prcomp_irlba(X_centered, n = q)
pca_rotation <- res_pca$rotation # (50176 x 10)

# 训练集降维结果
X_train_pca <- X_centered %*% pca_rotation # (2000 x 10)

# --- 修改3: 在训练集上聚类 ---
kmeans_train <- kmeans(X_train_pca, centers = 2, nstart = 25)
cluster_train <- kmeans_train$cluster

# 训练集真实标签（前1000正常=1，后1000癌症=2）
true_label_train <- rep(1:2, each = n)
# 计算训练集准确率（考虑标签翻转）
accuracy_train <- max(
  mean(cluster_train == true_label_train),
  mean(3 - cluster_train == true_label_train)
)
cat("训练集聚类准确率:", accuracy_train, "\n")
# 训练集聚类准确率: 0.6356667
