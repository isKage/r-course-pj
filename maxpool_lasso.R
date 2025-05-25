getwd()
rm(list = ls())
ls()


n <- 1000
N <- 2 * n

library(magick)
library(png)
library(Matrix)
library(irlba)
library(MASS)
library(glmnet)

# load img
read_one_png <- function(file_path) {
    img <- image_read(file_path)
    img <- image_scale(img, "56x56!")
    img_data <- as.numeric(image_data(img, channels = "gray"))
    img_matrix <- img_data[, , 1]
    img_vector <- as.vector(img_matrix)
    return(img_vector)
}

# transform the data into img and save
to_img <- function(v, file_path) {
    img <- matrix(v, nrow = 56)
    img_norm <- (img - min(img)) / (max(img) - min(img))
    writePNG(img_norm, file_path)
}

# data dir
normal_dir <- "data/MaxPool4/CT/normal/"
cancer_dir <- "data/MaxPool4/CT/cancer/"

# data file path
normal_file_path <- list.files(normal_dir)
normal_file_path <- paste(normal_dir, normal_file_path, sep = "")

cancer_file_path <- list.files(cancer_dir)
cancer_file_path <- paste(cancer_dir, cancer_file_path, sep = "")

length(normal_file_path) # 1500
length(cancer_file_path) # 1500

# ======================= Train =======================
set.seed(114514)
# random n samples
sampled_normal <- sample(normal_file_path, n)
sampled_cancer <- sample(cancer_file_path, n)
train_file_path <- c(sampled_normal, sampled_cancer)

# load train data
X <- matrix(0, nrow = N, ncol = 56 * 56) # (N, p)
for (i in 1:N) {
    X[i, ] <- read_one_png(train_file_path[i])
}

x <- as.matrix(X)
dim(x)
y <- as.factor(rep(c(0, 1), each = n))

cv_lasso <- cv.glmnet(x, y, family = "binomial", alpha = 1) # LASSO

selected_features <- which(coef(cv_lasso, s = "lambda.min") != 0)[-1] # 去掉截距项
X_train_selected <- x[, selected_features]
dim(X_train_selected)

# ======================= Test =======================
set.seed(1919810)
test_normal <- setdiff(normal_file_path, sampled_normal)
test_cancer <- setdiff(cancer_file_path, sampled_cancer)

test_file_path <- c(sample(test_normal, 300), sample(test_cancer, 300))

n_test <- length(test_file_path)

# load test data
X_test <- matrix(0, nrow = n_test, ncol = 56 * 56)
for (i in 1:n_test) {
    X_test[i, ] <- read_one_png(test_file_path[i])
}
dim(X_test)

# 预测概率（正类的概率）
prob_predictions <- predict(cv_lasso, X_test, type = "response")

# 转换为类别预测（阈值=0.5）
predictions <- as.vector(ifelse(prob_predictions > 0.5, 1, 0))
length(predictions)

y_test <- rep(c(0, 1), each = n_test / 2)
length(y_test)

sum(predictions == y_test) / n_test
# [1] 0.9816667


# ======================= Analysis =======================
selected_features
# 在原来 56 x 56 的图上标出这些被选出的位置
featrues_img <- rep(0, 56 * 56)
length(featrues_img)
featrues_img[selected_features] <- 1

featrues_img <- matrix(featrues_img, nrow = 56, ncol = 56)
dim(featrues_img)

writePNG(featrues_img, "Features/lasso_features56.png")

# 反池化
k <- 4
unpooled_img <- matrix(0, nrow = 224, ncol = 224)
for (i in 1:56) {
    for (j in 1:56) {
        unpooled_img[(i - 1) * k + 1:k, (j - 1) * k + 1:k] <- featrues_img[i, j]
    }
}
writePNG(unpooled_img, "Features/lasso_features224.png")
