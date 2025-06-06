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
cv_lasso$lambda.min # [1] 0.002218767
cv_lasso$lambda.1se # [1] 0.002551043

selected_features <- which(coef(cv_lasso, s = "lambda.min") != 0)[-1] # 去掉截距项
X_train_selected <- x[, selected_features]
dim(X_train_selected) # [1] 2000  167

# ======================= Test =======================
set.seed(1919810)
test_normal <- setdiff(normal_file_path, sampled_normal)
test_cancer <- setdiff(cancer_file_path, sampled_cancer)

test_file_path <- c(sample(test_normal, 500), sample(test_cancer, 500))

n_test <- length(test_file_path)

# load test data
X_test <- matrix(0, nrow = n_test, ncol = 56 * 56)
for (i in 1:n_test) {
    X_test[i, ] <- read_one_png(test_file_path[i])
}
dim(X_test)

# 预测概率（正类的概率）
# prob_predictions <- predict(cv_lasso, X_test, type = "response")
prob_predictions_with_lambda <- predict(cv_lasso, X_test, type = "response", s = "lambda.min")
round(prob_predictions_with_lambda, 2)

# dim(prob_predictions)
# dim(prob_predictions_with_lambda)
# all.equal(prob_predictions, prob_predictions_with_lambda)

# 转换为类别预测
predictions <- as.vector(ifelse(prob_predictions_with_lambda > 0.5, 1, 0))
length(predictions)

y_test <- rep(c(0, 1), each = n_test / 2)
length(y_test)

table(y_test, predictions) # confusing matrix
#       predictions
# y_test   0   1
#      0 494   6
#      1  11 489

sum(predictions == y_test) / n_test
# [1] 0.9816667 when n_test = 300 * 2
# [1] 0.983 when n_test = 500 * 2


# ======================= Analysis =======================
library(ggplot2)
library(reshape2)

coefficients <- coef(cv_lasso, s = cv_lasso$lambda.min)
# odds_ratios <- exp(coefficients)

coef_matrix <- matrix(coefficients[-1], nrow = 56)
dim(coef_matrix) # 56 56

# 生成平滑渐变颜色
gradient_colors <- colorRampPalette(c("blue", "white", "red"))(100) # 100 个颜色

# 设置 breaks
min_val <- min(coef_matrix)
max_val <- max(coef_matrix)
breaks <- seq(min_val, max_val, length.out = 101) # 101 个断点（比颜色数多 1）

# 绘制热图
image(
    coef_matrix,
    col = gradient_colors, # 平滑渐变颜色
    breaks = breaks, # 对应的断点
    axes = FALSE,
)

dev.copy(jpeg, "Features/lasso_features.jpg", width = 1000, height = 1000, quality = 100)
dev.off()

# ======================= Simple Feature =======================
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
