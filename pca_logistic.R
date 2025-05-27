getwd()
rm(list = ls())
ls()


n <- 1000
N <- 2 * n
q <- 10

library(magick)
library(png)
library(Matrix)
library(irlba)
library(MASS)

# load img
read_one_png <- function(file_path) {
    img <- image_read(file_path)
    img <- image_scale(img, "224x224!")
    img_data <- as.numeric(image_data(img, channels = "gray"))
    img_matrix <- img_data[, , 1]
    img_vector <- as.vector(img_matrix)
    return(img_vector)
}

# transform the data into img and save
to_img <- function(v, file_path) {
    img <- matrix(v, nrow = 224)
    img_norm <- (img - min(img)) / (max(img) - min(img))
    writePNG(img_norm, file_path)
}

# data dir
normal_dir <- "data/CT/normal/"
cancer_dir <- "data/CT/cancer/"

# data file path
normal_file_path <- list.files(normal_dir)
normal_file_path <- paste(normal_dir, normal_file_path, sep = "")

cancer_file_path <- list.files(cancer_dir)
cancer_file_path <- paste(cancer_dir, cancer_file_path, sep = "")

length(normal_file_path) # 1500
length(cancer_file_path) # 1500

set.seed(114514)
# random n samples
sampled_normal <- sample(normal_file_path, n)
sampled_cancer <- sample(cancer_file_path, n)
train_file_path <- c(sampled_normal, sampled_cancer)

# load train data
X <- matrix(0, nrow = N, ncol = 224 * 224) # (N, p)
for (i in 1:N) {
    X[i, ] <- read_one_png(train_file_path[i])
}

# standarized
X_centered <- scale(X, center = TRUE, scale = FALSE)


# ======================= PCA =======================
res_pca <- prcomp_irlba(X_centered, n = q)
pca <- res_pca$rotation # (p, q)
dim(pca)

# save the PCA result, the feature figure
for (i in 1:q) {
    to_img((pca[, i]), paste("PCA/pca", i, ".png", sep = ""))
}

# after PCA
X.new <- X_centered %*% pca
dim(X.new)


# ======================= Train =======================
# the label of train data
is_cancer <- as.factor(rep(c(0, 1), each = n))

# establish the data frame for logistic regression
data <- data.frame(X.new, cancer = is_cancer)
data$cancer <- as.factor(data$cancer)

# logistic regression model
model <- glm(cancer ~ ., data = data, family = binomial())
summary(model)

model.step <- step(model, direction = "both")
summary(model.step) # step

exp(coef(model.step))
# (Intercept)         PC1         PC2         PC3         PC4         PC5 
#   0.9894906   1.0139567   0.9792491   0.9854668   1.0054750   1.0369355 
#         PC6         PC7         PC8         PC9        PC10 
#   0.9816089   1.0286717   1.0350578   0.9817046   0.9136114 


# ======================= Test =======================
set.seed(1919810)
complement_normal <- setdiff(normal_file_path, sampled_normal)
complement_cancer <- setdiff(cancer_file_path, sampled_cancer)

test_file_path <- c(sample(complement_normal, 100), sample(complement_cancer, 100))

n_test <- length(test_file_path)
N_test <- n_test

# load test data
X_test <- matrix(0, nrow = N_test, ncol = 224 * 224)
for (i in 1:N_test) {
    X_test[i, ] <- read_one_png(test_file_path[i])
}
dim(X_test)

# standarized
X_test_centered <- scale(X_test, center = attr(X_centered, "scaled:center"), scale = FALSE)

# PCA
X_test_new <- X_test_centered %*% pca # (N_test, q)
dim(X_test_new)

# prediction
prob_test <- predict(model.step, newdata = as.data.frame(X_test_new), type = "response")
pred_test <- ifelse(prob_test >= 0.5, 1, 0)

# the real label
y_test <- rep(c(0, 1), each = 100)

# Confuse Matrix
table(pred_test, y_test)
#          y_test
# pred_test  0  1
#         0 75 23
#         1 25 77

# ACU
accuracy <- mean(pred_test == y_test)
cat("Test ACU:", accuracy, "\n")
# Test ACU: 0.76
