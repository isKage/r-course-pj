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
# Call:
# glm(formula = cancer ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 +
#     PC8 + PC9 + PC10, family = binomial(), data = data)

# Coefficients:
#              Estimate Std. Error z value Pr(>|z|)
# (Intercept) -0.011446   0.054808  -0.209   0.8346
# PC1          0.013574   0.001719   7.895 2.91e-15 ***
# PC2         -0.020755   0.002167  -9.578  < 2e-16 ***
# PC3         -0.014065   0.002778  -5.063 4.12e-07 ***
# PC4         -0.005797   0.002829  -2.049   0.0405 *
# PC5          0.035454   0.003586   9.888  < 2e-16 ***
# PC6         -0.019022   0.003659  -5.198 2.01e-07 ***
# PC7          0.026939   0.004148   6.494 8.36e-11 ***
# PC8          0.033887   0.004835   7.009 2.41e-12 ***
# PC9         -0.020469   0.005183  -3.949 7.85e-05 ***
# PC10         0.087743   0.005790  15.154  < 2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# (Dispersion parameter for binomial family taken to be 1)

#     Null deviance: 2772.6  on 1999  degrees of freedom
# Residual deviance: 2055.1  on 1989  degrees of freedom
# AIC: 2077.1

# Number of Fisher Scoring iterations: 4

exp(coef(model.step))
# (Intercept)         PC1         PC2         PC3         PC4         PC5
#   0.9886188   1.0136663   0.9794593   0.9860336   0.9942196   1.0360903
#         PC6         PC7         PC8         PC9        PC10
#   0.9811579   1.0273051   1.0344677   0.9797395   1.0917073


# ======================= Test =======================
set.seed(1919810)
complement_normal <- setdiff(normal_file_path, sampled_normal)
complement_cancer <- setdiff(cancer_file_path, sampled_normal)

test_file_path <- c(sample(complement_normal, 100), sample(complement_cancer, 100))

n_test <- length(test_file_path)
N_test <- n_test

# load test data
X_test <- matrix(0, nrow = N_test, ncol = 224 * 224)
for (i in 1:N_test) {
    X_test[i, ] <- read_one_png(test_file_path[i])
}

# standarized
X_test_centered <- scale(X_test, center = attr(X_centered, "scaled:center"), scale = FALSE)

# PCA
X_test_new <- X_test_centered %*% pca # (N_test, q)

# rename the beta to: PC1, PC2, ..., PCq
vars_used <- names(coef(model.step))[-1] # 去掉截距 (Intercept)
colnames(X_test_new) <- paste0("PC", 1:q)

# the final PCAs
X_test_selected <- X_test_new[, vars_used, drop = FALSE]

# prediction
prob_test <- predict(model.step, newdata = as.data.frame(X_test_new), type = "response")
pred_test <- ifelse(prob_test >= 0.5, 1, 0)

# the real label
y_test <- rep(c(0, 1), each = 100)

# Confuse Matrix
table(pred_test, y_test)
#          y_test
# pred_test  0  1
#         0 74 23
#         1 26 77

# ACU
accuracy <- mean(pred_test == y_test)
cat("Test ACU:", accuracy, "\n")
# Test ACU: 0.755
