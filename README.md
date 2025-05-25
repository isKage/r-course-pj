# 胸腔 CT 图像二分类问题

`R` `PCA` `Logistic` `Python` `PyTorch` `CNN`

> 数据集来自 Kaggle 的仓库 [CT-Images](https://www.kaggle.com/datasets/seifelmedany/ct-images)

**概述**

CT-Images 数据集包括 $1500$ 张正常胸腔 CT 截面图，$1500$ 张患癌胸腔 CT 截面图。项目目标是实现“正常-患癌”二分类问题，并对 CT 图片进行特征提取，以期获得更精确的医学诊断和解释，为以后的研究提供统计推断的依据。主要使用的方法包括：主成分分析法（PCA）、Logistic 回归、聚类分析、卷积神经网络。其中主成分分析用于图像降维（作为拓展，尝试使用 MaxPool 最大池化方法进行降维），聚类分析用于初步分析，Logistic 回归用于二分类，卷积神经用于更为精细的二分类。

## 1 数据描述

拟研究的问题：CI-Images 肺部 CT 图的“正常-患癌”二分类问题。CT-Images 数据集包括 $1500$ 张正常肺部 CT 图，$1500$ 张患癌肺部 CT 图。分别存储于文件夹 [data/normal/](data/CT/normal) 和 [data/cancer](data/CT/cancer) 中（数据平衡性满足）。

其中每一个 CT 图已经过处理，为 `.jpg` 格式，具有 RGB 三通道。但都经过了黑白化处理，可以转换为 `.png` 单通道格式读入程序（tips: 图片像素点在处理后值仅为 $0$ 和 $1$）。

## 2 图像处理

图像数据可以理解为一个 $3$ 维张量，分别为 $(H,\ W,\ C)$ 其中 $H,\ W$ 分别表示图片的长和宽，而 $C$ 代表了图片的通道数（例如 $C = 1$ 则为单通道灰度图，$C = 3$ 则为常见的 RGB 三通道图片）。

在本项目中，我们先读取图片，得到 $H \times W \times C$ 的张量，经处理得到灰度图，即为 $H \times W$ 的矩阵，每个元素的数值代表像素值。将图片矩阵展平，得到一个图片向量 $v \in \mathbb{R}^{HW}$ 。本项目将图片统一为 $p = 224 \times 224$ 维度的向量 $v \in \mathbb{R}^p$ 。

如此将 $n$ 个样本读入程序，按行合并成为一个样本矩阵 $X \in \mathbb{R}^{n \times p}$ 其中每一行代表一个图像样本向量，每一列代表某一位置的像素值。

## 3 数据降维：PCA 主成分分析 & 池化

### 3.1 PCA 主成分分析

由于原来的样本数据向量 $x \in \mathbb{R}^p,\quad p = 224\times 224 = 50176$ ，特征维度过高，难以进行分析和建模，故可以采用 **主成分分析 PCA** 方法进行降维。

---

主成分分析（Principal Component Analysis, PCA）由 Hotelling 于 1933 年首先提出 [[1]](#ref1) 。 目的是把多个变量压缩为少数几个综合指标（称为主成分），使得综合指标能够包含原来的多个变量的主要的信息。下面以总体主成分为例，进行算法推导：

对于我们这个问题的每一个样本，在被抽样前均为随机变量 $\widetilde{X} \in \mathbb{R}^p$ ，设其协方差存在且为 $\Sigma \in \mathbb{R}^{p \times p}$ ，对协方差进行谱分解得到：

```math
\Sigma = P \Lambda P^T \notag
```

其中 $P \in \mathbb{R}^{p\times p}$ 为正交阵，而 $\Lambda \in \mathbb{R}^{p \times p}$ 为对角阵，且对角元素为 $\Sigma$ 的特征值，即 $\Lambda = diag(\lambda_1,\ \lambda_2,\ \cdots,\ \lambda_p)$ ，且特证值按降序排列 $\lambda_{i} \geq \lambda_{i+1}$ 。设 $p_j$ 为第 $j$ 个的特征向量，即 $P$ 的第 $j$ 列。则有 $\widetilde{X}$ 的第 $j$ 个主成分：

```math
Y_j = p_j^T \widetilde{X} \notag
```

记 $Y = \left[Y_1,\ Y_2,\ \cdots,\ Y_p\right] = P^T \widetilde{X} \in \mathbb{R}^p$ ，则有：

```math
Cov(Y) = Cov(P^T X) = P^T Cov(\widetilde{X}) P = P^T \Sigma P = P^TP\Lambda P^TP = \Lambda \notag
```

如上构造的主成分 $Y_j$ 为 $\widetilde{X}$ 的线性组合，且满足在 $Y_j \perp Y_k,\quad k = 1,\ 2,\ \cdots,\ j-1$  使得 $Var(Y_j)$ 最大的线性组合，直观地理解就是 $\widetilde{X}$ 在 $Y_j$ 方向上被尽可能的分开（方差尽可能大）。

在本问题中，为实现降维，可以选择这 $p$ 个主成分的前 $q$ 个，即由原来样本 $x \in \mathbb{R}^p$ 降低到 $x' \in \mathbb{R}^q$ 。但这仍然存在问题，即数据维度 $p = 50176$ 远远大于样本量 $n = 3000$ ，同时我们也无法获得一个五万级别大小的矩阵的谱分解，所以下面提出一种更新的方法，借由矩阵 **奇异值分解（SVD 分解）**的思想计算。

---

如果我们已经获取到了样本信息，构造出了样本矩阵 $X \in \mathbb{R}^{n\times p}$ ，假设 $X$ 已经中心化，即 $\bar{X} = \bf{0} \in \mathbb{R}^p$ 。我们的目标是找到一组正交的向量 $v_1,\ v_2,\ \cdots,\ v_p \in \mathbb{R}^p$，将数据投影到这些方向上后，使得投影后的数据具有最大的方差，即：

```math
\max_{v_j \perp v_1,\ v_2,\ \cdots,\ v_{j-1}} Var(Xv_j) = \frac{1}{n} \frac{||Xv_j||^2}{||v_j||^2} = \frac{1}{n} \frac{v_j^T X^T X v_j}{v_j^Tv_j} \notag
```

由 <u>引理 1</u> ：正定阵 $A$ 第 $j$ 个特征值和特征向量为 $(\lambda_i,\ e_i),\quad \lambda_i \geq \lambda_{i+1}$ 则有

```math
\max_{x \perp e_1,\ e_2,\ \cdots,\ e_{j-1}} \frac{x^TAx}{x^Tx} = \lambda_j,\quad \text{when}\ \ x = e_{k} \notag
```

可知原问题最优解 $v_j^*$ 为 $X^TX$ 的第 $j$ 个特征向量。于是我们对 $X$ 进行 SVD 分解：

```math
X = U \Lambda V^T \notag
```

其中 $U \in \mathbb{R}^{n\times n},\ V \in \mathbb{R}^{p \times p}$ 且满足 $UU^T = I_n,\ VV^T = I_p$ 。而 $\Lambda \in \mathbb{R}^{n \times p}$ 为对角矩阵，主对角线为降序排列的奇异值，其他位置为零。注意到：

```math
X^TX = V\Lambda^T U^TU\Lambda V^T = V\Lambda^T \Lambda V^T \notag
```

注意到 $X^TX,\ \Lambda^T\Lambda,\ V \in \mathbb{R}^{p\times p}$ ，完全符合谱分解的形式，故 $V$ 的每一列均为 $X^TX$ 的特征向量，即这里的 $V$ 的每一列就是我们要找的最优 $v_j^*$ ，下面介绍如何更高效的求解 $X^TX$ 的特征向量。

若记 $u \in \mathbb{R}^n$ 为 $XX^T \in \mathbb{R}^{n\times n}$ 的特征向量，特征值为 $w$ ，则有 $XX^T u = wu$ ，左乘 $X^T$ 有：

```math
X^TXX^Tu = (X^TX) \cdot X^Tu = X^T wu = w \cdot (X^Tu) \notag
```

故有 $X^Tu$ 为 $X^TX$ 的特征向量。所以我们可以通过求解 $XX^T$ 的特征向量 $u$ 进而推出 $X^TX$ 的特征向量 $v = X^Tu$ 。需要注意的是，$XX^T \in \mathbb{R}^{n \times n}$ 维度为 $n$ ，当样本量远小于特征量 $p$ 时（例如本问题），采用这个方法能极大的提高计算效率。同样地，为了降维，我们只需选取 $XX^T$ 的前 $q$ 个特征向量即可。

---

虽然上面的方法已经极大地节约了计算成本，但当遇见 $n$ 样本量同样巨大的问题（例如本问题），求解 $XX^T \in \mathbb{R}^{n \times n}$ 的特征向量仍然十分困难。而且，出于降维的目的，我们不需要所有的 $n$ 个特征向量，而只需要前 $q$ 大的特征值对应的特征向量，所以可以采用近似的方法，只计算前 $q$ 个特征向量。

为了实现这个目标，G Golub, W Kahan 于 1965 年提出了 Golub-Kahan 双对角分解法 [[2]](#ref2) 用于求解 SVD 分解问题。而 J Baglama, L Reichel 于 2005 年进一步提出了 IRLBA 算法 [[3]](#ref3) 用于高效解决奇异值近似问题。

### 3.2 MaxPooling 最大池化

除了使用主成分分析法，在计算机视觉中 **池化 Polling** 也是常见的图像降维、图像压缩方法。常见的池化操作有均值池化、最大池化，本项目采用最大池化尝试进行降维。


## 4 初步分析：聚类分析

对于分类问题，一个简单的想法就是通过 **聚类分析**。我们使用 `R` 语言的 `cluster` 包，使用 `K-Means` 算法，对所有数据（$3000$ 张图片）进行降维后 $x \in \mathbb{R}^q$ ，再进行二分类的聚类分析。

对于样本 $x_i \in \mathbb{R}^q,\quad i=1,\ 2,\ \cdots,\ n$ 我们需要将其分为 $C_1,\ C_2,\ \cdots,\ C_k$ 类，使得簇内平方和误差最小：

```math
\min\limits_{C_1,\ C_2,\ \cdots,\ C_k}\sum\limits_{j=1}^k \sum\limits_{x_i \in C_j} ||x_i - \mu_j||^2
```

其中 $\mu_j$ 为 $C_j$ 的簇内均值：$\mu_j = \sum_{x_i \in C_j}x_i \left/ |C_j|\right.$ 。注意到，聚类分析采用”距离“作为分类标准，没有参数需要估计，且为无监督学习，即没有用到标签值的信息，结果不会太好。经计算，准确率为 $63.57\%$ 。

## 5 二分类问题：Logistic 回归

对于经过降维之后的数据向量 $x \in \mathbb{R}^q$ ，采用 **Logistic 回归** 的方式进行二分类，标签为 `normal` 和 `cancer` ，从样本的 normal 和 cancer 中分别随机抽取 $1000$ 张图片，总共 $2000$ 张图片进行训练，对于剩下的 $500$ 张 normal 和 $500$ 张 cancer 数据作为测试集，用于检查模型的准确率。

对于 Logistic 回归模型，需要估计的参数为 $\beta \in \mathbb{R}^{q+1}$ ，记 $p = P(\text{cancer})$ 为患癌概率，在原始数据 $X$ 的第一列前增加一列全 $1$ 向量 $\bf{1} \in \mathbb{R}^n$ ，即有 $X = \left[\bf{1}\quad X \right]\in \mathbb{R}^{n\times (q+1)}$ ，于是有模型：

```math
\log \frac{p}{1-p} = X\cdot \beta
```

使用数值方法估计参数 $\beta$ ，需要估计的参数量为 $q+1$ 个。在本项目中，选择 $q = 10$ ，经过计算，得到结果如下所示。

```R
             Estimate Std. Error z value Pr(>|z|)
(Intercept) -0.011446   0.054808  -0.209   0.8346
PC1          0.013574   0.001719   7.895 2.91e-15 ***
PC2         -0.020755   0.002167  -9.578  < 2e-16 ***
PC3         -0.014065   0.002778  -5.063 4.12e-07 ***
PC4         -0.005797   0.002829  -2.049   0.0405 *
PC5          0.035454   0.003586   9.888  < 2e-16 ***
PC6         -0.019022   0.003659  -5.198 2.01e-07 ***
PC7          0.026939   0.004148   6.494 8.36e-11 ***
PC8          0.033887   0.004835   7.009 2.41e-12 ***
PC9         -0.020469   0.005183  -3.949 7.85e-05 ***
PC10         0.087743   0.005790  15.154  < 2e-16 ***

(Intercept)         PC1         PC2         PC3         PC4         PC5
  0.9886188   1.0136663   0.9794593   0.9860336   0.9942196   1.0360903
        PC6         PC7         PC8         PC9        PC10
  0.9811579   1.0273051   1.0344677   0.9797395   1.0917073
```

模型解释：对于变量/主成分 $PC_i$ ，每增加一个单位，患癌事件 cancer 的 `odds` 就会扩大/缩小 $e^{\beta_i}$ 倍。

测试集检查：在剩下的 $1000$ 张图片中进行预测，总体准确度达到了 $75.50\%$ ，混淆矩阵如下：

```R
         y_test
pred_test  0  1
        0 74 23
        1 26 77
```

## 6 拓展：卷积神经网络 CNN

使用**卷积神经网络 CNN** 可以进一步提高预测的准确率，本项目实现了 2 个神经网络用于展示 `ShallowCNNModel` 和 `CNNModel` ，实现代码位于 [cnn.py](models/cnn.py) 文件中。其中 `ShallowCNNModel` 网络架构以及训练预测主程序框架来自 Kaggle 开源代码 [lung-Ctscan](https://www.kaggle.com/code/saikrishnakowshik/lung-ctscan) 。`CNNModel` 是本项目实现的更深的卷积神经，预测效果更好。

训练和预测日志可见 [logs](logs) 文件夹，其中 [logs_shallow.txt](logs/logs_shallow.txt) 为 lung-Ctscan 模型的日志， [logs.txt](logs/logs.txt) 为本项目模型的日志。 [checkpoints](checkpoints) 文件夹存储了训练后的模型参数，因为参数文件较大，故不上传。模型均在 Kaggle 提供的 GPU 环境下进行训练，结果如下所示。

| Model             | Train Loss | Train Acu | Val Loss | Val Acu   | Test Loss | Test Acu  |
| ----------------- | ---------- | --------- | -------- | --------- | --------- | --------- |
| `ShallowCNNModel` | $0.6076$   | $94.52\%$ | $0.1507$ | $93.33\%$ | $0.1240$  | $94.89\%$ |
| `CNNModel`        | $0.3526$   | $97.52\%$ | $0.1293$ | $95.11\%$ | $0.0835$  | $96.67\%$ |

## 7 总结

本项目完成了对胸腔截面 CT 医学图像进行二分类的任务，使用主成分分析（PCA）对图像进行特征提取和降维处理，利用 Logistic 模型作为分类器，实现了对数据集的二分类任务，最终测试集准确率达到 $75.50\%$。最后使用卷积神经网络进行进一步的分类，最终测试集准确率达到 $96.67\%$ 。

## Reference

[1] [Hotelling H. Analysis of a complex of statistical variables into principal components[J]. Journal of educational psychology, 1933, 24(6): 417.](https://psycnet.apa.org/record/1934-00645-001) <a id="ref1"></a>

[2] [Golub G, Kahan W. Calculating the singular values and pseudo-inverse of a matrix[J]. Journal of the Society for Industrial and Applied Mathematics, Series B: Numerical Analysis, 1965, 2(2): 205-224.](https://epubs.siam.org/doi/abs/10.1137/0702016) <a id="ref2"></a>

[3] [Baglama J, Reichel L. Augmented implicitly restarted Lanczos bidiagonalization methods[J]. SIAM Journal on Scientific Computing, 2005, 27(1): 19-42.](https://epubs.siam.org/doi/abs/10.1137/04060593X) <a id="ref3"></a>