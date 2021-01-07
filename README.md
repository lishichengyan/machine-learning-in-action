# machine-learning-in-action
code for the book *Machine Learning in Action*
## ch2 Classification
### 2.1 Machine Learning Basics
### 2.2 k-Nearest Neighbors
kNN的优点是容易实现、准确率高、不易受到异常数据点的影响；缺点是计算代价大。常用的距离度量：
1. 闵可夫斯基距离
   1. $(\Sigma|x_i-y_i|^p)^{1/p}$
2. 曼哈顿距离
   1. 是闵可夫斯基距离的特殊情况，$p=1$
3. 欧几里得距离
   1. 是闵可夫斯基距离的特殊情况，$p=2$
4. 海明距离
   1. 度量两个编码的相似性，如果两个字符串对应位置的比特不同则+1
5. 余弦距离
   1. $cos\theta=\frac{AB}{|A||B|}$
   2. 度量两个向量的相似度
6. 雅卡尔距离
   1. $J(A, B) = \frac{|A \cap B|}{|A|+|B|-|A \cup B|}$
   2. 度量两个集合的相似性
### 2.3 Decision Trees
**可视化树没看**
1. 如何衡量一个数据集的混乱程度
   1. 香农熵
      1. 一个系统$S$中有$n$个事件：$\{E_1, E_2, ..., E_n\}$，每个事件发生的概率是$\{P_1, P_2, ..., P_n\}$，那么事件$E_i$的自信息（self-information）是：$I(E_i) = -log_2P_i$
      2. 基于1，定义这个系统的熵$E(S) = \Sigma_{i=1}^nP_iI(E_i)=\Sigma_{i=1}^n-P_ilog_2P_i$，即熵是这个系统各个事件自信息的数学期望
      3. 基于1和2，按照某个属性$P$对数据集$S$进行划分后得到的信息增益（information gain）$IG(S, P) = E(S) - \Sigma E(S_i)$，其中$E(S)$是类别信息熵（也就是划分前数据集的信息熵，按照class label来计算，由此得名）,$S_i$是划分后得到的子集
   2. Gini不纯度
2. 有三种基本的决策树算法
   1. ID3
      1. 完全根据信息增益来构建决策树
   2. C4.5
      1. 根据信息增益率来构建决策树，是对ID3的改进
      2. 关于信息增益率的计算，步骤如下：
         1. 计算类别信息熵
         2. 计算每个属性的信息熵，$IG = \Sigma\{(每个属性可能的取值/总的可能)*(\Sigma 这种取值下子数据集的类别信息熵)\}$
         3. 计算每个属性的内在信息，$H = \Sigma 可能的取值*log_2(可能的取值)$，即计算使不考虑class label
         4. 计算信息增益率$IGR = IG / H$
      3. 优点：
         1. 可以处理连续的属性
         2. 可以处理不完整的数据
         3. 有剪枝优化
         4. 克服了ID3选择属性时偏向选择取值多的属性的不足
   3. CART
      1. 根据Gini不纯度来构建决策树
### 2.4 Naive Bayes
1. Bayes Rules
   1. 条件概率$P(A|B) = \frac{P(A)P(B)}{P(B)}$
   2. 由上式，$P(B|A) = \frac{P(A)P(B)}{P(A)}$ ,于是得到$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$，也就是所谓的Bayes公式
2. 处理文本的基本方法
   1. naive
      1. 有一堆文本的集合，首先把这堆文本都扫一遍，得到所有单词的集合，将这个集合看作一个很长的向量：$[w_1, w_2, w_3, ... , w_n]$，将每一份文本也看作一个等长的向量，如果某个词出现过，这个词对应的位置为1，否则为0。例如所有单词的集合是$\{'I', 'love', 'machine', 'learning'\}$，某个文本是"machine learning"，如果按照“I love machine learning”的顺序，该文本的向量就是$[0, 0, 1, 1]$
   2. bag of words
      1. 同时考虑单词出现的频率。例如某个文本是"machine learning learning machine"，向量是$[0, 0, 2, 2]$
3. word2vec
4. 应用：识别垃圾邮件

### 2.5 Logistic Regression
### 2.6 Support Vector Machine
### 2.7 AdaBoost
