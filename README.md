# machine-learning-in-action
code for the book *Machine Learning in Action*
## ch2
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
