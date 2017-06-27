#SPSS系列培训之： logistic回归
@文彤老师
## 模型简介
- 基于线性回归模型发展而来
- 线性回归研究的是连续性因变量与自变量之间的关系 
-  logistic回归研究的是因变量为分类变量与一组自变量之间的关系
  - 以治疗效果为因变量，结局为治愈/未治愈
  - 如果使用新的宣传方式，决定戒烟的概率是否更高？
  - 以客户流失为因变量，结局为保留/流失
  - 顾客是否后购买产品、订购服务。。。

##模型简介
$\hat Y =\beta _0  + {\beta _1}{x_1} +  \cdots  + {\beta _p}{x_p}$
$\hat P =\beta _0  + {\beta _1}{x_1} +  \cdots  + {\beta _p}{x_p}$

- 发生率P为因变量，它与自变量之间通常不存在线性关系
- 不能保证在自变量的各种组合下，因变量的取值仍在0~1范围之内

##因变量的分布

 - 因为因变量是二分类变量，所以不可能为正态分布
 - 因变量服从二项分布或更准确的说Bernoulli分布（n=1的二项分布） 
 - $Y_i \sim Binomial(n=1, \pi_i)$其中$\pi_i$为$Y_i=1$的概率
 -  与线性回归不同，模型中随机变量并不是确定量与随机项相加的形式。  
 - 需要寻找二分类的因变量的分布$\pi_i$与确定的系统项$\eta_i=x_i^T\beta$之间的关系形式
 - $\pi_i=\frac{1}{1+e^{-\eta_i}}=\frac{e^{\eta_i}}{1+e^{\eta_i}}$或$ln(\frac{\pi_i}{1-\pi_i})=\eta_i$


## 模型简介
$\log it(p) = \ln \frac{p}{{1 - p}}$
$\log it(p) = \beta _0  + {\beta _1}{x_1} +  \cdots  + {\beta _p}{x_p}$
注：我认为$\log it(p) = \beta _0  + {\beta _1}{x_1} +  \cdots  + {\beta _p}{x_p}+\epsilon$
是错误的，因为这里的随机干扰项不是可加的。$p$是二分类的因变量的值为1的概率。
- Odds（比数、机会比、发生比）反映事物发生的可能性的另一种方式。
- 在比较事物发生的可能性大小时，与用发生概率描述可能性等价
- 由于因变量为二分类，所以误差项服从二项分布，而不是正态分布
- 因此，常用的最小二乘法也不再适用；参数估计常用**最大似然法**

## 广义线性模型和连接函数
广义线性模型：$ g(\mu)=x^T\beta$，
其中，$\mu=E(Y)$，$Y$服从包括正态分布的指数分布族中的已知分布。
广义线性模型包括三个部分：
（1）随机部分，即变量所属的指数分布族的特定分布，如二项分布、正态分布、泊松分布。
（2）线性部分$x^T\beta$，它是自变量的线性表达式
（3）连接函数$g(\mu)$：它是因变量均值的函数。例如，logistc模型的连接函数为$ln(\frac{\mu}{1-\mu})$

- 线性回归的连接函数$g(\mu)=\mu=E(Y)$
```r
glm对象=glm(公式,family=binomial(link=logit),data=数据框)
```
## 模型用途
- 影响因素分析，求出哪些自变量对因变量发生概率有影响。并计算各自变量对因变量的比数比(**Odds Rate**，比值比、优势比)即表中的**exp(B)**
- 作为判别分析方法，来估计各种自变量组合条件下因变量各类别的发生概率，从而对结局进行预测。
- 该模型在结果上等价于判别分析

##模型简介
$\log it(p) =\beta _0  + {\beta _1}{x_1} +  \cdots  + {\beta _p}{x_p}$

- $\beta _0$是常数项，表示自变量取值全为0时，**比数**（Y=1与Y=0的概率比值，也称优势、机会比）的自然对数值
- $\beta _j,j\ne0$为logistic回归系数，表示当其他自变量取值保持不变时，自变量$x_j$取值增加一个单位引起**比数自然对数值**的变化量
- $Exp(\beta _j)$表示自变量$x_j$取值增加一个单位引起事件(Y=1)概率比值相比与原来的比值的倍数

## 参数估计的理论基础
令$(z_{j1},\cdots,z_{jp})$是$p$个预测变量的第$j$次观测，即第$j$条观测记录。
令$z_j=[1,z_{j1},\cdots,z_{jp}]^T$
假定观测值$Y_j$服从二项分布$P(Y_j=y_j)=p(z_j)^{y_j}[1-p(z_j)]^{1-y_j},y_j=0,1$
观测值$Y_j$的期望值为$E(Y_j)=p(z_j)$;方差为$Var(Y_j)=p(z_j)[1-p(z_j)]$
在logistic回归中，机会比的自然对数遵循线性模型
$ln(\frac{p(z_j)}{1-p(z_j)})=\beta_0+\beta_1z_{j1}+\cdots+\beta_pz_{jp}=\beta^Tz_j$
其中参数列向量$\beta=[\beta_0,\cdots,\beta_p]^T$

## 参数的极大似然估计 
似然函数$L(\beta)=\prod_{j=1}^np(z_j)^{y_j}[1-p(z_j)]^{1-y_j}$
参数的极大似然估计没有显式封闭解，需要从一个初始值出发通过迭代得到似然函数的最大值。这种迭代方法称为迭代加权最小二乘法（Iteratively reweighted least squares (IRLS)）记为极大似然估计的参数值$\hat \beta$。

## 参数的置信区间
当样本量足够大时，参数估计值$\hat\beta$渐进正态分布，渐近均值为$\beta$，渐近协方差矩阵为：$\hat{cov}  (\hat\beta)\approx[\sum_{j=1}^{n}\hat p(z_j)(1-\hat p(z_j))z_jz_j^T]^{-1}$
该协方差矩阵的对角元素的平方根分别为参数的标准差的估计值，即标准误差$SE(\hat\beta)$
$\beta_j$大样本的95%置信区间为$\hat\beta_j\pm 1.96SE(\hat\beta_j)$

## 参数的z检验
根据中心极限定理，当样本量足够大，参数估计值$\hat\beta$渐进正态分布。
$\frac{\hat\beta_j-\beta_j}{SE(\hat\beta_j)}\sim N(0,1)$
假设检验参数是否显著，原假设是$\beta_j=0$，
因此可以构造检验统计量$T_j=\frac{\hat\beta_j}{SE(\hat\beta)}$
在原假设成立时，该统计量近似服从标准正态分布。这种检验方法称为z检验。

## 似然比检验与离差
- 离差(deviance，DEV)：-2对数极大似然值
- $DEV=-2lnL(\hat\beta)$。这个似然值是特定自由参数集合对应的最大似然函数值。当模型选择的自由参数不同，其最大似然值不同，离差也不同。
- **似然比检验**：两模型的离差之差即为似然比统计量，自由度亦为两模型参数个数之差。 在原假设下，该似然比统计量服从卡方分布。可求出对应的p值，来判断是否能推翻原假设。
- R语言中卡方检验的p值计算：
`1-pchisq(卡方值,自由度)`或`pchisq(卡方值,自由度,lowtail=F)`

## 似然比检验
  都是参数少的模型的离差与参数多的模型的离差之差的卡方检验。

- 整体显著性的似然比检验
 -  只有常数项的空模型和现模型的离差之差的卡方检验。
- 单个系数$\beta_j$显著性的似然比检验
 - $\beta_j=0$模型和 $\beta_j$自由变化时的模型的离差之差的卡方检验。
- 两个模型的似然比检验
 - 两个模型的离差之差的卡方检验，两个模型应是嵌套的。 

## 离差与平方和的类比
离差|平方和
---|---
极大似然法|最小二乘法
空模型离差|总平方和
现模型离差|残差平方和
若随机扰动项为正态分布，线性回归的残差平方和也是一种离差。因为该情况下最小二乘法和极大似然法的估计是相同的。离差对于不是线性的回归模型也适用，只要能写出似然函数。


## 模型中用到的检验方法
1. Walds test：基于标准误估计值的单变量检验
没有考虑其他因素的综合作用，当因素间存在共线性时结果不可靠!
故在筛选变量时，用Walds法应慎重
2. LR test(似然比检验)：直接对两个模型进行的比较
当模型较为复杂时，建议使用似然比检验进行变量的筛选工作，以及模型间优劣的比较
- 两模型的离差之差即为似然比统计量，自由度亦为两模型参数个数之差。
- 离差(deviance)：-2对数似然值
3. Score test
考虑在已有模型基础上引入新变量之后模型效果是否发生改变

## 参数检验

在求出极大似然法估计$\hat\beta_i$的同时可以得到Fisher信息阵I：
$$I=\{\frac{\partial^2lnL}{\partial \beta_i\partial\beta_j}\vert \hat\beta_0, \hat\beta_1,\cdots,\hat\beta_p\}$$
它的逆阵$I^{-1}$是$\hat\beta_i$的协方差矩阵。$I^{-1}$的对角阵元素$(I^{-1})_{ii}$是$\hat\beta_i$的方差
$$var(\hat\beta_i)=(I^{-1})_{ii}\\ SE(\hat\beta_i)=\sqrt{(I^{-1})_{ii}}$$。
因此$\beta_i$的检验
检验统计量$$Z=\frac{\hat\beta_i}{SE(\hat\beta_i)}\sim N(0,1)$$
置信度为$1-\alpha$的置信区间$\hat\beta_i\pm Z_\alpha SE(\hat\beta_i)$




## 定性变量的处理--哑变量1

回归系数$\beta_j$表示其它自变量不变，$x_j$每改变一个单位时，所预测的比数对数值的平均或预期变化量。

- 当$x_j$为连续性/二分类变量时这样没有问题
- 当$x_j$为多分类变量时就不太合适了
 - 无序多分类：民族，各族之间不存在大小问题
 - 有序多分类：家庭收入分为高、中、低三档，它们之间的差距无法准确衡量
 - 强行规定为等距显然可能引入更大的误差
 
## 定性变量的处理--哑变量2
在以上这些情况时，我们就必须将原始的多分类变量转化为数个哑变量（Dummy Variable, 也称虚拟变量），每个哑变量只代表某两个级别或若干个级别间的差异，这样得到的回归结果才能有明确而合理的实际意义。
**注意：哑变量必须同进同出，否则含义可能改变** 


## 定性变量的处理--哑变量3
例如，我们想了解不同年级的大学生是否有购买华为手机的意愿。在自变量中有一个分类变量为年级。		
大一是作为对比水平（基础水平、基准水平），而哑变量V1、V2、V3分别代表了大二、大三、大四和大一相比的系数 
哑变量|大一|大二|大三|大四
---|---|---|---|---
v1|0|1|0|0
v1|0|0|1|0
v1|0|0|0|1


-  一个有$n$个类别的分类变量应转化为$n-1$个哑变量。所有哑变量都为0时，对应的类别为基准类别组。

## 模型选择
- AIC认为不存在唯一的真模型，只要预测精度好，变量个数适当即可
- BIC认为存在唯一真模型，对多余变量严加惩罚。

0-1变量回归（包括logistic回归和probit回归等）
$AIC=-2lnL+2df$
$BIC=-2lnL+ln(n)df$
其中,$L$为最大似然值；$n$为样本量；$df$是自由参数的个数（包括常数项）。

- $AIC$可能精度更高， $BIC$模型更简单。
```r
glm对象=glm(公式,family=binomial(link=logit), data=数据框)
step(glm对象, trace=0)#AIC
step(glm对象, k=log(n), trace=0)#BIC
```
类似函数还有MASS包的stepAIC函数。
## 预测和评估
```r
predict(glm对象, 自变量值)#返回预测的对数发生比
predict(glm对象, 自变量值, type="response")#返回预测的概率

```
分类阈值logistic回归可以得到因变量为1和0的概率，我们有时还要判断到底属于哪类.
这时需要确定一个分类阈值$\alpha$,这样当
当$P(Y_i=1|X_i)>\alpha$时，判断为$Y_i=1$，否则为$Y_i=0$

## ROC曲线和AUC

- TPR真阳性率：将真实阳性正确预测为阳性的概率。也称为灵敏性（sensitivity）
- FPR假阳性率：将真实阴性错误预测为阳性的概率。
- 特异性（Specification）：将真实阴性正确预测为阴性的概率。
接收器工作特性（ Receiver Operating Characteristic，ROC）曲线是不同分类阈值下真阳性率(Sensitivity) 和假阳性率（1-特异性）的关系图。ROC曲线上的每个点表示对应于特定决策阈值的灵敏度/特异性对。 具有完美辨别力（两个分布中无重叠）的测试具有通过左上角（100％灵敏度，100％特异性）的ROC曲线。 因此，ROC曲线越接近左上角，测试的总体精度就越高(Zweig & Campbell, 1993).
https://www.medcalc.org/manual/roc-curves.php
- AUC为ROC曲线下方所围面积。该值越大越好。

## R语言绘制ROC曲线
用pROC包画ROC曲线
```r

```
Logistic Regression
=====
type: section


logist回归案例：游乐园季度门票的销售
=====
数据包括季票销售记录和可能的解释变量：通过什么渠道知道有销售季票（电子邮件、直接邮寄或亲临公园）、是否和其他促销手段捆绑（提供免费停车）。
问题：捆绑促销（送免费停车）是否能刺激消费者购买季票？
```{r}
pass.df <- read.csv("http://goo.gl/J8MH6A")
pass.df$Promo <- factor(pass.df$Promo, levels=c("NoBundle", "Bundle"))
summary(pass.df)
```
Note: we set the `bundle` factor levels explicitly (why?)


Logistic回归 glm(): model 1
=====
是否捆绑促销 `Promotion` 有助于季票销售 `Pass` purchase?
```{r}
pass.m1 <- glm(Pass ~ Promo, data=pass.df, family=binomial)
summary(pass.m1)
```


捆绑促销效果：1
=====
```{r}
# 模型的参数
coef(pass.m1)
exp(coef(pass.m1))                    # 发生比odds ratio
exp(confint(pass.m1))                 # 发生比的置信区间
```
## 考虑两个因素的主效应
```{r}
pass.m2 <- glm(Pass ~ Promo + Channel, data=pass.df, family=binomial)
summary(pass.m2)
```

用数据可视化进行探索!
=====
当将销售按不同渠道分别观察时，发现在两个渠道中捆绑销售相对于不捆绑反而购买季票的比例更少！ `bundle` has lower proportional sales of `pass` after it is broken
out by `channel`. 这就是 _Simpson's paradox_.
```{r}
library(vcd)    # install if needed
doubledecker(table(pass.df))
```
辛普森悖论也叫Yule–Simpson 效应，是在概率论或统计学中的一个悖论。它指数据在不同分组呈现一种结论，但当这些分组结合时却表现出与分组不同甚至相反的结论的一种悖论。https://en.wikipedia.org/wiki/Simpson%27s_paradox.

考虑渠道和交互项的模型
=====
更复杂的模型表明捆绑销售只在电子邮件这一渠道中有效。 
我们没有必要在其他渠道投放捆绑销售（免费停车）(park, direct mail). 
```{r}
pass.m3 <- glm(Pass ~ Promo + Channel + Promo:Channel, 
               data=pass.df, family=binomial)
exp(confint(pass.m3))   # 发生比的置信区间
```
如果数据是真实的，后续研究将是如何理解为什么这会发生。


Bonus: 可视化分析
=====
我们可以比较不同干预措施的效果（渠道，捆绑销售）(channel, bundle).

First, get the coefs and CIs:
```{r}
pass.ci <- data.frame(confint(pass.m3))     # 置信区间
pass.ci$X50 <- coef(pass.m3)                # 估计值
pass.ci$Factor <- rownames(pass.ci)         # ggplot2标签
pass.ci
```

用ggplot绘图, 1
=====
首先，我们添加中点，置信区间的上限和下限数据：
```{r}
library(ggplot2)
p <- ggplot(pass.ci[-1, ], 
            aes(x=Factor, y=exp(X50), 
                ymax=exp(X97.5..), ymin=exp(X2.5..)))
```
We take the `exp()` of each one to get the odds ratio.

接下来添加中点和错误条:
```{r}
p <- p + geom_point(size=4) + geom_errorbar(width=0.25)
```
And add a line where odds ratio==1.0 (no effect):
```{r}
p <- p + geom_hline(yintercept=1, linetype="dotted", 
                    size=1.5, color="red")
```


Build a ggplot, 2
=====
Now plot that with titles (and rotate it for simplicity):
```{r}
p + ylab("Likehood by Factor (odds ratio, main effect)") +
  ggtitle(paste("95% CI: Pass sign up odds by factor")) + 
  coord_flip()
```








## SPSS系列培训之： Logistic模型族进阶
@文彤老师
无序多分类Logistic回归模型
研究问题
病例－对照研究中设立一组病例和多组对照，需要分析暴露是否和患病有关，则结局变量为无序三分类，应当使用该模型加以分析。
病例
医院对照
健康人群对照
无序多分类Logistic回归模型
因变量为无序多分类
除一个对照水平外，以每一分类与对照水平作比较 
例如结果变量有三个水平：a、b、c，如果以a为参照水平，就可以得到两个Logistic函数，一个是b与a相比，另一个是c与a相比
同时应当有：Pa+Pb+Pc=1 
无序多分类Logistic回归模型
模型简介
无序多分类Logistic回归模型
案例：不同背景人群的选举倾向
老布什、克林顿、佩罗在1992年进行的较量，数据来自SPSS自带的vote.sav。
pres92，所欲选的总统候选人；
age，年龄；
agecat，年龄分组；
educ，受教育年数；
degree，最高学历；
sex，性别。
有序多分类Logistic回归模型
研究问题
所测量的结局变量为等级，或者数量较少的评分（如1～5分）
疗效：痊愈、显效、好转、无效
单变量分析使用秩和检验即可，如果进行多变量分析，简单的按照连续变量来处理可能不合适
有序多分类Logistic回归模型
多分类有序因变量的资料，分类水平大于2且水平之间有等级关系。
拟合（水平数-1）个logit模型，称为累加logit模型（Cumulative logits model）。
例如对一个四分类有序变量，即应当同时拟合以下三个模型：π1、π2、π3分别为因变量取第一类、第二类、第三类时的概率，而第四类则作为用于对比的基础水平。 
## 有序多分类Logistic回归模型
模型简介
有序多分类Logistic回归模型
可见，这种模型实际上是依次将因变量划分为两个等级，不管模型中因变量的分割点在什么位置，模型中各自变量的系数β都保持不变，所改变的只是常数项α。此时求出的OR值是自变量每改变一个单位，因变量提高一个及一个以上等级的比数比。 
这种假设看似复杂，但大量实践证明，它是符合多数实际情况的。
案例：工作满意度影响因素分析

# 学习资源
Logistic Regression and Generalised Linear Models: Blood Screening,
Women’s Role in Society, and Colonic Polyps
https://onlinecourses.science.psu.edu/stat504/node/164

高级回归（推荐！）Patrick Breheny在肯塔基大学的高级回归的课件
http://web.as.uky.edu/statistics/users/pbreheny/760/S13/notes/3-26.pdf
http://web.as.uky.edu/statistics/users/pbreheny/760/S13/notes.html
http://web.as.uky.edu/statistics/users/pbreheny/teaching.html

https://cran.r-project.org/web/packages/HSAUR/vignettes/Ch_logistic_regression_glm.pdf
 anova(plasma_glm_1, plasma_glm_2, test = "Chisq")
 
#离差值的卡方检验，即似然比检验

residuals(glm对象, type = "deviance")#取出离差残差
http://r.789695.n4.nabble.com/Deviance-Residuals-td2332307.html
Residuals on the scale of the response, y - E(y); in a binary logistic regression, y is 0 or 1 and E(y) is the fitted probability of a 1. As it turns out, response residuals aren't terribly useful for a logit model. 

> residuals(glm1, "pearson") 

Components of the Pearson goodness-of-fit statistic. 

> residuals(glm1, "deviance") 

Components of the residual deviance for the model. 

> residuals(glm1, "working") - especially this one confuses me a lot! 

Residuals from the final weighted-least-squares regression of the IWLS procedure used to fit the model; useful, for example, for detecting nonlinearity. 



对于 logistic回归,
$lnL=\sum_i {{y_i
log\hat \pi_i + (1 − y_i) log(1 −\hat \pi_i)}}$

通过类比线性回归, 离差残差可以如下定义：

离差残差（deviance residual）:
$d_i = s_i \sqrt{−2 [{y_i log\hat{\pi}_i + (1 − y_i) log(1 −\hat{\pi}_i)}]},$
其中， $s_i $= 1若$y_i = 1$ 和$s_i $= −1若$y_i =0$

离差残差的平方和即离差:
$D =\sum_i{d_i^2}=-2lnL$
> Written with [StackEdit](https://stackedit.io/).

## 极大似然法
就是在假定整体模型**分布已知**，利用已知的样本结果信息，反推最具有可能（最大概率）导致这些样本结果出现的模型参数值。极大似然估计提供了一种给定观察数据来评估模型参数的方法，即：“模型已定（分布已知），参数未知”。
- 极大似然估计中采样需满足一个重要的假设，就是所有的采样都是**独立同分布**的。
- 极大似然估计回归可以处理非正态分布。
- 最小二乘法回归估计是在模型形式已知条件下，使得样本中各点的因变量观测值与拟合值之差（残差）的平方和最小的参数估计。
[教学视频](https://mp.weixin.qq.com/s?__biz=MzA4NDEyMzc2Mw==&mid=2649677782&idx=2&sn=17dfad34c5e65f2c2f6edb122a635083&chksm=87f676cab081ffdcab66f1f28526ed08d818a665bb42a9309f5a97318614b7ba7b8d47dcd3a0&mpshare=1&scene=1&srcid=062602U5XBL28mAGClDDLhNU#rd)

## Donner party唐纳派对案例

early snowfall
American pioneers
Our example data set from today involves the survival of the members of the Donner party
在1846年春天，一群美国先驱者从加州出发。然而，他们遭受了一系列的挫折，直到10月才到达内华达山脉。穿过山脉时，他们被不期而遇的降雪困住，不得不在那里度过冬天。条件恶劣，食物供应量不足，87名Donner派对中有40人在最终获救之前死亡。

The data set donner.txt contains the following information regarding the adult (over 15 years old) members of the Donner party:
Age Sex
Status: either Died or Survived

```r
fit <- glm(Status~Age*Sex,donner,family=binomial) 
summary(fit)

repeat { 
old <- b
eta <-  X*b
pi <- exp(eta)/(1+exp(eta))
W  <-  diag(as.numeric(pi*(1-pi))) 
z <- eta +  solve(W) *(y-pi)
#solve为求逆阵
b <- solve(t(X) * W  * X) * t(X) * W  * z    
if (converged(b,old)) break
}
```
接下来，将计算过程写出来。
首先，注意到对于二项分布$\mu_i=\pi_i$,
$W (\mu_i) = \mu_i(1 −\mu_i)= \pi_i(1 −\pi_i)$
W是一个对角阵，该对角阵的第i个主元为$\pi_i(1 −\pi_i)$
采用的算法为IRLS 算法:

```r
VarB <- solve(t(X)*W*X) 
SE   <- sqrt(diag(VarB))
z  <-  b/SE
p  <- 2*pnorm(-abs(z))
```
其中，VarB为系数的方差协方差矩阵；SE为系数的抽样分布的标准误。

|Item |  Estimate| Std. Error |z value |	P |
|---|---|---|---|---|
|(Intercept)|	0.3183|	1.1310|	0.2815|	0.7784|
|Age|	-0.0325|	0.0353|	-0.9209|	0.3571
|Female|	6.9280|	3.3989|	2.0383|	0.0415
|Age:Female|-0.1616|	0.0943|	-1.7143|0.0865

Let’s write out the model as:

$ln(\frac{\pi}{1-\pi})= \beta_0 +\beta_1Age +\beta_2Female + \beta_3Age·Female$

20岁男性的生存概率是多少?

$\hat\pi = .418$
所以，20岁男性的生存概率为 41.8%。
同理可得,  40岁男性的生存概率为 27.3% ； 6020岁男性的生存概率为16.4% 。

20岁女性的生存概率是多少?

η = β0 + 20β1 + β2 + 20β3
ηˆ = 3.3649
πˆ = .967

20岁女性的生存概率是96.7%; 40岁女性的生存概率是37.4% ；60岁女性的生存概率是  1.2%。

总结趋势：年轻成年女性的生存概率高于年轻成年男性，但是随着年龄的增长，其生存概率下降得更快，直到老年女性难以生存的年龄高于老年男性。

As you can see (most clearly for the females), logistic regression produces fitted probabilities that take on an “S” shape (or sigmoidal curve)
This comes from the logit link function, which as we have demonstrated earlier, constrains the fitted probabilities to lie within [0, 1]
Once again, we see the utility of the link function: it allows us to obtain a nonlinear fit from a linear model
 
Finally, all of the same questions one asks in linear regression about the systematic part of the model are still relevant to GLMs
For example, we have assumed a linear effect – is this reasonable?
Perhaps the probability of survival is low for young adults and the elderly, and at its maximum in middle age

在模型的系统组成部分中包括对年龄的二次效应，我们看到这个想法可能有一些理由。

##概率的估计值

We have already talked about estimation of probabilities based on the fit of a logistic regression model:
(1)给定一个解释变量x, 计算得到线性预测值$\hat\eta=x^T\beta$
(2)估计概率$\hat\pi=\frac{e^\eta}{1+e^\eta}$
由于最大似然估计对于转换是不变的, 当$\hat\eta$是线性估计值时，$\hat\pi$就是$\pi$的极大似然估计值。

## 概率的置信区间

因为有
$\frac{x^t\hat\beta-x^t\beta}{\hat{SE}} \sim z$
其中$\hat{SE}=\sqrt{x^T(X^TWX)^{-1}x}$
构建$\eta$的置信区间
$(L,U)=(\hat\eta-z_{\alpha/2}\hat{SE},\hat\eta+z_{\alpha/2}\hat{SE})$
根据不变性，可得$\pi$的置信度为$1-\alpha$的置信区间：
$(\frac{e^L}{1+e^L},\frac{e^U}{1+e^U})$


## logistic回归和病例对照研究

这对病例对照研究（case-control study）具有特别重要的后果。
定理：在病例对照研究中，当满足假设：
（a）模型是正确的
（b）案例和控制的选择与解释变量无关
最大似然估计$\hat\beta_1,\cdots,\hat\beta_{p-1}$（不包括$\hat\beta_0$）以及它们的近似抽样分布等同于从逻辑回归模型获得的数值。

- 假设选择病例和控制是独立于解释变量实际上是一个相当大的假设，在实际病例对照研究中经常被违反。
- 这是偏差（bias）的主要来源!
例如，病例对照研究发现儿童期白血病与暴露于电磁场（EMF）存在联系。

然而，随后的调查表明，这完全是因为病例对照偏差。
社会经济地位低的家庭更有可能生活在电磁场附近。
社会经济地位低的家庭也不太可能作为控制组对象参与研究。
社会经济状况不影响病例参与（病例通常渴望参与）。这导致EMF与白血病之间的虚假关联。

## odds ratio的估计
因为$ln\frac{\pi}{1-\pi}=x^T\beta$,即$\frac{\pi}{1-\pi}=exp(x^T\beta)$
比值比odds ratio
$$OR=\frac{\frac{\pi_2}{1-\pi_2}}{\frac{\pi_1}{1-\pi_1}}=exp((x_2-x_1)^T\beta)$$

特别地，考虑当其余的解释变量保持不变，$x_j$变化量$\delta_j$时比值比的情况：
$$OR = exp(\delta_j\beta_j)$$.
这正是我们需要的：所有其他变量消失，我们的估计仅依赖于βj和xj的变化.
我们可以因此估计这个优势比，以及完全基于$\beta_j$进行推理。特别是我们甚至不需要估计$\beta_0$，所以我们可以应用这些结果进行病例对照研究。

例如，Donner一名成员年龄增长10年在冬季不能生存的概率是甚么？ 
OR = exp（10·0.0325）= 1.4（男）
OR = exp（10·0.1941）= 7.0（女）
注意，这些优势比适用于任何10年差异（50对40,30与20等由于线性假设）

## odds ratio的置信区间



可以通过首先获得以$\hat\beta$的置信区间，然后变换来获得优势比的置信区间。
对于线性预测因子，年龄效应的斜率的置信区间为
（-0.0366, 0.1016）  （男性）
（ 0.0227, 0.3654）   （女性）
死亡比例比的置信区间（再次，年龄变化为10岁）为：
$e^{10·(L,U)} = (0.7, 2.8)$	(男性)
$e^{10·(L,U )} = (1.3, 38.6)$	(女性)

## 假设检验
优势比的假设检验（和事实上的概率）直接等效于回归系数的检验，因为以下都是等效的：
βj = 0
Odds ratio = 1
Difference in probabilities = 0
Ratio of probabilities (relative risk) = 1

例如, 检验H0 : $\beta_{Age|Male} = 0$ 
$z =\frac{\hat\beta_{Age|Male}}{SE}= \frac{−0.03248}{0.03527} =	0.921$,
$p = 2 \Phi (−0.921) =  0.36$
同理, 检验H0 :$ \beta_{Age|Female} = 0$, 可得$p = .03$
测试相对功效较低，因为只有45个被试; 因此尽管估计的效应值较大，但p值较高。
这种现象从置信区间也可以看出。


## 置信区间: Wald vs. likelihood ratio
这不是唯一 (甚至不是最优方法)来对 GLM模型进行推理
这个方法称为 Wald方法,是由统计学家 Abraham Wald提出的。
该方法基于对$\hat\beta$ 的近似估计; 它不一定提供关于$\beta$离$\hat\beta$距离的准确信息
下一次，我们将讨论一种基于似然比测试和构建置信区间的替代方法，并看到它从近似问题中受到的影响较小.

```r
## Data
donner <- read.delim("http://web.as.uky.edu/statistics/users/pbreheny/760/data/donner.txt")

## Three approaches:
##  A) A 'by hand' approach, doing the math yourself
##  B) A more automatic approach, but that involves refitting the model
##  C) Using a function 'estimate' that I put online

## Slides 16-17: Approach A
fit <- glm(Status~Age*Sex, donner, family=binomial)
b <- coef(fit)
lam.m <- -c(0,10,0,10)
lam.f <- -c(0,10,0,0)
exp(crossprod(lam.m, b))
exp(crossprod(lam.f, b))

SE.m <- sqrt(crossprod(lam.m, vcov(fit))%*%lam.m)
SE.f <- sqrt(crossprod(lam.f, vcov(fit))%*%lam.f)
exp(crossprod(lam.m, b) + qnorm(c(.025,.975))*SE.m)
exp(crossprod(lam.f, b) + qnorm(c(.025,.975))*SE.f)

## Slide 16-17: Approach B
Died <- 1*(donner$Status=="Died")
Female <- 1*(donner$Sex=="Female")
fit1 <- glm(Died~Age*Sex,donner,family=binomial)
fit2 <- glm(Died~Age*Female,donner,family=binomial)
summary(fit1)
summary(fit2)
exp(10*coef(fit1)["Age"]) ## Female
exp(10*coef(fit2)["Age"]) ## Male
SE1 <- 10*summary(fit1)$coef[2,2]
SE2 <- 10*summary(fit2)$coef[2,2]
exp(10*coef(fit1)["Age"] + qnorm(c(.025,.975))*SE1) ## Female
exp(10*coef(fit2)["Age"] + qnorm(c(.025,.975))*SE2) ## Male

## Slide 16-17: Approach C
source("http://web.as.uky.edu/statistics/users/pbreheny/760/S13/notes/estimate.R")
estimate(lam.m, fit, ci=TRUE, trans=exp)
estimate(lam.f, fit, ci=TRUE, trans=exp)

```



 各位亲们，能否推荐使用体验良好的一个数据统计分析的互动教学软件？能自己扩展教学内容，最后能联机考试的那种。我所在学院有点经费可以购买。
 "Age"	"Sex"	"Status"
23	"Male"	"Died"
40	"Female"	"Survived"
40	"Male"	"Survived"
30	"Male"	"Died"
28	"Male"	"Died"
40	"Male"	"Died"
45	"Female"	"Died"
62	"Male"	"Died"
65	"Male"	"Died"
45	"Female"	"Died"
25	"Female"	"Died"
28	"Male"	"Survived"
28	"Male"	"Died"
23	"Male"	"Died"
22	"Female"	"Survived"
23	"Female"	"Survived"
28	"Male"	"Survived"
15	"Female"	"Survived"
47	"Female"	"Died"
57	"Male"	"Died"
20	"Female"	"Survived"
18	"Male"	"Survived"
25	"Male"	"Died"
60	"Male"	"Died"
25	"Male"	"Survived"
20	"Male"	"Survived"
32	"Male"	"Survived"
32	"Female"	"Survived"
24	"Female"	"Survived"
30	"Male"	"Survived"
15	"Male"	"Died"
50	"Female"	"Died"
21	"Female"	"Survived"
25	"Male"	"Died"
46	"Male"	"Survived"
32	"Female"	"Survived"
30	"Male"	"Died"
25	"Male"	"Died"
25	"Male"	"Died"
25	"Male"	"Died"
30	"Male"	"Died"
35	"Male"	"Died"
23	"Male"	"Survived"
24	"Male"	"Died"
25	"Female"	"Survived"