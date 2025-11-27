结合这份TeX文档的具体内容（连续变量Bell测量、4个输入态、离散化条件概率公式），以下是**针对该CV-MDI场景的SDP详细求解方案**，需将文档中的具体参数/公式与SDP框架逐一绑定：


### 一、前置：文档核心参数与数据的预生成
在求解SDP前，需先基于文档内容生成**输入态密度矩阵**和**离散化条件概率**，这是SDP的基础输入。


#### 1. 确定核心参数的取值范围
- 平均光子数$\mu$：范围$[0,10]$（文档建议），后续需遍历该区间采样（如间隔$0.5$，共21个采样点）；
- 离散化数量$n$：用户自定义（如$n=2$，对应$k,l \in \{1,2\}$，共$n^2=4$个离散结果）；
- 离散化边界：按文档设置$c_0=d_0=-10$，$c_1=d_1=0$，$c_2=d_2=10$（覆盖$[-10,10]$区间，足够包含高斯分布的主要概率）。


#### 2. 生成输入态的密度矩阵（针对每个$\mu$）
文档中输入态是**纯态**，密度矩阵为$\rho_\sigma = |\psi_\sigma\rangle\langle\psi_\sigma|$（$\sigma$对应4个$(s_1,s_2)$组合），步骤：
1. 对当前采样的$\mu$，计算$\delta = e^{-2\mu}$（文档中式子）；
2. 根据文档中4个输入态的向量表示，构造$4\times4$的复向量$|\psi_\sigma\rangle$（如$\sigma=(+1,+1)$对应$\begin{pmatrix}1\\0\\0\\0\end{pmatrix}$）；
3. 计算外积$\rho_\sigma = |\psi_\sigma\rangle \otimes |\psi_\sigma\rangle^\dagger$（得到$4\times4$的Hermitian密度矩阵）。


#### 3. 预计算离散化条件概率$P((k,l)|\sigma)$（针对每个$\mu$）
利用文档中误差函数形式的概率公式，计算所有$\sigma$（4个）和所有$(k,l)$（$n^2$个）对应的概率值，步骤：
1. 对每个$\sigma=(s_1,s_2)$，计算$\mu_{+\sigma} = \sqrt{2}(s_1\sqrt{\mu} + s_2\sqrt{\mu}) = \sqrt{2\mu}(s_1+s_2)$（因文档中$\mu_1=\mu_2=\mu$）；
2. 对每个$k \in \{1,2\}$，取离散化边界$c_{k-1},c_k$（如$k=1$对应$c_0=-10,c_1=0$），计算$X_+$的积分项：
   $$\text{Term}_k(\sigma) = \frac{1}{2}\left[\text{erf}\left(\frac{c_k}{\sqrt{2}} - s_1\sqrt{\mu} - s_2\sqrt{\mu}\right) - \text{erf}\left(\frac{c_{k-1}}{\sqrt{2}} - s_1\sqrt{\mu} - s_2\sqrt{\mu}\right)\right]$$
3. 对每个$l \in \{1,2\}$，取离散化边界$d_{l-1},d_l$，计算$P_-$的积分项：
   $$\text{Term}_l = \frac{1}{2}\left[\text{erf}\left(\frac{d_l}{\sqrt{2}}\right) - \text{erf}\left(\frac{d_{l-1}}{\sqrt{2}}\right)\right]$$
4. 计算联合概率：$P((k,l)|\sigma) = \text{Term}_k(\sigma) \cdot \text{Term}_l$（文档中公式的简化，因$\mu_1=\mu_2$）。


### 二、SDP的具体建模（针对单个$\mu$）
结合文档中输入态的维度（$A0B0$空间为$4$维），细化SDP的变量、目标函数与约束。


#### 1. 变量维度的具体定义
- 联合POVM算子$\tilde{M}_{a,b,e}^{A0B0}$：
  - 索引：$a=k$（$X_+$离散结果，$1\sim n$）、$b=l$（$P_-$离散结果，$1\sim n$）、$e=(a,b)$（Eve的猜测结果，共$n^2$个）；
  - 维度：$4\times4$的Hermitian半正定矩阵（因$A0B0$是4维两模子空间）。
- 局部POVM算子：
  - $\tilde{M}_{b,e}^{B0}$：$2\times2$的Hermitian半正定矩阵（$B0$是单模2维子空间）；
  - $\tilde{M}_{a,e}^{A0}$：$2\times2$的Hermitian半正定矩阵（$A0$是单模2维子空间）。
- Eve的概率$p(e)$：$n^2$维非负实向量。


#### 2. 目标函数的具体构造
文档中需计算特定输入态（如$\sigma^*=(+1,+1)$）对应的$G_{x^*,y^*}^{MDI}$，步骤：
1. 取目标输入态的密度矩阵$\rho_{\sigma^*} = |\psi_{\sigma^*}\rangle\langle\psi_{\sigma^*}|$（如$\sigma^*=(+1,+1)$对应$\rho_{\sigma^*} = \begin{pmatrix}1&0&0&0\\0&0&0&0\\0&0&0&0\\0&0&0&0\end{pmatrix}$）；
2. 选取$e=(a,b)$对应的联合POVM算子$\tilde{M}_{a,b,e=(a,b)}^{A0B0}$；
3. 构造目标函数：
   $$\text{Objective} = \text{Tr}\left(\sum_{a,b} \tilde{M}_{a,b,e=(a,b)}^{A0B0} \cdot \rho_{\sigma^*}\right)$$


#### 3. 约束的具体实现（绑定文档参数）
将文档中的输入态、概率代入SDP约束：
1. **约束1（观测数据一致）**：
   对每个$\sigma$（4个）、每个$(a,b)$（$n^2$个），添加等式：
   $$\text{Tr}\left(\sum_e \tilde{M}_{a,b,e}^{A0B0} \cdot \rho_\sigma\right) = P((a,b)|\sigma)$$
   （$\rho_\sigma$是预生成的输入态密度矩阵，$P((a,b)|\sigma)$是预计算的条件概率）。

2. **约束2（联合POVM非负）**：
   声明所有$\tilde{M}_{a,b,e}^{A0B0} \succeq 0$（$4\times4$半正定矩阵）。

3. **约束3-4（无信号）**：
   - 对每个$b,e$，添加：$\sum_a \tilde{M}_{a,b,e}^{A0B0} = \mathbb{I}^{A0} \otimes \tilde{M}_{b,e}^{B0}$（$\mathbb{I}^{A0}$是$2\times2$单位矩阵）；
   - 对每个$a,e$，添加：$\sum_b \tilde{M}_{a,b,e}^{A0B0} = \tilde{M}_{a,e}^{A0} \otimes \mathbb{I}^{B0}$（$\mathbb{I}^{B0}$是$2\times2$单位矩阵）。

4. **约束5-6（局部归一化）**：
   - 对每个$e$，添加：$\sum_b \tilde{M}_{b,e}^{B0} = p(e) \cdot \mathbb{I}^{B0}$；
   - 对每个$e$，添加：$\sum_a \tilde{M}_{a,e}^{A0} = p(e) \cdot \mathbb{I}^{A0}$。

5. **约束7（概率归一化）**：
   添加：$\sum_e p(e) = 1$。


### 三、遍历$\mu$的批量求解与图像绘制
文档中$\mu$是优化参数，需遍历其范围计算$H_{\text{min}}$并绘图。


#### 1. 批量求解流程（针对$\mu \in [0,10]$）
对每个$\mu$采样点，循环执行：
1. 计算$\delta = e^{-2\mu}$；
2. 生成4个输入态的密度矩阵$\{\rho_\sigma\}$；
3. 预计算所有$\sigma$和$(a,b)$对应的$P((a,b)|\sigma)$；
4. 构建上述SDP并调用MOSEK求解，得到最优猜测概率$G_{\text{opt}}(\mu)$；
5. 计算最小熵：$H_{\text{min}}(\mu) = -\log_2(G_{\text{opt}}(\mu))$（若$G_{\text{opt}}(\mu)=0$，设下界为$10^{-10}$避免对数无意义）。


#### 2. 绘制Figure 1的具体方案
- **横轴**：平均光子数$\mu$（范围$[0,10]$）；
- **纵轴**：最小熵$H_{\text{min}}(\mu)$；
- **绘图细节**：
  1. 将所有$\mu$采样点与对应的$H_{\text{min}}(\mu)$组成数据对；
  2. 用折线连接数据点（添加标记点如“○”以增强可读性）；
  3. 添加坐标轴标签（横轴：“Average Photon Number $\mu$”，纵轴：“Min-Entropy $H_{\text{min}}$”）；
  4. 补充网格线、图例（若有不同$n$的对比），并设置纵轴范围以突出变化趋势（如$[0,5]$）。
