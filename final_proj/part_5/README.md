# 开放题说明
### quantum_wavefunc_redis
这是一个完整的python项目，用于存储和管理量子波函数，并利用Redis数据库来高效存储和检索量子波函数数据，支持基本的增删改查操作。

### 项目结构
```
quantum_wavefunc_redis/
├── pyproject.toml          # 项目配置文件
├── README.md               # 项目说明文档
└── quantum_wavefunc_redis/ # 核心代码目录
    ├── __init__.py         # 包初始化文件
    ├── __main__.py         # 程序入口文件
    └── wave_function.py    # 波函数核心逻辑文件
```

### 灵感来源
本学期课程上学了WSL（Windows Subsystem for Linux），因此项目中使用了基于Linux运行的本机Redis服务作为数据存储。并且选择了量子力学中的含时波函数作为项目的核心数据类型，充分发挥了Redis在高并发、低延迟场景下的优势，同时贴合物理实用编程的课程特点。

--- 
# 线性回归Python实现问答记录
## 我：
在实现线性回归的Python代码时，我完成了最小二乘法的手动实现、sklearn库调用对比，以及残差分析，但在可视化模块遇到两个核心问题：一是绘制残差图时抛出“ValueError: x and y must be the same size”，定位到plt.scatter(residuals, y_pred)这行；二是即使修复维度问题，仍偶发“UserWarning: FixedFormatter should only be used together with FixedLocator”的警告。从Matplotlib的数据映射逻辑来看，scatter的x和y必须是等长数组，但我最初误将残差数组做了维度变换，这可能是核心问题。另外，我注意到残差图的坐标轴刻度设置用了plt.xticks固定值，却未匹配对应的Locator，这会触发警告。还有两个算法层面的疑问：第一，手动实现的最小二乘法和sklearn的LinearRegression在计算逻辑上具体有哪些差异？第二，为什么多重共线性会导致回归系数不稳定，这个现象的本质原因是什么？

## AI：
### 一、最小二乘法实现的逻辑差异
1. 计算精度：手动实现通常基于numpy矩阵运算，未做数值稳定性优化；sklearn的LinearRegression采用QR分解/奇异值分解（SVD），能避免矩阵求逆的数值误差，适配病态矩阵。
2. 功能覆盖：手动实现多仅支持基本线性回归（无正则化、无缺失值处理）；sklearn内置缺失值过滤、特征中心化、正则化扩展（如Ridge/Lasso）。
3. 结果输出：手动实现需自行计算R²、残差等指标；sklearn封装了coef_、intercept_、score()等接口，直接返回评估指标。
4. 效率层面：手动实现的循环/矩阵运算在大数据量下效率低；sklearn基于Cython优化，支持批量处理和并行计算。

### 二、多重共线性导致系数不稳定的原因
无多重共线性时，特征矩阵X列满秩，(XᵀX)可逆，回归系数β=(XᵀX)⁻¹Xᵀy有唯一解；存在多重共线性时，XᵀX接近奇异矩阵（行列式趋近于0），其逆矩阵会被微小的数值波动放大，导致β的估计值剧烈变化。
本质是特征间的线性依赖使“每个特征对因变量的独立贡献”无法被准确分离，比如两个高度相关的特征（如身高/体重），其中一个的系数增大时，另一个可通过减小来补偿，最终系数失去物理意义。

### 三、残差图维度与刻度警告问题
scatter的x和y必须是长度完全一致的一维数组，需检查残差数组是否因广播、切片操作导致维度错位（如二维数组转一维）；
刻度警告源于xticks的FixedFormatter与自动生成的Locator不匹配，需显式创建FixedLocator并绑定，或改用自动刻度适配逻辑，避免手动固定刻度值。

## 我：
补充一个细节：我在验证线性回归的残差正态性时，发现Shapiro-Wilk检验的p值约0.04，略低于0.05的显著性水平，但Q-Q图显示残差基本贴合正态分布，这应该是样本量较小（n=50）导致的检验偏差。我尝试用α=0.1的显著性水平来判断，这个阈值调整是否合理？另外，在模型评估中，我仅计算了R²和均方误差（MSE），是否需要补充调整R²来完善分析？

## AI：
### 一、残差正态性检验的阈值调整
这个调整是合理的：样本量较小时（n<100），Shapiro-Wilk检验的统计功效低，易出现“假显著”（即实际正态却拒绝原假设）；α=0.1能降低第一类错误的概率，且Q-Q图的可视化证据优先级高于单一检验值，符合“小样本下结合图形与检验”的分析原则。若需更严谨，可增大样本量或改用Kolmogorov-Smirnov检验（对小样本更稳健）。

### 二、模型评估的完善
需要补充调整R²：普通R²会随特征数量增加而单调上升（即使新增无关特征），无法反映模型的实际解释力；调整R²引入了特征数量的惩罚项，公式为R²_adj = 1 - [(1-R²)(n-1)/(n-p-1)]（n为样本量，p为特征数），能更客观地评估“特征对因变量的真实解释能力”，尤其适合多特征回归模型。

## 我：
最后一个问题：在可视化模块中，我注意到matplotlib的subplots_adjust(hspace=0.3)能调整子图间距，但在Jupyter Notebook中偶发子图重叠。从渲染原理来看，subplots_adjust的工作机制是什么？如何在保证子图布局合理的同时避免Jupyter中的重叠问题？

## AI：
### 一、subplots_adjust的工作机制
subplots_adjust通过调整子图的间距（hspace/vspace）、边距（left/right/top/bottom）来分配画布空间：默认布局会自动分配子图位置，但当子图标题、坐标轴标签过长时，自动分配的空间不足；该函数手动修改子图的bbox（边界框）参数，重新划分画布的像素区域，让子图间保留足够空白。

### 二、Jupyter Notebook中子图重叠的规避
1. 优先使用tight_layout()替代手动调整，它会自动根据标签/标题长度计算最优间距，适配Notebook的渲染画布；
2. 若需手动调整，结合figsize参数增大画布尺寸（如fig, ax = plt.subplots(2,2, figsize=(10,8))），避免空间挤压；
3. 显式指定subplots_adjust的所有参数（left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3），而非仅调整hspace；
4. 在Notebook中使用%matplotlib inline魔法命令时，添加dpi参数（如%matplotlib inline dpi=100），提升渲染分辨率，减少布局错位；
5. 若仍重叠，改用GridSpec手动划分子图区域，精准控制每个子图的位置和尺寸。