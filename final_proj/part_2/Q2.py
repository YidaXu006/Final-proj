import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre, erf  # 导入erf函数

# ===================== 1. 基本数值积分方法 =====================
def trapezoidal_rule(f, a, b, n):
    """梯形法计算积分∫[a,b]f(x)dx"""
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1]))

def simpson_rule(f, a, b, n):
    """Simpson法计算积分∫[a,b]f(x)dx（n需为偶数）"""
    if n % 2 != 0:
        raise ValueError("Simpson法要求n为偶数")
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h/3 * (y[0] + y[-1] + 4*np.sum(y[1::2]) + 2*np.sum(y[2:-1:2]))

# 定义被积函数
f = lambda x: np.sin(x)
exact_I = 2  # 积分精确值
n_list = [4, 8, 16, 32, 64]  # 分割数
a, b = 0, np.pi  # 积分区间

# 计算不同n下的积分结果与误差
trap_errors = []
simp_errors = []
for n in n_list:
    # 梯形法
    trap_val = trapezoidal_rule(f, a, b, n)
    trap_errors.append(np.abs(trap_val - exact_I))
    # Simpson法
    simp_val = simpson_rule(f, a, b, n)
    simp_errors.append(np.abs(simp_val - exact_I))

# 绘制误差双对数图
plt.figure(figsize=(8, 6))
h_list = (b - a) / np.array(n_list)  # 步长h
# 梯形法：O(h²)，拟合斜率应为2
plt.loglog(h_list, trap_errors, 'o-', label='梯形法')
# Simpson法：O(h⁴)，拟合斜率应为4
plt.loglog(h_list, simp_errors, 's-', label='Simpson法')
# 绘制理论阶数参考线
h_ref = np.logspace(-3, 0, 100)
plt.loglog(h_ref, 0.1*h_ref**2, 'k--', label='O(h²)')
plt.loglog(h_ref, 0.01*h_ref**4, 'r--', label='O(h⁴)')
plt.xlabel('步长h')
plt.ylabel('误差')
plt.title('梯形法/Simpson法误差双对数图')
plt.legend()
plt.grid(which='both', alpha=0.3)
plt.savefig('integration_error.png', dpi=300)
plt.show()

# ===================== 2. 高斯积分应用 =====================
def gaussian_quadrature(f, a, b, n_points):
    """在区间[a,b]上用n点高斯-勒让德积分计算∫f(x)dx"""
    x, w = roots_legendre(n_points)  # 获取Legendre节点和权重
    # 映射到区间[a,b]
    t = 0.5 * (x * (b - a) + (a + b))
    weight = 0.5 * (b - a) * w
    return np.sum(f(t) * weight)

# 原函数
g = lambda x: np.exp(-x**2)
exact_gauss = np.sqrt(np.pi)  # 精确值

# 1. 截断到[-L,L]的精度分析
L_list = [1, 2, 3, 4, 5]
trunc_errors = []
for L in L_list:
    # 用20点高斯积分计算[-L,L]
    val = gaussian_quadrature(g, -L, L, 20)
    trunc_errors.append(np.abs(val - exact_gauss))

# 2. 无限区间映射到[-1,1]（变换t = x/√(1-x²)）
def transformed_g(t):
    """变换后的函数：原函数*雅可比行列式"""
    x = t / np.sqrt(1 - t**2)
    jacobian = 1 / (1 - t**2)**(3/2)
    return g(x) * jacobian

# 用20点高斯积分计算映射后的积分
mapped_val = gaussian_quadrature(transformed_g, -1, 1, 20)
mapped_error = np.abs(mapped_val - exact_gauss)

# 输出结果
print("===== 高斯积分结果 =====")
print(f"精确值: {exact_gauss:.8f}")
print("截断到[-L,L]的误差：")
for L, err in zip(L_list, trunc_errors):
    print(f"L={L}: 误差={err:.8f}")
print(f"无限区间映射后的积分值: {mapped_val:.8f}, 误差={mapped_error:.8f}")

# ===================== 3. 蒙特卡洛积分（选做） =====================
def monte_carlo_integral(f, a, b, N):
    """蒙特卡洛法计算∫[a,b]f(x)dx"""
    x = np.random.uniform(a, b, N)
    y = f(x)
    return (b - a) * np.mean(y)

# 被积函数
mc_f = lambda x: np.exp(-x**2)
# 修正：用scipy.special.erf计算精确值
exact_mc = np.sqrt(np.pi)/2 * (1 - erf(0))  # ∫₀¹e^(-x²)dx的精确值（约0.7468241328）

# 不同N下的结果
N_mc_list = [100, 1000, 10000, 100000, 1000000]
mc_errors = []
for N in N_mc_list:
    val = monte_carlo_integral(mc_f, 0, 1, N)
    mc_errors.append(np.abs(val - exact_mc))

# 输出结果
print("\n===== 蒙特卡洛积分结果 =====")
print(f"精确值: {exact_mc:.8f}")
for N, err in zip(N_mc_list, mc_errors):
    print(f"N={N}: 误差={err:.8f}")