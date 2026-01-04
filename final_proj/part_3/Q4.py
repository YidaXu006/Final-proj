import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ===================== 全局设置：解决Matplotlib中文显示问题 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换为系统支持的中文字体（如SimHei/微软雅黑）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ===================== 1. 参数设置与初始化 =====================
# 物理参数
L = 1.0               # 杆长 (m)
alpha = 0.01          # 热扩散系数 (m²/s)
t_targets = [0, 0.1, 0.5, 1, 5]  # 需绘制的时间点
t_max = max(t_targets)   # 最大模拟时间

# 数值离散参数
nx = 100              # 空间网格数
dx = L / (nx - 1)     # 空间步长
r = 0.4               # 稳定性条件 r = α*dt/dx² ≤ 0.5，取0.4保证稳定
dt = r * dx**2 / alpha# 时间步长
nt = int(np.ceil(t_max / dt))  # 向上取整确保覆盖最大时间

# 空间网格
x = np.linspace(0, L, nx)

# 初始条件: T(x,0) = sin(πx)
T_current = np.sin(np.pi * x)
# 边界条件: T(0,t)=0, T(1,t)=0（始终满足）

# 存储目标时间点的数值解和解析解
T_num = np.zeros((len(t_targets), nx))
T_analytic = np.zeros_like(T_num)

# 初始化t=0时刻的结果
T_num[0, :] = T_current.copy()
T_analytic[0, :] = np.sin(np.pi * x)  # t=0时解析解与初始条件一致

# ===================== 2. 显式差分格式求解（精准匹配目标时间点） =====================
t_current = 0.0  # 当前模拟时间
target_idx = 1   # 下一个要记录的目标时间索引

for n in range(1, nt+1):
    # 显式差分格式核心计算
    T_next = T_current.copy()
    T_next[1:-1] = T_current[1:-1] + r * (T_current[2:] - 2*T_current[1:-1] + T_current[:-2])
    
    # 更新时间和温度场
    t_current += dt
    T_current = T_next.copy()
    
    # 精准匹配目标时间点（允许微小浮点误差）
    if target_idx < len(t_targets):
        target_t = t_targets[target_idx]
        if abs(t_current - target_t) < 1e-4 or t_current >= target_t:
            # 记录当前时间点的数值解
            T_num[target_idx, :] = T_current.copy()
            # 计算对应时间的解析解
            T_analytic[target_idx, :] = np.exp(-alpha * np.pi**2 * target_t) * np.sin(np.pi * x)
            target_idx += 1

# ===================== 3. 数值误差分析 =====================
error = np.abs(T_num - T_analytic)
print("===== 数值误差分析 =====")
for i, t in enumerate(t_targets):
    max_err = np.max(error[i, :])
    mean_err = np.mean(error[i, :])
    print(f"时间 t={t}s: 最大误差={max_err:.6f}, 平均误差={mean_err:.6f}")

# ===================== 4. 可视化：指定时间点温度分布对比 =====================
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue', 'magenta', 'cyan']
linestyles_num = '--'
linestyles_ana = '-'

for i in range(len(t_targets)):
    t = t_targets[i]
    # 绘制数值解
    plt.plot(x, T_num[i, :], color=colors[i], linestyle=linestyles_num, 
             label=f'数值解 t={t}s')
    # 绘制解析解
    plt.plot(x, T_analytic[i, :], color=colors[i], linestyle=linestyles_ana, 
             label=f'解析解 t={t}s')

plt.xlabel('位置 x (m)')
plt.ylabel('温度 T (℃)')
plt.title('一维热传导方程数值解与解析解对比')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===================== 5. 可视化：3D曲面图（温度-时间-空间） =====================
# 重构完整时间序列和温度场（用于3D绘图）
t_all = np.linspace(0, t_max, nt+1)
T_all = np.zeros((nt+1, nx))
T_all[0, :] = np.sin(np.pi * x)

# 重新计算完整时间序列的温度场
T_recompute = np.sin(np.pi * x)
for n in range(1, nt+1):
    T_next = T_recompute.copy()
    T_next[1:-1] = T_recompute[1:-1] + r * (T_recompute[2:] - 2*T_recompute[1:-1] + T_recompute[:-2])
    T_all[n, :] = T_next.copy()
    T_recompute = T_next.copy()

# 绘制3D曲面
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
X, T_mesh = np.meshgrid(x, t_all)
surf = ax.plot_surface(X, T_mesh, T_all, cmap='hot', alpha=0.8, linewidth=0)

ax.set_xlabel('位置 x (m)')
ax.set_ylabel('时间 t (s)')
ax.set_zlabel('温度 T (℃)')
ax.set_title('温度随时间和空间的演化（3D曲面图）')
fig.colorbar(surf, label='温度 (℃)', shrink=0.8)
plt.tight_layout()
plt.show()

# ===================== 6. 可视化：热力图（时间-空间） =====================
plt.figure(figsize=(10, 6))
im = plt.imshow(T_all, extent=[0, L, 0, t_max], aspect='auto', 
                cmap='hot', origin='lower')
plt.xlabel('位置 x (m)')
plt.ylabel('时间 t (s)')
plt.title('温度演化热力图（热-冷色图）')
plt.colorbar(im, label='温度 (℃)')
plt.tight_layout()
plt.show()

# ===================== 7. 可选：动态演示热量扩散过程 =====================
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot(x, T_all[0, :], 'r-', label='温度分布')
ax.set_xlabel('位置 x (m)')
ax.set_ylabel('温度 T (℃)')
ax.set_ylim(0, 1.1)
ax.set_title('一维热传导热量扩散过程')
ax.legend()
ax.grid(True, alpha=0.3)

def update(frame):
    line.set_ydata(T_all[frame, :])
    return line,

# 每10帧更新一次，加快动画速度（避免卡顿）
ani = animation.FuncAnimation(fig, update, frames=range(0, nt+1, 10), 
                              interval=50, blit=True, repeat=False)
plt.tight_layout()
plt.show()