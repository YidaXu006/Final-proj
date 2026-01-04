import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.fftpack import fft, fftfreq

# ===================== 1. 方波的傅里叶展开 =====================
# 1.1 定义方波函数
def square_wave(x):
    """周期2π的方波函数"""
    x = x % (2 * np.pi)  # 周期化处理
    if 0 < x < np.pi:
        return 1
    elif np.pi < x < 2 * np.pi:
        return -1
    else:
        return 0  # 间断点值不影响积分

# 向量化方波函数（方便绘图）
square_wave_vec = np.vectorize(square_wave)

# 1.2 计算傅里叶系数 an 和 bn
def compute_fourier_coeffs(n_max):
    """计算方波的傅里叶系数 an, bn"""
    an_list = []
    bn_list = []
    for n in range(n_max + 1):
        # 计算an: (1/π)∫(-π到π) f(x)cos(nx)dx
        def an_integrand(x):
            return square_wave(x) * np.cos(n * x)
        an = (1 / np.pi) * quad(an_integrand, -np.pi, np.pi)[0]
        an_list.append(an)
        
        # 计算bn: (1/π)∫(-π到π) f(x)sin(nx)dx
        def bn_integrand(x):
            return square_wave(x) * np.sin(n * x)
        bn = (1 / np.pi) * quad(bn_integrand, -np.pi, np.pi)[0]
        bn_list.append(bn)
    return np.array(an_list), np.array(bn_list)

# 验证系数（n=0到10）
an, bn = compute_fourier_coeffs(10)
print("===== 方波傅里叶系数验证 =====")
print(f"a0 = {an[0]:.6f} (理论值0)")
for n in range(1, 11):
    if n % 2 == 1:  # 奇数n
        theoretical_bn = 4 / (np.pi * n)
        print(f"n={n}(奇数): bn={bn[n]:.6f}, 理论值={theoretical_bn:.6f}")
    else:  # 偶数n
        print(f"n={n}(偶数): bn={bn[n]:.6f} (理论值0)")

# 1.3 方波傅里叶级数展开式
def square_fourier_series(x, N):
    """方波前N项傅里叶级数"""
    an, bn = compute_fourier_coeffs(N)
    series = an[0] / 2  # a0/2
    for n in range(1, N + 1):
        series += an[n] * np.cos(n * x) + bn[n] * np.sin(n * x)
    return series

# 1.4 绘制原函数与不同项数的傅里叶级数对比（吉布斯现象）
x = np.linspace(-np.pi, 3 * np.pi, 2000)  # 扩展区间观察周期性
y_original = square_wave_vec(x)

# 待绘制的项数
N_list = [3, 5, 11, 51]
plt.figure(figsize=(12, 8))
plt.plot(x, y_original, label='原方波', color='black', linewidth=2)

for N in N_list:
    y_series = square_fourier_series(x, N)
    plt.plot(x, y_series, label=f'前{N}项傅里叶级数', alpha=0.7, linewidth=1.5)

plt.title('方波傅里叶级数展开（吉布斯现象）', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(np.pi, color='gray', linestyle='--', alpha=0.5)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('square_wave_fourier.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n===== 吉布斯现象说明 =====")
print("在方波间断点（如x=π）附近，傅里叶级数会出现固定幅度的过冲，且项数增加时过冲位置向间断点靠近，但幅度不消失（约为原函数跳变的9%），这就是吉布斯现象。")

# ===================== 2. 三角波的傅里叶展开 =====================
# 2.1 定义三角波函数
def triangle_wave(x):
    """周期2π的三角波 g(x)=|x|, -π<x<π"""
    x = x % (2 * np.pi)  # 周期化
    if x > np.pi:
        x = 2 * np.pi - x
    return np.abs(x)

# 向量化三角波函数
triangle_wave_vec = np.vectorize(triangle_wave)

# 2.2 三角波傅里叶系数计算
def triangle_fourier_coeffs(n_max):
    """计算三角波的傅里叶系数"""
    an_list = []
    bn_list = []
    for n in range(n_max + 1):
        # an: (1/π)∫(-π到π) |x|cos(nx)dx
        def an_integrand(x):
            return np.abs(x) * np.cos(n * x)
        an = (1 / np.pi) * quad(an_integrand, -np.pi, np.pi)[0]
        an_list.append(an)
        
        # bn: 三角波是偶函数，sin项系数为0（验证）
        def bn_integrand(x):
            return np.abs(x) * np.sin(n * x)
        bn = (1 / np.pi) * quad(bn_integrand, -np.pi, np.pi)[0]
        bn_list.append(bn)
    return np.array(an_list), np.array(bn_list)

# 2.3 三角波傅里叶级数
def triangle_fourier_series(x, N):
    """三角波前N项傅里叶级数"""
    an, bn = triangle_fourier_coeffs(N)
    series = an[0] / 2
    for n in range(1, N + 1):
        series += an[n] * np.cos(n * x) + bn[n] * np.sin(n * x)
    return series

# 2.4 绘制三角波收敛过程
x = np.linspace(-2 * np.pi, 2 * np.pi, 2000)
y_original = triangle_wave_vec(x)

N_triangle = [1, 3, 5, 10]
plt.figure(figsize=(12, 8))
plt.plot(x, y_original, label='原三角波', color='black', linewidth=2)

for N in N_triangle:
    y_series = triangle_fourier_series(x, N)
    plt.plot(x, y_series, label=f'前{N}项傅里叶级数', alpha=0.7, linewidth=1.5)

plt.title('三角波傅里叶级数收敛过程', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('g(x)', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('triangle_wave_fourier.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n===== 三角波收敛更快的原因 =====")
print("1. 三角波是连续函数（仅一阶导数间断），方波是不连续函数；")
print("2. 三角波的傅里叶系数衰减速度为O(1/n²)，方波为O(1/n)；")
print("3. 系数衰减越快，级数收敛越快，因此三角波傅里叶级数更快收敛到原函数。")

# ===================== 3. 简单信号合成 =====================
# 3.1 定义复合信号
def composite_signal(t):
    """s(t) = sin(2π·3t) + 0.5sin(2π·7t) + 0.3sin(2π·11t)"""
    return np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t) + 0.3 * np.sin(2 * np.pi * 11 * t)

# 3.2 绘制时域波形
t = np.linspace(0, 2, 2000, endpoint=False)  # t∈[0,2]
s_t = composite_signal(t)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, s_t)
plt.title('复合信号时域波形', fontsize=14)
plt.xlabel('t (s)', fontsize=12)
plt.ylabel('s(t)', fontsize=12)
plt.grid(alpha=0.3)

# 3.3 FFT频谱分析
N = len(t)
Fs = N / 2  # 采样频率（总时长2s，N个点）
fft_result = fft(s_t)
freqs = fftfreq(N, 1/Fs)  # 频率轴
amplitude = np.abs(fft_result) / N * 2  # 幅度谱（单边）

# 只取正频率部分
positive_freq_idx = freqs >= 0
freqs_pos = freqs[positive_freq_idx]
amplitude_pos = amplitude[positive_freq_idx]

# 绘制频谱图（移除use_line_collection参数适配低版本matplotlib）
plt.subplot(2, 1, 2)
plt.stem(freqs_pos, amplitude_pos, basefmt='b-')
plt.xlim(0, 15)  # 聚焦0-15Hz
plt.title('复合信号频谱图', fontsize=14)
plt.xlabel('频率 (Hz)', fontsize=12)
plt.ylabel('幅度', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('composite_signal_fft.png', dpi=300, bbox_inches='tight')
plt.show()

# 验证频率成分
print("\n===== 复合信号频率验证 =====")
# 找出幅度峰值对应的频率
peak_freqs = freqs_pos[np.argsort(amplitude_pos)[-3:]]
peak_amps = amplitude_pos[np.argsort(amplitude_pos)[-3:]]
for f, amp in sorted(zip(peak_freqs, peak_amps)):
    print(f"频率: {f:.1f}Hz, 幅度: {amp:.3f} (理论值: 3Hz(1),7Hz(0.5),11Hz(0.3))")