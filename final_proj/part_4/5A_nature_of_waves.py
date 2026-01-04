import numpy as np
import matplotlib.pyplot as plt
import time

# -------------------------- 全局设置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
plt.rcParams['figure.figsize'] = (12, 8)    # 画布尺寸
plt.rcParams['figure.facecolor'] = '#f8f9fa'  # 画布背景色（浅灰，更护眼）
x = np.linspace(0, 20, 1000)  # 空间坐标定义域

# 默认参数配置（支持一键重置）
DEFAULT_PARAMS = {
    "A1": 1.0,
    "A2": 1.0,
    "phase_diff": 0.0,
    "lamda": 4.0
}
current_params = DEFAULT_PARAMS.copy()  # 当前参数缓存（全局变量）

# -------------------------- 核心计算函数 --------------------------
def wave_superposition(A1, A2, phi1, phi2, lamda):
    """计算两列简谐波及叠加波的位移"""
    k = 2 * np.pi / lamda  # 波数
    y1 = A1 * np.sin(k * x + phi1)
    y2 = A2 * np.sin(k * x + phi2)
    y_sum = y1 + y2
    return y1, y2, y_sum

# -------------------------- 绘图函数 --------------------------
def plot_waves(A1, A2, phase_diff, lamda):
    """绘制三列波形：波1、波2、叠加波"""
    phi1 = 0  # 固定波1初相位，简化相位差调节
    phi2 = phi1 + phase_diff
    y1, y2, y_sum = wave_superposition(A1, A2, phi1, phi2, lamda)
    
    # 清空画布（避免多次绘制重叠）
    plt.clf()
    ax = plt.gca()
    # 设置图表属性
    ax.set_xlabel('空间坐标 $x$（无单位）', fontsize=12, fontweight='bold')
    ax.set_ylabel('位移 $y$（无单位）', fontsize=12, fontweight='bold')
    ax.set_title(
        f'波的叠加与干涉现象演示\n振幅A1={A1:.1f}, A2={A2:.1f} | 相位差Δφ={phase_diff:.2f}rad | 波长λ={lamda:.1f}',
        fontsize=14, fontweight='bold', pad=20, color='#2c3e50'
    )
    ax.grid(True, linestyle='--', alpha=0.7, color='#bdc3c7')
    ax.axhline(y=0, color='#2c3e50', linewidth=1.5)  # 零位移基准线
    
    # 绘制波形（优化配色，更易区分）
    ax.plot(x, y1, '#3498db', linewidth=2, label='波1 $y_1$', alpha=0.8)
    ax.plot(x, y2, '#e74c3c', linewidth=2, linestyle='--', label='波2 $y_2$', alpha=0.8)
    ax.plot(x, y_sum, '#2ecc71', linewidth=3, label='叠加波 $y=y_1+y_2$', alpha=0.9)
    
    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True, facecolor='white')
    plt.tight_layout()
    plt.show(block=False)  # 非阻塞显示，允许继续输入参数

# -------------------------- 输入优化函数 --------------------------
def print_separator():
    print("\n" + "-"*70 + "\n")

def show_quick_params():
    print("📌 快捷参数模板（直接复制数值输入）：")
    print("  1. 完全相长干涉：A1=1.0, A2=1.0, 相位差=0.0, 波长=4.0")
    print("  2. 完全相消干涉：A1=1.0, A2=1.0, 相位差=3.14, 波长=4.0")
    print("  3. 振幅不等干涉：A1=2.0, A2=1.0, 相位差=1.57, 波长=5.0")

def get_valid_input(prompt, min_val, max_val, default, unit="", dtype=float):
    # 构建提示文本（包含单位、范围、默认值）
    unit_text = f"（{unit}）" if unit else ""
    prompt_text = f"{prompt}{unit_text}\n  范围：{min_val}~{max_val} | 当前默认值：{default:.2f}\n  请输入数值（直接回车使用默认值）："
    
    while True:
        try:
            user_input = input(prompt_text).strip()
            # 空输入使用默认值
            if not user_input:
                val = default
                print(f"  ✅ 使用默认值：{val:.2f}{unit_text}")
                return val
            # 校验输入类型
            val = dtype(user_input)
            # 范围校验
            if min_val <= val <= max_val:
                print(f"  ✅ 输入有效：{val:.2f}{unit_text}")
                return val
            else:
                print(f"  ❌ 输入超出范围！请输入{min_val}到{max_val}之间的数值。")
        except ValueError:
            print(f"  ❌ 输入无效！请输入{dtype.__name__}类型的数值（如 1.0、3.14）。")

def reset_default_params():
    """重置当前参数为初始默认值"""
    global current_params  # 声明使用全局变量
    current_params = DEFAULT_PARAMS.copy()
    print("  🔄 参数已重置为初始默认值！")
    time.sleep(0.5)  # 短暂延迟，让用户看到提示

# -------------------------- 主交互逻辑 --------------------------
def interactive_script():
    """优化后的主交互逻辑"""
    global current_params  # 关键：声明使用全局的current_params变量
    print("="*70)
    print("          🎯波的叠加与干涉交互式演示工具")
    print("="*70)
    print("✨ 操作说明：")
    print("  1. 输入参数时直接按回车，将使用当前默认值；")
    print("  2. 输入'reset'可重置所有参数为初始默认值；")
    print("  3. 输入'q'可随时退出程序；")
    print("  4. 输入'help'可查看快捷参数模板；")
    print("📚 核心提示：")
    print("  - 相位差输入0 → 相长干涉（振幅增强）；")
    print("  - 相位差输入3.14（π）→ 相消干涉（振幅抵消）；")
    print("="*70)
    
    while True:
        print_separator()
        # 接收前置指令（退出/重置/帮助）
        cmd = input("请输入指令（q=退出 | reset=重置参数 | help=快捷参数 | 回车=继续输入参数）：").strip().lower()
        if cmd == 'q':
            print("👋 程序已退出！")
            plt.close('all')
            break
        elif cmd == 'reset':
            reset_default_params()
            continue
        elif cmd == 'help':
            show_quick_params()
            continue
        elif cmd != "":
            print(f"  ❌ 未知指令：{cmd}，请重新输入！")
            continue
        
        # 进度提示
        print("\n📝 开始输入参数（共4项）：")
        time.sleep(0.3)
        
        # 1. 波1振幅
        A1 = get_valid_input(
            prompt="1/4 波1振幅（A1）",
            min_val=0.1, max_val=3.0,
            default=current_params["A1"],
            unit="（振幅单位）"
        )
        
        # 2. 波2振幅
        A2 = get_valid_input(
            prompt="2/4 波2振幅（A2）",
            min_val=0.1, max_val=3.0,
            default=current_params["A2"],
            unit="（振幅单位）"
        )
        
        # 3. 相位差
        phase_diff = get_valid_input(
            prompt="3/4 两列波相位差（Δφ）",
            min_val=0, max_val=2*np.pi,
            default=current_params["phase_diff"],
            unit="rad（弧度）"
        )
        
        # 4. 波长
        lamda = get_valid_input(
            prompt="4/4 波长（λ）",
            min_val=2.0, max_val=8.0,
            default=current_params["lamda"],
            unit="（长度单位）"
        )
        
        # 绘制前提示
        print_separator()
        print("🎨 正在绘制波形图，请稍候...")
        time.sleep(0.5)
        
        # 绘制波形
        plot_waves(A1, A2, phase_diff, lamda)
        
        # 更新当前默认参数（下次可直接复用）
        current_params = {
            "A1": A1,
            "A2": A2,
            "phase_diff": phase_diff,
            "lamda": lamda
        }
        
        # 绘制完成提示
        print("✅ 波形图已生成！可继续输入参数查看新波形。")

# -------------------------- 启动程序 --------------------------
if __name__ == "__main__":
    interactive_script()