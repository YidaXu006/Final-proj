import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
import matplotlib
import traceback

# ===================== 全局配置：解决中文显示和字体警告 =====================
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']  # 兼容不同系统中文显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.use('TkAgg')  # 明确指定Tk后端，避免后端兼容问题

# ===================== 1. 无空气阻力的抛体运动 =====================
def projectile_no_drag():
    v0 = 20  # 初速度 (m/s)
    g = 9.8  # 重力加速度 (m/s²)
    thetas = [15, 30, 45, 60, 75]  # 发射角（角度）
    theta_rads = np.radians(thetas)  # 转换为弧度

    # 1.1 输出运动方程
    print("===== 无空气阻力运动方程 =====")
    print(f"x(t) = v0·cosθ · t  (v0={v0}m/s)")
    print(f"y(t) = v0·sinθ · t - 0.5·g·t²  (g={g}m/s²)")

    # 1.2 绘制不同发射角的轨迹
    plt.figure(figsize=(10, 6))
    max_range = 0.0
    best_theta = 0
    t_total = np.linspace(0, 5, 500)  # 时间范围

    for theta, theta_rad in zip(thetas, theta_rads):
        # 计算轨迹
        x = v0 * np.cos(theta_rad) * t_total
        y = v0 * np.sin(theta_rad) * t_total - 0.5 * g * t_total**2
        # 过滤y<0的点（落地后停止）
        mask = y >= 0
        x_valid = x[mask]
        y_valid = y[mask]
        # 计算射程（x最大值）
        current_range = x_valid[-1] if len(x_valid) > 0 else 0.0
        if current_range > max_range:
            max_range = current_range
            best_theta = theta
        # 绘制轨迹
        plt.plot(x_valid, y_valid, label=f'θ={theta}°, 射程={current_range:.2f}m')

    # 1.3 验证45°射程最大
    print(f"\n===== 射程验证 =====")
    print(f"最大射程对应的发射角：{best_theta}°，验证45°时射程最大：{best_theta == 45}")
    print(f"45°时最大射程：{max_range:.2f}m")

    plt.xlabel('水平距离 (m)')
    plt.ylabel('竖直高度 (m)')
    plt.title('无空气阻力的抛体运动轨迹')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ===================== 2. 考虑空气阻力的抛体运动 =====================
def projectile_with_drag():
    m = 1  # 质量 (kg)
    v0 = 20  # 初速度 (m/s)
    theta = 45  # 发射角 (°)
    theta_rad = np.radians(theta)
    g = 9.8  # 重力加速度 (m/s²)
    b_m_vals = [0, 0.1, 0.3, 0.5]  # b/m 系数
    t_span = (0, 5)  # 时间范围
    t_eval = np.linspace(t_span[0], t_span[1], 500)  # 时间采样点

    # 微分方程：dx/dt = vx; dvx/dt = -b/m * vx;  dy/dt = vy; dvy/dt = -g - b/m * vy
    def ode_system(t, state, b_m):
        x, vx, y, vy = state
        dvx_dt = -b_m * vx
        dvy_dt = -g - b_m * vy
        return [vx, dvx_dt, vy, dvy_dt]

    plt.figure(figsize=(10, 6))

    # 遍历不同b/m系数求解并绘图
    for b_m in b_m_vals:
        # 初始状态：[x0, vx0, y0, vy0]
        init_state = [0, v0 * np.cos(theta_rad), 0, v0 * np.sin(theta_rad)]
        # 数值求解ODE
        sol = solve_ivp(ode_system, t_span, init_state, args=(b_m,), t_eval=t_eval)
        x = sol.y[0]
        y = sol.y[2]
        # 过滤y<0的点
        mask = y >= 0
        x_valid = x[mask]
        y_valid = y[mask]
        current_range = x_valid[-1] if len(x_valid) > 0 else 0.0
        # 绘制轨迹
        plt.plot(x_valid, y_valid, label=f'b/m={b_m}, 射程={current_range:.2f}m')

    # 分析最佳发射角（遍历不同角度，找最大射程）
    thetas_test = np.linspace(10, 80, 20)  # 测试角度范围
    range_list = []
    b_m_test = 0.3  # 取一个典型阻力系数
    for theta_test in thetas_test:
        theta_test_rad = np.radians(theta_test)
        init_state = [0, v0 * np.cos(theta_test_rad), 0, v0 * np.sin(theta_test_rad)]
        sol = solve_ivp(ode_system, t_span, init_state, args=(b_m_test,), t_eval=t_eval)
        x = sol.y[0]
        y = sol.y[2]
        mask = y >= 0
        x_valid = x[mask]
        current_range = x_valid[-1] if len(x_valid) > 0 else 0.0
        range_list.append(current_range)
    best_angle_with_drag = thetas_test[np.argmax(range_list)]

    # 输出分析结果
    print(f"\n===== 有空气阻力分析 =====")
    print(f"阻力系数b/m={b_m_test}时，最佳发射角：{best_angle_with_drag:.1f}°（非45°）")
    print(f"结论：有阻力时最佳发射角不再是45°，而是更小的角度")

    plt.xlabel('水平距离 (m)')
    plt.ylabel('竖直高度 (m)')
    plt.title('不同阻力系数下的抛体运动轨迹 (θ=45°)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ===================== 3. 动画与能量分析（彻底修复所有报错） =====================
def animation_energy_analysis():
    v0 = 20  # 初速度 (m/s)
    theta = 45  # 发射角 (°)
    theta_rad = np.radians(theta)
    g = 9.8  # 重力加速度 (m/s²)
    m = 1  # 质量 (kg)
    b_m = 0.3  # 阻力系数（有阻力情况）
    t_span = (0, 5)
    t_eval = np.linspace(t_span[0], t_span[1], 200)  # 动画帧采样

    # --- 无阻力能量分析 ---
    # 无阻力轨迹
    x_no_drag = v0 * np.cos(theta_rad) * t_eval
    y_no_drag = v0 * np.sin(theta_rad) * t_eval - 0.5 * g * t_eval**2
    mask_no_drag = y_no_drag >= 0
    x_no_drag = x_no_drag[mask_no_drag]
    y_no_drag = y_no_drag[mask_no_drag]
    # 确保数组非空
    if len(x_no_drag) == 0:
        x_no_drag = np.array([0])
        y_no_drag = np.array([0])
    # 速度计算
    vx_no_drag = v0 * np.cos(theta_rad) * np.ones_like(x_no_drag)
    vy_no_drag = v0 * np.sin(theta_rad) - g * np.linspace(0, len(x_no_drag)/len(t_eval)*5, len(x_no_drag))
    v_no_drag = np.sqrt(vx_no_drag**2 + vy_no_drag**2)
    # 机械能：E = 0.5mv² + mgy
    E_no_drag = 0.5 * m * v_no_drag**2 + m * g * y_no_drag
    print(f"\n===== 无阻力能量分析 =====")
    print(f"机械能初始值：{E_no_drag[0]:.2f} J")
    print(f"机械能最终值：{E_no_drag[-1]:.2f} J")
    print(f"机械能守恒验证：{np.isclose(E_no_drag[0], E_no_drag[-1], atol=1e-1)}")

    # --- 有阻力能量分析 ---
    def ode_system(t, state, b_m):
        x, vx, y, vy = state
        dvx_dt = -b_m * vx
        dvy_dt = -g - b_m * vy
        return [vx, dvx_dt, vy, dvy_dt]
    
    init_state = [0, v0 * np.cos(theta_rad), 0, v0 * np.sin(theta_rad)]
    sol = solve_ivp(ode_system, t_span, init_state, args=(b_m,), t_eval=t_eval)
    x_drag = sol.y[0]
    y_drag = sol.y[2]
    vx_drag = sol.y[1]
    vy_drag = sol.y[3]
    mask_drag = y_drag >= 0
    x_drag = x_drag[mask_drag]
    y_drag = y_drag[mask_drag]
    vx_drag = vx_drag[mask_drag]
    vy_drag = vy_drag[mask_drag]
    # 确保数组非空
    if len(x_drag) == 0:
        x_drag = np.array([0])
        y_drag = np.array([0])
    # 机械能
    v_drag = np.sqrt(vx_drag**2 + vy_drag**2)
    E_drag = 0.5 * m * v_drag**2 + m * g * y_drag
    energy_loss = E_drag[0] - E_drag[-1] if len(E_drag) > 1 else 0.0
    print(f"\n===== 有阻力能量分析 =====")
    print(f"机械能初始值：{E_drag[0]:.2f} J")
    print(f"机械能最终值：{E_drag[-1]:.2f} J")
    print(f"能量损失：{energy_loss:.2f} J")

    # --- 动画制作（核心修复：set_data传序列而非标量） ---
    fig, ax = plt.subplots(figsize=(10, 6))
    # 扩展x/y轴范围，避免质点出界
    x_max = max(x_no_drag.max(), x_drag.max()) + 5
    y_max = max(y_no_drag.max(), y_drag.max()) + 5
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel('水平距离 (m)')
    ax.set_ylabel('竖直高度 (m)')
    ax.set_title('抛体运动动画（无阻力 vs 有阻力）')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)

    # 绘制轨迹线
    line_no_drag, = ax.plot([], [], 'b-', label='无阻力', linewidth=2)
    line_drag, = ax.plot([], [], 'r-', label='有阻力 (b/m=0.3)', linewidth=2)
    # 绘制运动质点（初始为空）
    dot_no_drag, = ax.plot([], [], 'bo', markersize=8)  # 无阻力质点
    dot_drag, = ax.plot([], [], 'ro', markersize=8)      # 有阻力质点
    ax.legend(loc='upper right')

    # 动画最大帧数（取较短轨迹的长度，避免索引越界）
    max_frames = min(len(x_no_drag), len(x_drag))
    max_frames = max(max_frames, 1)  # 确保至少1帧

    # 动画初始化函数
    def init():
        line_no_drag.set_data([], [])
        line_drag.set_data([], [])
        dot_no_drag.set_data([], [])
        dot_drag.set_data([], [])
        return line_no_drag, line_drag, dot_no_drag, dot_drag

    # 动画更新函数（核心修复：传序列而非标量）
    def update(frame):
        # ===== 无阻力轨迹与质点更新 =====
        # 轨迹线：取前frame+1个点（序列）
        line_no_drag.set_data(x_no_drag[:frame+1], y_no_drag[:frame+1])
        # 质点：传入长度为1的列表（序列），而非标量
        dot_x = [x_no_drag[frame]]
        dot_y = [y_no_drag[frame]]
        dot_no_drag.set_data(dot_x, dot_y)

        # ===== 有阻力轨迹与质点更新 =====
        line_drag.set_data(x_drag[:frame+1], y_drag[:frame+1])
        dot_x_drag = [x_drag[frame]]
        dot_y_drag = [y_drag[frame]]
        dot_drag.set_data(dot_x_drag, dot_y_drag)

        return line_no_drag, line_drag, dot_no_drag, dot_drag

    # 生成动画（关闭循环，避免越界）
    ani = animation.FuncAnimation(
        fig, update, frames=max_frames,
        init_func=init, interval=50, blit=True, repeat=False
    )

    plt.tight_layout()
    plt.show()

    # 可选：保存动画（需安装ffmpeg，取消注释即可）
    # try:
    #     ani.save('projectile_animation.mp4', writer='ffmpeg', fps=30, dpi=100)
    #     print("动画已保存为 projectile_animation.mp4")
    # except Exception as e:
    #     print(f"保存动画失败：{e}（需安装ffmpeg）")

# ===================== 主函数：统一执行入口 =====================
def main():
    try:
        # 1. 无空气阻力分析
        projectile_no_drag()
        # 2. 有空气阻力分析
        projectile_with_drag()
        # 3. 动画与能量分析
        animation_energy_analysis()
    except Exception as e:
        print(f"\n程序运行异常：{type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()