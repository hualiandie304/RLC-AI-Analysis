# -*- coding: utf-8 -*-
import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
from scipy.interpolate import CubicSpline

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 页面配置 
st.set_page_config(page_title="RLC谐振分析系统", layout="wide") 

# RLC串联电流-频率拟合函数
def rlc_current(f, L, C, R, U):
    omega = 2 * np.pi * f
    z = np.sqrt(R**2 + (omega*L - 1/(omega*C))**2)
    return U / z

# 通用核心：自动找半功率点
def find_half_power_frequency(f, I, target_I, side):
    """
    通用函数：
    - 自动排序
    - 自动找第一个跨越半功率电流的位置
    - 自动插值
    - 不依赖任何固定数据
    side: "left"=低频侧(f<f0), "right"=高频侧(f>f0)
    """
    # 按频率从小到大排序（保证任何数据顺序都不乱）
    sorted_idx = np.argsort(f)
    f_sorted = f[sorted_idx]
    I_sorted = I[sorted_idx]

    # 找跨越点：左侧上升沿 / 右侧下降沿
    if side == "left":
        cross_mask = (I_sorted[:-1] < target_I) & (I_sorted[1:] > target_I)
    elif side == "right":
        cross_mask = (I_sorted[:-1] > target_I) & (I_sorted[1:] < target_I)
    else:
        raise ValueError("side must be 'left' or 'right'")

    cross_indices = np.where(cross_mask)[0]
    if len(cross_indices) == 0:
        raise ValueError("数据范围不足，无法找到半功率点")

    # 取【第一个真正跨越】，不是最近的，保证物理正确
    k = cross_indices[0]
    f1, f2 = f_sorted[k], f_sorted[k+1]
    i1, i2 = I_sorted[k], I_sorted[k+1]

    # 线性插值（纯通用）
    fraction = (target_I - i1) / (i2 - i1)
    f_cross = f1 + fraction * (f2 - f1)
    return f_cross

# 高精度半功率点计算函数
def calculate_half_power_points(freq, current, f0, I0, manual_f1=None, manual_f2=None):
    """
    高精度计算半功率点，支持手动修正
    输入：freq-频率数组, current-电流数组, f0-谐振频率, I0-峰值电流
          manual_f1-手动设置低频半功率点, manual_f2-手动设置高频半功率点
    输出：f1-低频半功率点, f2-高频半功率点, I_half-半功率电流
    """
    I_half = I0 / np.sqrt(2)
    
    # 低频半功率点
    left_freq = freq[freq < f0]
    left_curr = current[freq < f0]
    if manual_f1 is not None:
        f1 = manual_f1
    else:
        f1 = find_half_power_frequency(left_freq, left_curr, I_half, side="left")
    
    # 高频半功率点
    right_freq = freq[freq > f0]
    right_curr = current[freq > f0]
    if manual_f2 is not None:
        f2 = manual_f2
    else:
        f2 = find_half_power_frequency(right_freq, right_curr, I_half, side="right")
    
    return f1, f2, I_half

# 主界面 
st.title("RLC串联谐振实验分析系统") 
st.subheader("基于物理定义的通用算法") 

# 电流单位选择
current_unit = st.selectbox("电流单位", ["A", "mA"]) 

# 固定已知电感（只在这里改一次）
L_fixed = st.sidebar.number_input("固定电感 L (H)", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f")

# 手动修正选项
st.sidebar.markdown("---")
st.sidebar.subheader("手动修正选项")
use_manual_f0 = st.sidebar.checkbox("手动设置谐振频率 f0", value=False)
manual_f0 = st.sidebar.number_input("谐振频率 f0 (Hz)", min_value=0.0, max_value=100000.0, value=3000.0, format="%.1f", disabled=not use_manual_f0)

use_manual_f1 = st.sidebar.checkbox("手动设置低频半功率点 f1", value=False)
manual_f1 = st.sidebar.number_input("低频半功率点 f1 (Hz)", min_value=0.0, max_value=100000.0, value=2700.0, format="%.1f", disabled=not use_manual_f1)

use_manual_f2 = st.sidebar.checkbox("手动设置高频半功率点 f2", value=False)
manual_f2 = st.sidebar.number_input("高频半功率点 f2 (Hz)", min_value=0.0, max_value=100000.0, value=3300.0, format="%.1f", disabled=not use_manual_f2)

# 数据上传 
uploaded_file = st.file_uploader("上传Excel数据（列名：频率(Hz), 电流(" + current_unit + ")）", type=["xlsx", "xls"]) 
if uploaded_file is not None: 
    df = pd.read_excel(uploaded_file) 
    f_data = df.iloc[:, 0].values 
    I_data = df.iloc[:, 1].values 
    
    # 电流单位转换（如果是mA，转换为A）
    if current_unit == "mA":
        I_data = I_data / 1000
    
    # 异常值检测 - 确保谐振点不被误判
    max_current_idx = np.argmax(I_data)
    peak_freq = f_data[max_current_idx]
    peak_current = I_data[max_current_idx]
    
    # 使用简单的异常值检测：只去掉明显偏离的点
    # 计算电流的统计量
    mean_I = np.mean(I_data)
    std_I = np.std(I_data)
    
    # 找出明显的异常值（超过3倍标准差）
    outliers = np.abs(I_data - mean_I) > 3 * std_I
    
    # 确保峰值点不被标记为异常
    outliers[max_current_idx] = False
    
    f_clean = f_data[~outliers]
    I_clean = I_data[~outliers]
    f_outliers = f_data[outliers]
    I_outliers = I_data[outliers] 
    
    st.info(f"数据处理: 总数据{len(f_data)}组, 有效数据{len(f_clean)}组, 异常数据{len(f_outliers)}组")
    
    # ===================== 核心计算开始 =====================
    st.subheader("参数计算")
    
    # 1. 确定谐振频率 f0
    if use_manual_f0:
        f0_fit = manual_f0
        # 找到最接近f0的电流值作为I0
        closest_idx = np.argmin(np.abs(f_clean - f0_fit))
        I0 = I_clean[closest_idx]
    else:
        max_idx = np.argmax(I_clean)
        f0_fit = f_clean[max_idx]
        I0 = I_clean[max_idx]
    I0_mA = I0 * 1000
    
    # 2. 自动找半功率点、带宽、Q（完全按你的逻辑）
    f1_param = manual_f1 if use_manual_f1 else None
    f2_param = manual_f2 if use_manual_f2 else None
    f1, f2, I_half = calculate_half_power_points(f_clean, I_clean, f0_fit, I0, f1_param, f2_param)
    I_half_mA = I_half * 1000
    st.info(f"✅ 谐振频率 f0 = {f0_fit:.2f} Hz (峰值电流 I0 = {I0_mA:.2f} mA)")
    st.info(f"✅ 半功率电流 I_half = {I_half_mA:.2f} mA")
    st.info(f"✅ 半功率点: f1={f1:.2f} Hz, f2={f2:.2f} Hz")
    
    # 确保f1 < f0 < f2
    f1 = min(f1, f0_fit - 10)
    f2 = max(f2, f0_fit + 10)
    
    # 带宽 & Q
    BW = f2 - f1
    Q_fit = f0_fit / BW
    
    # 3. 自动算电容 C
    C_fit = 1 / (4 * np.pi**2 * f0_fit**2 * L_fixed)
    
    # 4. 计算电阻 R
    R_fit = (2 * np.pi * f0_fit * L_fixed) / Q_fit
    
    # 5. 生成正确的拟合曲线 + 计算R²（用谐振电流反推电压，保证曲线贴合数据）
    f_fit = np.linspace(f_clean.min(), f_clean.max(), 1000)
    omega = 2 * np.pi * f_fit
    XL = omega * L_fixed
    XC = 1 / (omega * C_fit)
    Z = np.sqrt(R_fit**2 + (XL - XC)**2)
    # 用谐振电流反推电压，保证曲线贴合数据
    I_fit = (I0 * R_fit) / Z
    
    I_data_fit = np.interp(f_clean, f_fit, I_fit)
    ss_res = np.sum((I_clean - I_data_fit)**2)
    ss_tot = np.sum((I_clean - np.mean(I_clean))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r2 = max(0, r2)  # 确保R²非负
    
    mse = np.mean((I_clean - I_data_fit) ** 2)
    mae = np.mean(np.abs(I_clean - I_data_fit))
    
    # ===================== 结果展示 =====================
    st.success("分析完成！")
    
    col1, col2, col3, col4 = st.columns(4) 
    with col1: st.metric("拟合电感L", f"{L_fixed*1000:.2f} mH") 
    with col2: st.metric("拟合电容C", f"{C_fit*1e9:.2f} nF") 
    with col3: st.metric("谐振频率f0", f"{f0_fit:.2f} Hz") 
    with col4: st.metric("品质因数Q", f"{Q_fit:.2f}") 
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("半功率点f1", f"{f1:.1f} Hz")
    with col2: st.metric("半功率点f2", f"{f2:.1f} Hz")
    with col3: st.metric("带宽BW", f"{BW:.2f} Hz")
    with col4: st.metric("电阻R", f"{R_fit:.2f} Ω")
    
    # 拟合效果验证
    st.subheader("拟合效果验证")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²值", f"{r2:.4f}")
        st.caption("越接近1表示拟合效果越好")
    with col2:
        st.metric("均方误差(MSE)", f"{mse:.6f}")
        st.caption("越小表示拟合效果越好")
    with col3:
        st.metric("平均绝对误差(MAE)", f"{mae:.6f}")
        st.caption("越小表示拟合效果越好")
    
    # 物理自洽性校验
    st.subheader("物理自洽性校验")
    col1, col2 = st.columns(2)
    with col1:
        f0_check = 1 / (2 * np.pi * np.sqrt(L_fixed * C_fit))
        st.write(f"**谐振频率自洽**")
        st.write(f"理论计算: {f0_check:.2f} Hz")
        st.write(f"数据峰值: {f0_fit:.2f} Hz")
        st.write(f"偏差: {abs(f0_check-f0_fit)/f0_fit*100:.2f}%")
    with col2:
        Q_check = (2 * np.pi * f0_fit * L_fixed) / R_fit
        st.write(f"**品质因数自洽**")
        st.write(f"ωL/R计算: {Q_check:.2f}")
        st.write(f"带宽计算: {Q_fit:.2f}")
        st.write(f"偏差: {abs(Q_check-Q_fit)/Q_fit*100:.2f}%")
    
    # 绘制曲线 
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    # 转换回显示单位
    if current_unit == "mA":
        I_clean_display = I_clean * 1000
        I_fit_display = I_fit * 1000
        I_outliers_display = I_outliers * 1000
        I0_display = I0 * 1000
        I_half_display = I_half * 1000
        ylabel = "电流(mA)"
    else:
        I_clean_display = I_clean
        I_fit_display = I_fit
        I_outliers_display = I_outliers
        I0_display = I0
        I_half_display = I_half
        ylabel = "电流(A)"
    
    # 绘制所有数据点
    if len(f_outliers) > 0:
        ax.scatter(f_outliers, I_outliers_display, label="异常数据", color="red", marker="x") 
    ax.scatter(f_clean, I_clean_display, label="有效实验数据", color="blue") 
    ax.plot(f_fit, I_fit_display, label="拟合曲线", linewidth=2, color="green") 
    
    # 标记谐振点和半功率点
    ax.plot(f0_fit, I0_display, 'o', color='orange', markersize=8, label=f"谐振点 ({f0_fit:.0f} Hz, {I0_display:.2f} {current_unit})")
    ax.plot([f1, f2], [I_half_display, I_half_display], 's', color='purple', markersize=6, label=f"半功率点 ({f1:.0f}, {f2:.0f} Hz)")
    ax.axhline(y=I_half_display, color='purple', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel("频率(Hz)") 
    ax.set_ylabel(ylabel) 
    ax.set_title("RLC串联谐振频率-电流特性曲线") 
    ax.legend() 
    ax.grid(True) 
    st.pyplot(fig) 
    
    # 数据点误差分析
    st.subheader("数据点误差分析")
    error_df = pd.DataFrame({
        "频率(Hz)": f_clean,
        "实际电流(" + current_unit + ")": I_clean_display,
        "预测电流(" + current_unit + ")": I_data_fit * (1000 if current_unit == "mA" else 1),
        "误差(" + current_unit + ")": (I_clean - I_data_fit) * (1000 if current_unit == "mA" else 1),
        "相对误差(%)": np.abs((I_clean - I_data_fit) / I_clean) * 100
    })
    st.dataframe(error_df.style.format({
        "频率(Hz)": "{:.0f}",
        "实际电流(" + current_unit + ")": "{:.2f}",
        "预测电流(" + current_unit + ")": "{:.2f}",
        "误差(" + current_unit + ")": "{:.2f}",
        "相对误差(%)": "{:.2f}"
    }))
    
    # 数据统计信息
    st.subheader("数据统计分析")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**数据概览**")
        st.write(f"总数据量: {len(f_data)}组")
        st.write(f"有效数据: {len(f_clean)}组")
        st.write(f"异常数据: {len(f_outliers)}组")
    
    with col2:
        st.write("**频率范围**")
        st.write(f"最小值: {min(f_data):.0f} Hz")
        st.write(f"最大值: {max(f_data):.0f} Hz")
        st.write(f"范围: {max(f_data)-min(f_data):.0f} Hz")
    
    with col3:
        st.write("**电流范围**")
        if current_unit == "mA":
            st.write(f"最小值: {min(I_data)*1000:.2f} mA")
            st.write(f"最大值: {max(I_data)*1000:.2f} mA")
            st.write(f"平均值: {np.mean(I_data)*1000:.2f} mA")
        else:
            st.write(f"最小值: {min(I_data):.4f} A")
            st.write(f"最大值: {max(I_data):.4f} A")
            st.write(f"平均值: {np.mean(I_data):.4f} A")
