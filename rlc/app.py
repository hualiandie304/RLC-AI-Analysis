import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 页面配置
st.set_page_config(page_title="RLC谐振AI分析系统", layout="wide")

# RLC串联电流-频率拟合函数
def rlc_current(f, L, C, R, U):
    omega = 2 * np.pi * f
    z = np.sqrt(R**2 + (omega*L - 1/(omega*C))**2)
    return U / z

# 主界面
st.title("RLC串联谐振实验AI分析系统")
st.subheader("物理约束拟合+异常值自动剔除")

# 数据上传
uploaded_file = st.file_uploader("上传Excel数据（列名：频率(Hz), 电流(A)）", type=["xlsx", "xls"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    f_data = df.iloc[:, 0].values
    I_data = df.iloc[:, 1].values
    
    # 异常值检测
    clf = IsolationForest(contamination=0.05, random_state=42)
    outliers = clf.fit_predict(I_data.reshape(-1, 1))
    f_clean = f_data[outliers == 1]
    I_clean = I_data[outliers == 1]
    
    # 物理约束拟合
    bounds = ([0.001, 1e-9, 1, 0], [1, 1e-5, 1000, 20])
    try:
        popt, pcov = curve_fit(rlc_current, f_clean, I_clean, bounds=bounds)
        L_fit, C_fit, R_fit, U_fit = popt
        f0_fit = 1 / (2 * np.pi * np.sqrt(L_fit * C_fit))
        Q_fit = 1 / (2 * np.pi * f0_fit * C_fit * R_fit)
        
        # 结果展示
        st.success("AI分析完成")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("拟合电感L", f"{L_fit*1000:.2f} mH")
        with col2: st.metric("拟合电容C", f"{C_fit*1e9:.2f} nF")
        with col3: st.metric("谐振频率f0", f"{f0_fit:.2f} Hz")
        with col4: st.metric("品质因数Q", f"{Q_fit:.2f}")
        
        # 绘制曲线
        fig, ax = plt.subplots(figsize=(10, 6))
        f_plot = np.linspace(min(f_clean), max(f_clean), 1000)
        I_plot = rlc_current(f_plot, L_fit, C_fit, R_fit, U_fit)
        ax.scatter(f_clean, I_clean, label="有效实验数据")
        ax.plot(f_plot, I_plot, label="AI拟合曲线", linewidth=2)
        ax.set_xlabel("频率(Hz)")
        ax.set_ylabel("电流(A)")
        ax.set_title("RLC串联谐振频率-电流特性曲线")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        st.info(f"剔除异常数据{len(f_data)-len(f_clean)}组，有效数据{len(f_clean)}组")
        
    except:
        st.error("拟合失败，请检查数据格式")
