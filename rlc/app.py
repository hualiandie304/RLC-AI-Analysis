import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import requests
import json
import base64
import io
from datetime import datetime

# ===================== 基础配置 =====================
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="RLC串联谐振实验AI智能辅助系统", layout="wide")

# ===================== 全局变量 =====================
API_URL_RESPONSES = "https://ark.cn-beijing.volces.com/api/v1/responses"
API_URL_CHAT = "https://ark.cn-beijing.volces.com/api/v1/chat/completions"

# 常用模型/endpoint列表
COMMON_ENDPOINTS = [
    {"name": "doubao-seed-2-0-pro-260215", "label": "豆包 Seed 2.0 Pro (260215)"},
    {"name": "doubao-seed-2-0-pro", "label": "豆包 Seed 2.0 Pro"},
    {"name": "doubao-3-5-mini", "label": "豆包 3.5 Mini"},
    {"name": "doubao-3-5-turbo", "label": "豆包 3.5 Turbo"},
    {"name": "doubao-3-5-pro", "label": "豆包 3.5 Pro"},
]

# ===================== 预设RLC元件库 =====================
PRESET_RLC = [
    {"name": "标准教学套件 A (常用)", "L": 0.010, "C": 0.28e-6, "R": 5.0,  "f0": 3010, "Q": 37.9, "note": "高校物理实验最常用配置，谐振频率约3kHz"},
    {"name": "低频教学套件 B",        "L": 0.100,  "C": 1.0e-6,  "R": 10.0, "f0": 503,  "Q": 31.6, "note": "适合观察低频谐振特性"},
    {"name": "高频教学套件 C",        "L": 0.001,  "C": 0.01e-6, "R": 2.0,  "f0": 50330,"Q": 15.8, "note": "高频实验，注意屏蔽干扰"},
    {"name": "高Q值套件 D",           "L": 0.050,  "C": 0.1e-6,  "R": 3.0,  "f0": 2251, "Q": 235.7,"note": "高Q值，谐振峰非常尖锐"},
    {"name": "低Q值套件 E (欠阻尼)",   "L": 0.010,  "C": 1.0e-6,  "R": 50.0, "f0": 1592, "Q": 2.0,  "note": "低Q值，曲线较平缓，适合观察欠阻尼"},
    {"name": "自定义参数",             "L": 0.010,  "C": 0.28e-6, "R": 5.0,  "f0": 3010, "Q": 37.9, "note": "根据实际元件自行设置"},
]

# ===================== 全流程实验步骤 =====================
EXPERIMENT_STEPS = [
    {
        "title": "第一步：元件与仪器准备",
        "content": """
**所需仪器和元件：**
- 信号发生器（或函数发生器）1台
- 交流毫伏表（或双踪示波器）1台
- 定值电感 L 1个
- 定值电容 C 1个
- 定值电阻 R 1个
- 连接导线若干
- 万用表（可选，用于预先测量元件实际值）

**检查要点：**
1. 检查仪器外观是否完好，电源指示灯是否正常
2. 记录元件标称值：电感 L、电容 C、电阻 R
3. 如有万用表，可预先实测各元件实际值，与标称值对比
        """,
        "ai_check": "请帮我检查：实验开始前，仪器和元件是否准备齐全？"
    },
    {
        "title": "第二步：电路连线",
        "content": """
**RLC串联电路连接方法：**

1. 将信号发生器的输出端串联连接：电阻 R → 电感 L → 电容 C，形成闭合回路
2. 交流毫伏表（或示波器通道1）并联在信号发生器输出端，用于监测输入电压
3. 另一个毫伏表（或示波器通道2）并联在电阻 R 两端，用于测量电阻电压（与电流成正比）

**关键注意事项：**
- ⚠️ 所有仪器和元件必须串联成闭合回路，不能有开路
- ⚠️ 毫伏表必须并联在被测元件两端，不能串联
- ⚠️ 接线柱必须拧紧，避免接触不良导致数据异常
- ⚠️ 仪器接地端应连接在一起，避免共模干扰
- 建议保持接线整洁，避免导线交叉引起电磁干扰
        """,
        "ai_check": "我已经完成电路连线，请帮我分析是否存在接线错误或操作问题？"
    },
    {
        "title": "第三步：仪器参数设置",
        "content": """
**仪器参数设置要点：**

**信号发生器：**
1. 输出波形：正弦波
2. 输出电压：建议 1-5 V（根据仪器和元件参数调整，保持信号幅度不变）
3. 初始频率：设置在预估谐振频率的50%左右
4. 保持输出电压在测量过程中恒定不变（重要！）

**毫伏表/示波器：**
1. 量程选择：先选较大量程，根据读数逐步减小
2. 示波器模式：若使用示波器，选择合适的时基和电压档
3. 确保输入耦合方式为 AC 耦合

**预估谐振频率（非常有用）：**
根据公式 f0 = 1 / (2π√(LC))，可以先估算谐振频率的大致位置
例如：L=10mH, C=0.28μF → f0 ≈ 3000Hz
这样可以在谐振频率附近密集测量，远离谐振频率处稀疏测量，提高效率
        """,
        "ai_check": "请帮我评估当前的仪器参数设置是否合理？"
    },
    {
        "title": "第四步：开始数据测量",
        "content": """
**测量步骤与要点：**

1. **打开仪器电源，预热3-5分钟**，使仪器进入稳定工作状态

2. **设置初始频率**（约为预估谐振频率的50%），记录此时：
   - 输入电压（应保持恒定）
   - 电阻两端电压（或直接读取电流值）

3. **逐步增加频率，逐点测量**：
   - 远离谐振频率：步长可较大（如 50-200 Hz）
   - 接近谐振频率：步长缩小（如 10-20 Hz）
   - 谐振频率附近：步长最小（如 2-5 Hz），以捕捉到尖锐的谐振峰

4. **记录数据**：每组数据应包含「频率 f」和「电流 I（或电阻电压）」

5. **数据量建议**：测量 30-60 个数据点，覆盖 0.5f0 到 1.5f0 范围

**操作注意事项：**
- ⚠️ 调节频率时动作要缓慢，读数前等待仪器示数稳定
- ⚠️ 每改变一次频率，检查信号发生器的输出电压是否保持恒定
- ⚠️ 如发现数据异常（突然跳变），先重复测量该点确认
- ⚠️ 注意仪器量程变化，防止过载或读数溢出
        """,
        "ai_check": "测量过程中发现某个数据点异常，可能是什么原因？"
    },
    {
        "title": "第五步：数据分析与处理",
        "content": """
**数据分析步骤：**

1. **绘制电流-频率曲线**，观察曲线形状是否为单峰、对称的谐振曲线

2. **寻找谐振频率 f0**：电流最大点对应的频率即为谐振频率

3. **计算半功率点 f1、f2**：电流为最大值的 1/√2 时对应的两个频率

4. **计算品质因数 Q**：Q = f0 / (f2 - f1)

5. **计算等效参数**：
   - 由 f0 反算实际电容 C（若已知 L）
   - 由 Q 反算等效串联电阻 R

6. **验证与分析**：
   - 比较实测 f0 与理论计算值，分析误差来源
   - 分析 Q 值的物理意义和影响因素
   - 判断是否存在异常数据点，如有需分析原因

**常见误差来源：**
- 元件实际值与标称值的偏差
- 信号源输出电压未能保持恒定
- 接线接触电阻的影响
- 仪器读数的不确定度
- 环境电磁干扰
        """,
        "ai_check": "请帮我分析当前的数据处理过程是否正确？"
    },
    {
        "title": "第六步：实验总结与报告",
        "content": """
**实验报告应包含以下内容：**

1. **实验目的**
   - 观察RLC串联电路的谐振现象
   - 测量谐振频率、品质因数等参数
   - 理解谐振特性的物理意义

2. **实验原理**
   - RLC串联电路的阻抗特性
   - 谐振条件和谐振频率公式
   - 品质因数的物理意义

3. **实验仪器与元件**
   - 列出使用的仪器型号和元件参数

4. **实验数据记录**
   - 原始测量数据表
   - 数据曲线图

5. **数据处理与分析**
   - 谐振频率的确定
   - 品质因数的计算
   - 误差分析

6. **实验结论**
   - 实验得到的主要结论
   - 与理论预测的对比
   - 对异常现象的分析（如有）

7. **讨论与建议**
   - 实验过程中遇到的问题
   - 改进实验的建议
   - 个人收获与体会
        """,
        "ai_check": "请帮我撰写一份完整的实验总结报告"
    },
]

# ===================== 会话状态初始化 =====================
DEFAULTS = {
    'experiment_step': 1,
    'f_data': None,
    'I_data': None,
    'ai_analysis_result': None,
    'student_feedback': '',
    'modification_applied': False,
    'final_summary': None,
    'fitting_results': None,
    'api_key': None,
    'debug_mode': False,
    'api_status': 'not_checked',
    'api_status_message': '',
    'endpoint': '',
    'use_custom_endpoint': True,
    'api_format': 'chat',
    'timeout': 30,
    'chat_messages': [],
    'ai_analysis_context': '',
    'selected_preset_idx': 0,
    'design_L': 0.010,
    'design_C': 0.28e-6,
    'design_R': 5.0,
    'design_Vin': 1.0,
    'single_point_history': [],
    'wiring_check_result': None,
    'active_tab': 'guide',
    'guide_step_index': 0,
    'auto_input_frequency': '',
    'auto_input_current': '',
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ===================== 辅助函数 =====================
def find_half_power_frequency(f, I, target_I, side):
    sorted_idx = np.argsort(f)
    f_sorted = f[sorted_idx]
    I_sorted = I[sorted_idx]
    if side == "left":
        cross_mask = (I_sorted[:-1] < target_I) & (I_sorted[1:] > target_I)
    elif side == "right":
        cross_mask = (I_sorted[:-1] > target_I) & (I_sorted[1:] < target_I)
    else:
        raise ValueError("side must be 'left' or 'right'")
    cross_indices = np.where(cross_mask)[0]
    if len(cross_indices) == 0:
        raise ValueError("数据范围不足，无法找到半功率点")
    k = cross_indices[0]
    f1, f2 = f_sorted[k], f_sorted[k+1]
    i1, i2 = I_sorted[k], I_sorted[k+1]
    fraction = (target_I - i1) / (i2 - i1)
    f_cross = f1 + fraction * (f2 - f1)
    return f_cross

def calculate_half_power_points(freq, current, f0, I0, manual_f1=None, manual_f2=None):
    I_half = I0 / np.sqrt(2)
    left_freq = freq[freq < f0]
    left_curr = current[freq < f0]
    if manual_f1 is not None:
        f1 = manual_f1
    else:
        f1 = find_half_power_frequency(left_freq, left_curr, I_half, side="left")
    right_freq = freq[freq > f0]
    right_curr = current[freq > f0]
    if manual_f2 is not None:
        f2 = manual_f2
    else:
        f2 = find_half_power_frequency(right_freq, right_curr, I_half, side="right")
    return f1, f2, I_half

def enhanced_outlier_check(freq, current):
    outliers_mask = np.zeros(len(freq), dtype=bool)
    reasons = []
    max_idx = np.argmax(current)
    mean_I = np.mean(current)
    std_I = np.std(current)
    if std_I > 0:
        sigma_outliers = np.abs(current - mean_I) > 3 * std_I
    else:
        sigma_outliers = np.zeros(len(freq), dtype=bool)
    sigma_outliers[max_idx] = False
    outliers_mask = outliers_mask | sigma_outliers
    if np.any(sigma_outliers):
        for i in np.where(sigma_outliers)[0]:
            reasons.append(f"⚠️ 3σ检测：f={freq[i]:.1f}Hz 电流偏离均值超过3σ")
    window = 3
    for i in range(len(current)):
        if abs(i - max_idx) > window:
            continue
        if i < 1 or i >= len(current) - 1:
            continue
        prev_val = current[i-1]
        next_val = current[i+1]
        curr_val = current[i]
        if curr_val < prev_val * 0.5 and curr_val < next_val * 0.5:
            outliers_mask[i] = True
            reasons.append(f"⚠️ 物理规则检测：f={freq[i]:.1f}Hz 谐振峰附近电流断崖式下跌")
    return np.any(outliers_mask), outliers_mask, reasons

# ===================== RLC理论曲线计算 =====================
def compute_theoretical_curve(L, C, R, Vin, f_min, f_max, num_points=200):
    """根据RLC参数计算理论谐振曲线"""
    freq = np.linspace(f_min, f_max, num_points)
    omega = 2 * np.pi * freq
    omega0 = 1 / np.sqrt(L * C)
    f0 = omega0 / (2 * np.pi)
    Q = (1 / R) * np.sqrt(L / C)
    # I = Vin / sqrt(R^2 + (omega*L - 1/(omega*C))^2)
    Z = np.sqrt(R**2 + (omega * L - 1 / (omega * C))**2)
    I = Vin / Z
    I0 = Vin / R  # 谐振时阻抗为R
    BW = f0 / Q if Q > 0 else 0
    return freq, I, f0, Q, BW, I0

# ===================== 单点数据合理性检测（增强版） =====================
def compute_theoretical_current(freq_point, L, C, R, Vin):
    """根据RLC元件参数计算某频率下的理论电流"""
    omega = 2 * np.pi * freq_point
    Z = np.sqrt(R**2 + (omega * L - 1 / (omega * C))**2)
    I_theory = Vin / Z
    return I_theory

def filter_outlier_points(freq_data, current_data):
    """过滤已有数据中的明显异常点，返回干净的数据子集（用于趋势预测）"""
    if freq_data is None or len(freq_data) < 5:
        return freq_data, current_data, np.zeros(len(freq_data) if freq_data is not None else 0, dtype=bool)
    # 基于3σ准则做初步过滤
    mean_I = np.mean(current_data)
    std_I = np.std(current_data)
    if std_I == 0:
        return freq_data, current_data, np.zeros(len(freq_data), dtype=bool)
    sigma_mask = np.abs(current_data - mean_I) > 3 * std_I
    # 基于局部一致性：若某点与左右邻居差距过大，也标为异常
    sorted_idx = np.argsort(freq_data)
    f_sorted = freq_data[sorted_idx]
    I_sorted = current_data[sorted_idx]
    local_mask = np.zeros(len(freq_data), dtype=bool)
    window = 2
    for i in range(len(freq_data)):
        if i < window or i >= len(freq_data) - window:
            continue
        neighbors_left = I_sorted[i-window:i]
        neighbors_right = I_sorted[i+1:i+1+window]
        neighbors = np.concatenate([neighbors_left, neighbors_right])
        neighbor_mean = np.mean(neighbors)
        if neighbor_mean > 0 and abs(I_sorted[i] - neighbor_mean) / neighbor_mean > 0.5:
            local_mask[sorted_idx[i]] = True
    combined_mask = sigma_mask | local_mask
    clean_freq = freq_data[~combined_mask]
    clean_current = current_data[~combined_mask]
    return clean_freq, clean_current, combined_mask

def check_single_point_reasonable(freq_point, current_point, freq_data, current_data, L=None, C=None, R=None, Vin=1.0):
    """
    综合判断单点数据是否合理，返回三重基准：
    - theoretical: 基于RLC参数计算的理论电流（最可靠物理基准）
    - expected_from_clean: 基于过滤异常点后的实测数据插值得到的预期值
    - deviation_vs_theory: 相对于理论值的偏差百分比
    - deviation_vs_data: 相对于干净实测数据的偏差百分比
    """
    result = {
        'theoretical': None,
        'expected_from_clean': None,
        'deviation_vs_theory': None,
        'deviation_vs_data': None,
        'clean_data_count': 0,
        'filtered_out_count': 0
    }
    # 基准1：理论值计算
    if L is not None and C is not None and R is not None:
        try:
            result['theoretical'] = compute_theoretical_current(freq_point, L, C, R, Vin)
            if result['theoretical'] > 0:
                result['deviation_vs_theory'] = abs(current_point - result['theoretical']) / result['theoretical'] * 100
        except:
            pass
    # 基准2：过滤后的实测数据插值
    if freq_data is not None and len(freq_data) >= 3:
        clean_freq, clean_current, outlier_mask = filter_outlier_points(freq_data, current_data)
        result['clean_data_count'] = len(clean_freq)
        result['filtered_out_count'] = np.sum(outlier_mask)
        if len(clean_freq) >= 2:
            sorted_idx = np.argsort(clean_freq)
            f_sorted = clean_freq[sorted_idx]
            I_sorted = clean_current[sorted_idx]
            if freq_point < f_sorted[0] or freq_point > f_sorted[-1]:
                if freq_point < f_sorted[0]:
                    idx = 0
                else:
                    idx = len(f_sorted) - 1
                result['expected_from_clean'] = I_sorted[idx]
            else:
                result['expected_from_clean'] = np.interp(freq_point, f_sorted, I_sorted)
            if result['expected_from_clean'] is not None and result['expected_from_clean'] > 0:
                result['deviation_vs_data'] = abs(current_point - result['expected_from_clean']) / result['expected_from_clean'] * 100
    # 兼容旧调用：返回(expected, deviation)格式
    if result['theoretical'] is not None:
        return result['theoretical'], result['deviation_vs_theory'], result
    elif result['expected_from_clean'] is not None:
        return result['expected_from_clean'], result['deviation_vs_data'], result
    else:
        return None, None, result

# ===================== API验证函数 =====================
def check_api_connection(api_key, endpoint, api_format="chat", timeout=10):
    if not api_key or len(api_key.strip()) < 10:
        return False, "API密钥太短或为空"
    if not endpoint or len(endpoint.strip()) < 3:
        return False, "Endpoint为空或太短"
    api_key = api_key.strip()
    endpoint = endpoint.strip()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    if api_format == "chat":
        url = API_URL_CHAT
        data = {
            "model": endpoint,
            "messages": [{"role": "user", "content": "请回复：连接测试成功"}],
            "max_tokens": 10
        }
    else:
        url = API_URL_RESPONSES
        data = {
            "model": endpoint,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "请回复：连接测试成功"}]}]
        }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=timeout)
        resp.raise_for_status()
        resp_json = resp.json()
        if api_format == "chat":
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                return True, "✅ API连接测试成功"
            else:
                return False, f"API响应格式异常：{resp.text[:200]}"
        else:
            if "output" in resp_json and "text" in resp_json["output"]:
                return True, "✅ API连接测试成功"
            else:
                return False, f"API响应格式异常：{resp.text[:200]}"
    except requests.exceptions.Timeout:
        return False, "❌ 连接超时，请检查网络或增加超时时间"
    except requests.exceptions.ConnectionError:
        return False, "❌ 无法连接到API服务器，请检查网络"
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = resp.json()
            return False, f"❌ HTTP错误：{error_detail.get('error', {}).get('message', str(e))}"
        except:
            return False, f"❌ HTTP错误：{str(e)}"
    except Exception as e:
        return False, f"❌ 未知错误：{str(e)}"

# ===================== AI调用函数 =====================
def get_api_key():
    if st.session_state.api_key and len(st.session_state.api_key) >= 10:
        return st.session_state.api_key
    try:
        return st.secrets.get("DOUBAO_API_KEY", "")
    except:
        return ""

def call_ai(prompt, max_tokens=2000):
    api_key = get_api_key()
    if api_key:
        api_key = api_key.strip()
    if not api_key or len(api_key) < 10:
        return "❌ 错误：未配置API密钥！\n\n请在侧边栏输入豆包API密钥。"
    if not st.session_state.endpoint:
        return "❌ 错误：未配置Endpoint！\n\n请在侧边栏输入您的推理接入点Endpoint。"
    if st.session_state.api_format == "chat":
        api_url = API_URL_CHAT
        data = {
            "model": st.session_state.endpoint,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": max_tokens
        }
    else:
        api_url = API_URL_RESPONSES
        data = {
            "model": st.session_state.endpoint,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
        }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    debug_info = f"\n🔍 [调试信息]\n- API URL: {api_url}\n- API格式: {st.session_state.api_format}\n- Endpoint: {st.session_state.endpoint}\n- 密钥长度: {len(api_key)}\n- 密钥前缀: {api_key[:15]}...\n"
    try:
        if st.session_state.debug_mode:
            st.info("正在发送API请求...")
            st.code(debug_info)
            st.code(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
        timeout = st.session_state.timeout
        resp = requests.post(api_url, headers=headers, json=data, timeout=timeout)
        if st.session_state.debug_mode:
            st.info(f"响应状态码: {resp.status_code}")
            st.code(f"响应内容: {resp.text}")
        try:
            error_detail = resp.json()
        except:
            error_detail = resp.text
        resp.raise_for_status()
        resp_json = resp.json()
        if st.session_state.debug_mode:
            st.code(f"解析后的响应: {json.dumps(resp_json, ensure_ascii=False, indent=2)}")
        if "output" in resp_json and "text" in resp_json["output"]:
            return resp_json["output"]["text"]
        elif "choices" in resp_json and len(resp_json["choices"]) > 0:
            return resp_json["choices"][0]["message"]["content"]
        else:
            return f"❌ 无法解析响应格式: {resp.text}"
    except requests.exceptions.HTTPError as e:
        return f"❌ HTTP错误 ({resp.status_code}): {str(e)}\n\n详细信息: {error_detail}\n\n{debug_info}"
    except requests.exceptions.Timeout:
        return f"❌ 错误：请求超时 ({timeout}秒)！\n请在侧边栏增加超时时间或检查网络连接。"
    except requests.exceptions.ConnectionError:
        return "❌ 错误：无法连接到 API 服务器！\n请检查网络连接和代理设置。"
    except KeyError as e:
        return f"❌ API响应格式错误: {str(e)}\n响应: {resp.text}"
    except Exception as e:
        return f"❌ 未知错误: {str(e)}\n\n{debug_info}"

# ===================== AI功能函数 - 异常检测 =====================
def ai_detect_anomalies(freq, current):
    full_data = []
    for f, i in zip(freq, current):
        full_data.append(f"f={f:.1f}Hz, I={i:.2e}A")
    full_data_str = "\n".join(full_data)
    try:
        local_has_outlier, outliers_mask, local_reasons = enhanced_outlier_check(freq, current)
        anomaly_points = [f"{freq[i]:.1f}Hz" for i in np.where(outliers_mask)[0]]
        local_analysis = f"""本地分析结果：
- 是否检测到异常：{local_has_outlier}
- 异常点：{', '.join(anomaly_points) if anomaly_points else '无'}
- 本地检测原因：{'; '.join(local_reasons) if local_reasons else '无'}
"""
    except:
        local_analysis = "本地分析不可用"
    prompt = f"""你是资深的物理实验指导专家，专门帮助学生分析RLC串联谐振实验数据。

【实验数据】
{full_data_str}

【背景知识】
RLC串联谐振电流-频率曲线特性：
1. 曲线应平滑、单峰、先升后降
2. 谐振峰两侧应对称
3. 电流值应为正数
4. 常见异常原因：
   1. 操作失误（接线松动、频率调节过快、仪器读数未稳定）
   2. 参数设置错误（恒流源/信号源参数设置不当）
   3. 数据记录错误（抄写错误、单位混淆）
   4. 仪器问题（仪器故障、量程选择不当）
   5. 环境干扰（电源波动、电磁干扰）

【{local_analysis}】

【异常检测结果】
1. 异常是否存在：是/否
2. 异常点：列出具体频率值
3. 判断依据：详细说明判断理由

【异常原因分析】
4. 可能原因分析：
   原因1：具体说明
   原因2：具体说明

【验证方法】
5. 验证方法：
   方法1：具体操作步骤
   方法2：具体操作步骤

【修改建议】
6. 修改建议：针对最可能的原因给出具体方案

【重要要求】
- 绝对不要使用任何数学公式或LaTeX符号
- 用通俗易懂的中文描述物理概念
- 内容要深入具体，不要太笼统
"""
    st.session_state.ai_analysis_context = prompt
    return call_ai(prompt, max_tokens=1800)

# ===================== AI功能函数 - 对话 =====================
def ai_chat(user_message, context=""):
    system_prompt = """你是一位专业的物理实验指导老师，专门帮助学生分析RLC串联谐振实验。
回答要求：
- 专业、耐心、有启发性
- 用通俗易懂的中文解释物理原理
- 绝对不要使用任何数学公式或LaTeX符号
- 引导学生思考而不是直接给出答案
- 如果涉及异常分析，给出具体可行的验证方法
"""
    if context:
        prompt = f"""{system_prompt}

【上下文信息】
{context}

【学生问题】
{user_message}

请针对学生的问题给出专业指导。
"""
    else:
        prompt = f"""{system_prompt}

【学生问题】
{user_message}

请回答学生的问题。
"""
    return call_ai(prompt, max_tokens=800)

# ===================== AI功能函数 - 实验总结 =====================
def ai_generate_summary(freq, current, fitting_params, student_feedback="", has_anomaly=False, anomaly_info=""):
    full_data = []
    for f, i in zip(freq, current):
        full_data.append(f"f={f:.1f}Hz, I={i:.2e}A")
    full_data_str = "\n".join(full_data)
    fitting_info = f"""
拟合结果：
- 谐振频率 f0 = {fitting_params.get('f0', 'N/A')} Hz
- 品质因数 Q = {fitting_params.get('Q', 'N/A')}
- 带宽 BW = {fitting_params.get('BW', 'N/A')} Hz
- R² = {fitting_params.get('r2', 'N/A')}
"""
    prompt = f"""你是资深物理实验教师，为RLC实验写总结。

实验数据：
{full_data_str}

{fitting_info}
"""
    if has_anomaly:
        prompt += f"""
本次实验过程：
- 初始数据存在异常
- AI检测到的异常信息：{anomaly_info}
- 学生反馈：{student_feedback if student_feedback else '学生未提供详细反馈'}
"""
        prompt += """
请按以下结构撰写总结：

【实验概述】
简要说明实验目的和完成情况

【异常原因深入剖析】
详细分析本次异常的根本原因，以及该异常是如何影响实验结果的

【实验结果分析】
分析拟合结果、谐振特性等

【实验收获与改进建议】
总结学生从本次异常处理中学到的经验，以及未来如何避免类似问题，给出具体可操作的建议
"""
    else:
        prompt += """
请按以下结构撰写总结：

【实验概述】
简要说明实验目的和完成情况

【实验结果分析】
分析拟合结果、谐振特性等

【实验收获与改进建议】
总结实验经验和未来改进方向，给出具体建议
"""
    prompt += """

【重要要求】
- 绝对不要使用任何数学公式或LaTeX符号
- 用通俗易懂的中文描述物理概念
- 内容要深入具体，不要太笼统
"""
    return call_ai(prompt, max_tokens=2200)

# ===================== AI功能函数 - 参数设计辅助 =====================
def ai_design_helper(target_f0, target_Q, current_L, current_C, current_R):
    """AI辅助参数设计：根据目标谐振特性推荐RLC参数"""
    prompt = f"""你是一位资深物理实验教学专家，擅长指导学生进行RLC串联谐振实验的参数设计。

【当前参数设置】
- 电感 L = {current_L*1000:.3f} mH
- 电容 C = {current_C*1e6:.4f} μF
- 电阻 R = {current_R:.2f} Ω

【实验目标】
- 目标谐振频率 f0 = {target_f0:.0f} Hz
- 目标品质因数 Q = {target_Q:.1f}

请分析当前参数是否能满足实验目标，如果不能，请给出具体的参数调整建议。

【参数分析】
1. 当前参数能否达到目标？：是/否/部分达到
2. 实际计算的谐振频率和Q值是多少？
3. 当前参数的主要问题是什么？

【参数调整建议】
4. 建议的调整方案1：具体修改哪几个元件的参数，修改后的数值
5. 建议的调整方案2：另一种可行的参数组合
6. 推荐理由：为什么这样调整能达到目标

【实验提示】
7. 使用这套参数实验时的注意事项
8. 可能遇到的问题及解决方案

【重要要求】
- 绝对不要使用任何数学公式或LaTeX符号
- 所有数值请用中文单位说明（例如：3000赫兹，10毫亨，0.28微法）
- 用通俗易懂的中文解释物理原理
- 建议要具体可操作，元件参数要在实验室常用范围内
- 电感范围建议：1毫亨到100毫亨
- 电容范围建议：0.01微法到10微法
- 电阻范围建议：1欧姆到100欧姆
"""
    return call_ai(prompt, max_tokens=1500)

# ===================== AI功能函数 - 单点判断 =====================
def ai_single_point_check(freq_point, current_point, theoretical_current=None, expected_from_data=None, deviation_vs_theory=None, deviation_vs_data=None, context_data="", L=None, C=None, R=None, Vin=None, filtered_count=0):
    """AI辅助单点数据判断 - 基于理论曲线 + 过滤后实测数据双重基准"""
    context = context_data if context_data else "刚开始测量，数据点有限"
    # 构建基准信息描述
    theory_info = f"根据RLC元件参数（L={L*1000:.2f}mH, C={C*1e6:.3f}μF, R={R:.2f}Ω, Vin={Vin:.1f}V）计算的理论预期电流约为 {theoretical_current:.4f} A，当前数据与理论值偏差约 {deviation_vs_theory:.1f}%" if theoretical_current else "无理论基准（需在参数设计页确认元件参数）"
    data_info = f"基于过滤掉异常点后的历史实测数据趋势，预测的预期电流约为 {expected_from_data:.4f} A，与趋势偏差约 {deviation_vs_data:.1f}%" if expected_from_data else "无足够历史数据做趋势预测"
    filter_info = f"（已从历史数据中自动排除 {filtered_count} 个明显异常点，避免被坏数据干扰判断）" if filtered_count > 0 else ""
    prompt = f"""你是一位严谨的物理实验指导老师，正在协助学生判断单点实验数据是否合理。

【当前测量数据】
- 测量频率：{freq_point:.1f} Hz
- 测量电流：{current_point:.4f} A

【双重判断基准】
1. 物理理论基准：{theory_info}
2. 实测趋势基准：{data_info} {filter_info}

【背景信息】
{context}

【你的分析目标】
请综合上述两个基准来判断这个数据点是否合理：
- 如果与理论值偏差小 + 与实测趋势一致 → 正常数据
- 如果与理论值偏差大，但与趋势一致 → 可能是元件实际参数与标称值有偏差，或信号源电压不恒定
- 如果与理论值一致，但与趋势偏差大 → 可能之前的实测数据中存在系统性问题
- 如果两个基准都偏差大 → 高概率是操作/接线/仪器读数异常

【分析结果输出结构】
【数据点判断结论】
1. 是否合理：合理/存疑/异常
2. 综合偏差评估：与理论值偏差XX%，与实测趋势偏差XX%

【可能原因分析】
3. 最可能原因：具体说明（优先考虑接线松动、频率调节过快、仪器读数未稳定、量程不当、单位混淆、元件实际值与标称值偏差等）
4. 次要可能原因：具体说明

【验证与处理建议】
5. 建议的验证步骤：学生应该做什么来验证（逐项列出可操作步骤）
6. 处理建议：保留并记录说明/建议重新测量/标记为异常点并跳过

【重要要求】
- 不要使用任何数学公式或LaTeX符号
- 建议要具体可操作，让学生知道检查什么和怎么检查
- 语气要耐心鼓励，帮助学生发现问题而不是批评
"""
    return call_ai(prompt, max_tokens=1200)

# ===================== AI功能函数 - 连线检测（基于图像） =====================
def ai_wiring_check(image_description="", manual_input=""):
    """AI辅助连线检测 - 基于图像描述或直接分析"""
    if image_description:
        context = f"【图像描述】学生对实验台/接线的描述：{image_description}\n\n"
    else:
        context = ""
    if manual_input:
        context += f"【学生观察】{manual_input}\n\n"
    prompt = f"""你是一位经验丰富的物理实验教学专家，负责检查学生的RLC串联谐振实验接线是否正确。

{context}
【RLC串联电路的正确接线要求】
1. 信号发生器的输出端连接一个由电阻R、电感L、电容C串联组成的闭合回路
2. 仪器的连接顺序应该是：信号发生器 → R → L → C → 返回信号发生器，形成串联闭合回路
3. 交流毫伏表1（或示波器通道1）应并联在信号发生器的输出两端，用于监测输入电压
4. 交流毫伏表2（或示波器通道2）应并联在电阻R两端，用于测量电阻电压（与电流成正比）
5. 所有仪器的接地端应连接在一起，形成公共参考点
6. 接线柱必须拧紧，不能有松动
7. 毫伏表必须是并联（跨接）在被测元件两端，绝不能串联
8. 信号发生器输出应设置为正弦波，且在整个测量过程中保持输出电压恒定

请判断学生的接线是否可能存在以下常见错误：
1. 电路未形成闭合回路（开路）
2. 毫伏表被错误地串联接入电路
3. 元件连接顺序错误
4. 仪器接地不正确
5. 接线柱松动
6. 信号源参数设置错误
7. 量程选择不当
8. 其他可能的操作问题

【接线检查总体结论】
1. 总体评估：基本正确/存在潜在问题/存在明显错误
2. 可信度评估：高/中/低

【发现的可能问题】
3. 问题1：问题描述 + 具体检查方法 + 纠正建议
4. 问题2：问题描述 + 具体检查方法 + 纠正建议

【操作注意事项】
5. 实验过程中需要特别注意的操作细节
6. 测量前的确认清单

【重要要求】
- 不要使用任何数学公式或LaTeX符号
- 所有建议要具体可操作，让学生知道"检查什么"和"怎么检查"
- 保持耐心和鼓励的语气
"""
    return call_ai(prompt, max_tokens=1500)

# ===================== AI功能函数 - 全流程指导 =====================
def ai_step_guidance(step_index, student_question=""):
    """AI辅助全流程实验指导"""
    if 0 <= step_index < len(EXPERIMENT_STEPS):
        step_info = EXPERIMENT_STEPS[step_index]
        step_title = step_info["title"]
        step_content = step_info["content"]
    else:
        step_title = "实验综合指导"
        step_content = "实验进行中..."
    prompt = f"""你是一位非常有经验的物理实验教学专家，正在实时指导学生进行RLC串联谐振实验。

【当前实验阶段】
{step_title}

【该步骤的参考操作要点】
{step_content}

【学生当前情况】
{student_question if student_question else '学生正在进行该步骤，需要你的实时指导和提醒。'}

请针对当前步骤，提供专业、可操作的实时指导。

【当前步骤关键要点回顾】
1. 3-5条该步骤最核心的注意事项，每条简洁明确

【实时操作建议】
2. 下一步具体应该做什么
3. 操作过程中需要检查和确认什么
4. 可能会遇到的典型问题及解决方法

【安全与质量提示】
5. 该步骤的安全注意事项
6. 保证实验数据质量的建议

【重要要求】
- 不要使用任何数学公式或LaTeX符号
- 指令要具体明确，让学生知道"做什么"和"怎么做"
- 语气要亲切鼓励，帮助学生建立信心
"""
    return call_ai(prompt, max_tokens=1500)

# ===================== 侧边栏配置 =====================
with st.sidebar:
    st.subheader("⚙️ 系统设置")
    st.session_state.debug_mode = st.checkbox("启用调试模式", value=st.session_state.debug_mode, help="显示详细的API调用信息")

    st.markdown("---")
    st.subheader("🔑 API密钥配置")
    current_key = get_api_key()
    if current_key and len(current_key) >= 10:
        st.success(f"✅ API密钥已配置 (前10位: {current_key[:10]}...)")
    else:
        st.warning("⚠️ 请配置API密钥")
    new_key = st.text_input("豆包API密钥", type="password", placeholder="ark-...", value=st.session_state.api_key if st.session_state.api_key else "", key="api_key_input")
    if new_key and new_key != st.session_state.api_key:
        st.session_state.api_key = new_key
        st.success("✅ API密钥已更新")
        st.session_state.api_status = "not_checked"
        st.rerun()

    st.markdown("---")
    st.subheader("🎯 Endpoint配置")
    st.info("💡 需要在火山引擎控制台创建推理接入点获取endpoint")
    use_preset = st.checkbox("使用预设endpoint", value=not st.session_state.use_custom_endpoint)
    st.session_state.use_custom_endpoint = not use_preset
    if use_preset:
        preset_options = [m["label"] for m in COMMON_ENDPOINTS]
        preset_names = [m["name"] for m in COMMON_ENDPOINTS]
        try:
            current_idx = preset_names.index(st.session_state.endpoint) if st.session_state.endpoint in preset_names else 0
        except:
            current_idx = 0
        selected_label = st.selectbox("选择预设endpoint", preset_options, index=current_idx)
        selected_name = [m["name"] for m in COMMON_ENDPOINTS if m["label"] == selected_label][0]
        if selected_name != st.session_state.endpoint:
            st.session_state.endpoint = selected_name
            st.session_state.api_status = "not_checked"
            st.rerun()
    else:
        custom_endpoint = st.text_input("输入您的推理接入点Endpoint", value=st.session_state.endpoint if st.session_state.endpoint else "", placeholder="例如: ep-20240101000000-abcde", key="endpoint_input")
        if custom_endpoint and custom_endpoint != st.session_state.endpoint:
            st.session_state.endpoint = custom_endpoint.strip()
            st.session_state.api_status = "not_checked"
            st.rerun()
    if st.session_state.endpoint:
        st.success(f"✅ 当前Endpoint: {st.session_state.endpoint}")
    else:
        st.warning("⚠️ 请配置Endpoint")

    st.markdown("---")
    st.subheader("✅ API连接检查")
    if st.session_state.api_status == "checking":
        st.info("🔄 正在检查API连接...")
    elif st.session_state.api_status == "success":
        st.success(st.session_state.api_status_message)
    elif st.session_state.api_status == "error":
        st.error(st.session_state.api_status_message)
    else:
        st.info("⚪ 点击下方按钮检查连接")
    if st.button("检查API连接", type="primary", key="check_api_btn"):
        st.session_state.api_status = "checking"
        st.rerun()
    if st.session_state.api_status == "checking":
        with st.spinner("正在检查连接..."):
            success, message = check_api_connection(current_key, st.session_state.endpoint, st.session_state.api_format, min(15, st.session_state.timeout))
        if success:
            st.session_state.api_status = "success"
        else:
            st.session_state.api_status = "error"
        st.session_state.api_status_message = message
        st.rerun()

    st.markdown("---")
    st.subheader("⚙️ 高级设置")
    api_format_option = st.radio("API格式", ["Chat Completions (推荐)", "Responses"], index=0 if st.session_state.api_format == "chat" else 1)
    new_format = "chat" if "Chat" in api_format_option else "responses"
    if new_format != st.session_state.api_format:
        st.session_state.api_format = new_format
        st.session_state.api_status = "not_checked"
        st.rerun()
    new_timeout = st.slider("请求超时时间(秒)", 10, 120, st.session_state.timeout)
    if new_timeout != st.session_state.timeout:
        st.session_state.timeout = new_timeout
        st.rerun()

    st.markdown("---")
    st.subheader("🔬 实验参数")
    current_unit = st.selectbox("电流单位", ["A", "mA"], key="current_unit_sidebar")
    st.session_state.current_unit = current_unit
    use_manual_f0 = st.checkbox("手动设置谐振频率 f0", value=False, key="man_f0_sidebar")
    manual_f0 = st.number_input("f0 (Hz)", 0.0, 100000.0, 3000.0, format="%.1f", disabled=not use_manual_f0, key="man_f0_val")
    use_manual_f1 = st.checkbox("手动设置低频半功率点 f1", value=False, key="man_f1_sidebar")
    manual_f1 = st.number_input("f1 (Hz)", 0.0, 100000.0, 2700.0, format="%.1f", disabled=not use_manual_f1, key="man_f1_val")
    use_manual_f2 = st.checkbox("手动设置高频半功率点 f2", value=False, key="man_f2_sidebar")
    manual_f2 = st.number_input("f2 (Hz)", 0.0, 100000.0, 3300.0, format="%.1f", disabled=not use_manual_f2, key="man_f2_val")
    st.session_state.use_manual_f0 = use_manual_f0
    st.session_state.manual_f0 = manual_f0
    st.session_state.use_manual_f1 = use_manual_f1
    st.session_state.manual_f1 = manual_f1
    st.session_state.use_manual_f2 = use_manual_f2
    st.session_state.manual_f2 = manual_f2

# ===================== 主标题 =====================
st.title("🔬 RLC串联谐振实验 · AI智能辅助系统")
st.markdown("""
<div style='background: linear-gradient(90deg, #e3f2fd, #fce4ec); padding: 16px; border-radius: 10px; margin-bottom: 16px;'>
<b>✨ 核心功能：</b>
全流程实验指导 · AI参数设计辅助 · 预设元件库 · 预测曲线对照 · 实时单点判断 · 图像连线检测 · 异常数据诊断 · 智能实验总结
</div>
""", unsafe_allow_html=True)

# ===================== 多Tab主界面 =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📘 全流程实验指导",
    "🎯 参数设计与预测",
    "📊 数据采集与分析",
    "🔍 单点实时判断",
    "📷 连线操作检测"
])

# ===================== Tab 1: 全流程实验指导 =====================
with tab1:
    st.header("📘 全流程实验实时指导")
    st.markdown("跟随AI助手一步步完成实验，每个环节都有专业指导。")

    # 步骤导航
    st.subheader("🧭 实验步骤导航")
    step_titles = [f"{i+1}. {s['title']}" for i, s in enumerate(EXPERIMENT_STEPS)]
    current_step_idx = st.session_state.guide_step_index
    selected_step = st.radio("选择当前实验阶段", step_titles, index=current_step_idx, horizontal=True)
    new_idx = step_titles.index(selected_step)
    if new_idx != current_step_idx:
        st.session_state.guide_step_index = new_idx
        st.rerun()

    current_step = EXPERIMENT_STEPS[new_idx]

    # 显示当前步骤内容
    st.markdown("---")
    st.subheader(f"📍 {current_step['title']}")
    st.markdown(current_step["content"])

    # 进度条
    progress = (new_idx + 1) / len(EXPERIMENT_STEPS)
    st.progress(progress)
    st.caption(f"完成进度：{new_idx + 1} / {len(EXPERIMENT_STEPS)}")

    # 按钮行
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ 上一步", disabled=(new_idx == 0), use_container_width=True):
            st.session_state.guide_step_index = max(0, new_idx - 1)
            st.rerun()
    with col2:
        if st.button("➡️ 下一步", disabled=(new_idx == len(EXPERIMENT_STEPS) - 1), use_container_width=True):
            st.session_state.guide_step_index = min(len(EXPERIMENT_STEPS) - 1, new_idx + 1)
            st.rerun()
    with col3:
        if st.button("💬 AI实时指导", type="primary", use_container_width=True):
            with st.spinner("AI正在思考最适合当前步骤的指导建议..."):
                guidance = ai_step_guidance(new_idx)
                st.session_state[f"step_{new_idx}_guidance"] = guidance

    # 显示AI指导结果
    if f"step_{new_idx}_guidance" in st.session_state:
        st.markdown("---")
        st.subheader("🤖 AI实时指导建议")
        st.markdown(st.session_state[f"step_{new_idx}_guidance"])

    # 自由对话区域
    st.markdown("---")
    st.subheader("💬 随时提问 AI 助手")
    st.info("在实验的任何环节都可以随时向AI提问，获取专业指导。可以问原理、操作、误差分析等任何问题。")

    # 快捷问题按钮
    q_cols = st.columns(3)
    quick_questions = [
        f"当前步骤（{current_step['title']}）有什么特别需要注意的？",
        "如果我发现数据异常，应该怎么排查？",
        "请帮我解释一下RLC谐振的物理原理"
    ]
    for i, (col, q) in enumerate(zip(q_cols, quick_questions)):
        with col:
            if st.button(f"💡 {q[:20]}...", key=f"qq_{new_idx}_{i}", help=q):
                with st.spinner("AI正在回答..."):
                    response = ai_step_guidance(new_idx, student_question=q)
                    st.session_state.chat_messages.append({"role": "user", "content": q})
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})

    # 自由输入
    user_q = st.text_area("输入你的问题（或描述当前遇到的困难）：", height=80, key=f"guide_chat_{new_idx}", placeholder="例如：我在测量时发现电流突然变小，可能是什么原因？")
    if st.button("📤 发送给 AI", key=f"send_guide_{new_idx}", type="secondary"):
        if user_q.strip():
            with st.spinner("AI正在回答..."):
                response = ai_step_guidance(new_idx, student_question=user_q)
                st.session_state.chat_messages.append({"role": "user", "content": user_q})
                st.session_state.chat_messages.append({"role": "assistant", "content": response})

    # 历史对话
    if st.session_state.chat_messages:
        with st.expander("📜 查看历史对话记录", expanded=False):
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

# ===================== Tab 2: 参数设计与预测 =====================
with tab2:
    st.header("🎯 AI辅助参数设计 & 预测对照")
    st.markdown("实验前先设计参数，预测谐振特性；实验后与实测数据对比分析。")

    # 预设元件库
    st.subheader("📦 预设RLC元件库")
    preset_names = [p["name"] for p in PRESET_RLC]
    selected_idx = st.selectbox("选择一套预设元件参数", range(len(preset_names)),
                                format_func=lambda i: preset_names[i],
                                index=st.session_state.selected_preset_idx)
    if selected_idx != st.session_state.selected_preset_idx:
        st.session_state.selected_preset_idx = selected_idx
        preset = PRESET_RLC[selected_idx]
        st.session_state.design_L = preset["L"]
        st.session_state.design_C = preset["C"]
        st.session_state.design_R = preset["R"]
        st.rerun()

    preset = PRESET_RLC[selected_idx]
    st.info(f"📋 {preset['note']} | 预估 f0 ≈ {preset['f0']:.0f} Hz, Q ≈ {preset['Q']:.1f}")

    # 参数调节区
    st.markdown("---")
    st.subheader("🔧 元件参数设置（可微调）")
    col_L, col_C, col_R = st.columns(3)
    with col_L:
        design_L = st.number_input("电感 L (mH)", 0.1, 1000.0, st.session_state.design_L * 1000, 0.1, format="%.3f") / 1000
    with col_C:
        design_C = st.number_input("电容 C (μF)", 0.001, 100.0, st.session_state.design_C * 1e6, 0.001, format="%.4f") / 1e6
    with col_R:
        design_R = st.number_input("电阻 R (Ω)", 0.1, 500.0, float(st.session_state.design_R), 0.1, format="%.2f")
    col_V1, col_V2 = st.columns([1, 3])
    with col_V1:
        design_Vin = st.number_input("输入电压 Vin (V)", 0.1, 20.0, float(st.session_state.design_Vin), 0.1, format="%.1f")

    # 更新会话状态
    st.session_state.design_L = design_L
    st.session_state.design_C = design_C
    st.session_state.design_R = design_R
    st.session_state.design_Vin = design_Vin

    # 计算理论曲线
    f_min = 100.0
    f_max = 100000.0
    # 自动计算合理的频率范围
    theoretical_f0 = 1 / (2 * np.pi * np.sqrt(design_L * design_C))
    f_min_calc = max(f_min, theoretical_f0 * 0.2)
    f_max_calc = min(f_max, theoretical_f0 * 5.0)

    col_fmin, col_fmax = st.columns(2)
    with col_fmin:
        f_min_user = st.number_input("显示频率下限 (Hz)", 10.0, 100000.0, f_min_calc, 10.0, format="%.0f")
    with col_fmax:
        f_max_user = st.number_input("显示频率上限 (Hz)", 10.0, 500000.0, f_max_calc, 10.0, format="%.0f")

    t_freq, t_I, t_f0, t_Q, t_BW, t_I0 = compute_theoretical_curve(design_L, design_C, design_R, design_Vin, f_min_user, f_max_user)

    # 显示理论参数
    st.markdown("---")
    st.subheader("📐 理论计算结果")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("理论谐振频率 f0", f"{t_f0:.1f} Hz")
    with c2:
        st.metric("理论品质因数 Q", f"{t_Q:.2f}")
    with c3:
        st.metric("理论带宽 BW", f"{t_BW:.1f} Hz")
    with c4:
        st.metric("谐振时电流 I_max", f"{t_I0:.4f} A")

    # 绘制理论曲线
    st.subheader("📈 预测谐振曲线")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_freq, t_I, "b-", linewidth=2, label="理论预测曲线")
    # 标注谐振点
    ax.axvline(x=t_f0, color="r", linestyle="--", alpha=0.7, label=f"谐振频率 f0={t_f0:.0f} Hz")
    # 标注半功率点
    I_half = t_I0 / np.sqrt(2)
    try:
        f1_ideal = t_f0 - t_BW / 2
        f2_ideal = t_f0 + t_BW / 2
        ax.axhline(y=I_half, color="m", linestyle=":", alpha=0.7, label=f"半功率点 I={I_half:.4f} A")
        ax.axvline(x=f1_ideal, color="m", linestyle=":", alpha=0.5)
        ax.axvline(x=f2_ideal, color="m", linestyle=":", alpha=0.5)
    except:
        pass
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("电流 (A)")
    ax.set_title("RLC串联谐振理论预测曲线")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # AI参数设计建议
    st.markdown("---")
    st.subheader("🤖 AI参数设计建议")
    st.info("根据您的实验目标（希望的谐振频率、品质因数等），让AI为您推荐合适的元件参数组合。")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        target_f0 = st.number_input("目标谐振频率 f0 (Hz)", 50.0, 100000.0, t_f0, 10.0)
    with col_t2:
        target_Q = st.number_input("目标品质因数 Q", 1.0, 500.0, t_Q, 1.0)

    if st.button("🧠 获取AI参数设计建议", type="primary"):
        with st.spinner("AI正在为您分析最佳参数组合..."):
            design_advice = ai_design_helper(target_f0, target_Q, design_L, design_C, design_R)
            st.session_state.design_advice = design_advice

    if "design_advice" in st.session_state:
        st.markdown(st.session_state.design_advice)

    # 与实测数据对比
    if st.session_state.f_data is not None and len(st.session_state.f_data) > 2:
        st.markdown("---")
        st.subheader("📊 预测曲线 vs 实测数据对比")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        # 理论曲线
        unit = st.session_state.get('current_unit', 'A')
        scale = 1000 if unit == 'mA' else 1
        ax2.plot(t_freq, t_I * scale, "b-", linewidth=2, alpha=0.7, label="理论预测曲线")
        # 实测数据
        ax2.scatter(st.session_state.f_data, st.session_state.I_data * scale, c="red", s=30, marker="o", label="实测数据", zorder=5)
        ax2.set_xlabel("频率 (Hz)")
        ax2.set_ylabel(f"电流 ({unit})")
        ax2.set_title("理论预测 vs 实测数据对比")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)
        st.success("✅ 可以直观看到实测数据与理论预测的吻合程度，帮助分析实验误差来源。")
    else:
        st.info("💡 提示：在「数据采集与分析」标签页上传或生成实验数据后，这里会自动显示预测与实测对比图。")

# ===================== Tab 3: 数据采集与分析 =====================
with tab3:
    st.header("📊 数据采集与分析（AI异常检测+拟合分析）")

    # ========== 数据输入方式选择 ==========
    st.subheader("📥 数据输入方式")
    input_method = st.radio("选择数据输入方式", ["上传Excel文件", "批量粘贴数据", "手动逐行输入", "AI生成示例数据"], horizontal=True)

    # 方式1: 上传Excel
    if input_method == "上传Excel文件":
        uploaded_file = st.file_uploader("上传Excel文件（第一列频率，第二列电流，按频率从小到大排列）", type=["xlsx", "xls"], key="data_upload_tab3")
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                f_data_new = df.iloc[:, 0].values
                I_data_new = df.iloc[:, 1].values
                if st.session_state.get('current_unit', 'A') == 'mA':
                    I_data_new = I_data_new / 1000.0
                st.session_state.f_data = f_data_new
                st.session_state.I_data = I_data_new
                st.session_state.ai_analysis_result = None
                st.session_state.final_summary = None
                st.success(f"✅ 数据上传成功！共 {len(f_data_new)} 个数据点")
            except Exception as e:
                st.error(f"数据读取失败：{str(e)}")

    # 方式2: 批量粘贴
    elif input_method == "批量粘贴数据":
        st.info("💡 在下方粘贴数据，每行一个数据点，格式：`频率,电流`（支持空格/逗号/Tab分隔）。也可以直接从Excel复制粘贴。")
        paste_data = st.text_area("批量粘贴数据", height=200, placeholder="例如：\n1500, 0.015\n1800, 0.025\n2000, 0.040\n...\n3000, 0.200\n...\n4500, 0.030")
        if st.button("📋 解析粘贴的数据"):
            try:
                lines = paste_data.strip().split("\n")
                freq_list = []
                I_list = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # 支持多种分隔符
                    parts = line.replace(",", " ").replace("\t", " ").split()
                    if len(parts) >= 2:
                        freq_list.append(float(parts[0]))
                        I_list.append(float(parts[1]))
                if len(freq_list) >= 3:
                    f_data_new = np.array(freq_list)
                    I_data_new = np.array(I_list)
                    if st.session_state.get('current_unit', 'A') == 'mA':
                        I_data_new = I_data_new / 1000.0
                    # 按频率排序
                    sorted_idx = np.argsort(f_data_new)
                    st.session_state.f_data = f_data_new[sorted_idx]
                    st.session_state.I_data = I_data_new[sorted_idx]
                    st.session_state.ai_analysis_result = None
                    st.session_state.final_summary = None
                    st.success(f"✅ 成功解析 {len(freq_list)} 个数据点！")
                else:
                    st.error("❌ 有效数据点不足3个，请检查数据格式。")
            except Exception as e:
                st.error(f"❌ 数据解析失败：{str(e)}\n请检查每行是否都是两个数字。")

    # 方式3: 手动逐行输入（自动化输入）
    elif input_method == "手动逐行输入":
        st.info("💡 适合实验过程中边测量边输入，支持随时追加数据。")
        col_f, col_i, col_btn = st.columns([2, 2, 1])
        with col_f:
            new_f = st.number_input("频率 (Hz)", 1.0, 1000000.0, 3000.0, 1.0, format="%.1f", key="auto_f")
        with col_i:
            new_i = st.number_input("电流", 0.0, 10000.0, 0.1, 0.001, format="%.4f", key="auto_i", help="单位与侧边栏设置一致")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ 添加数据点", type="primary", use_container_width=True):
                new_i_si = new_i / 1000.0 if st.session_state.get('current_unit', 'A') == 'mA' else new_i
                if st.session_state.f_data is None:
                    st.session_state.f_data = np.array([new_f])
                    st.session_state.I_data = np.array([new_i_si])
                else:
                    st.session_state.f_data = np.append(st.session_state.f_data, new_f)
                    st.session_state.I_data = np.append(st.session_state.I_data, new_i_si)
                # 重新排序
                sorted_idx = np.argsort(st.session_state.f_data)
                st.session_state.f_data = st.session_state.f_data[sorted_idx]
                st.session_state.I_data = st.session_state.I_data[sorted_idx]
                st.session_state.ai_analysis_result = None
                st.rerun()

    # 方式4: AI生成示例
    elif input_method == "AI生成示例数据":
        col1, col2, col3 = st.columns(3)
        with col1:
            sample_f0 = st.number_input("示例谐振频率 (Hz)", 100.0, 10000.0, 3000.0, key="sample_f0_tab3")
        with col2:
            sample_Q = st.number_input("示例品质因数 Q", 1.0, 100.0, 30.0, key="sample_Q_tab3")
        with col3:
            sample_noise = st.slider("添加噪声水平", 0.0, 0.2, 0.02, 0.01, key="sample_noise_tab3")
        add_anomaly = st.checkbox("添加模拟异常点（模拟实验中的操作失误）", value=False)
        if st.button("🎲 生成示例数据", type="primary"):
            num_points = 50
            freq = np.linspace(sample_f0 * 0.5, sample_f0 * 1.5, num_points)
            omega = 2 * np.pi * freq
            omega0 = 2 * np.pi * sample_f0
            I = 1 / np.sqrt(1 + (sample_Q * (omega/omega0 - omega0/omega0))**2)
            I = I * (1 + np.random.normal(0, sample_noise, num_points))
            if add_anomaly:
                anomaly_idx = np.argmin(np.abs(freq - sample_f0)) + 5
                if 0 < anomaly_idx < len(I):
                    I[anomaly_idx] = I[anomaly_idx] * 0.3
            st.session_state.f_data = freq
            st.session_state.I_data = I
            st.session_state.ai_analysis_result = None
            st.session_state.final_summary = None
            st.success("✅ 示例数据已生成！")
            st.rerun()

    # 数据管理
    if st.session_state.f_data is not None:
        with st.expander(f"📊 数据管理（当前 {len(st.session_state.f_data)} 个数据点）", expanded=False):
            df_show = pd.DataFrame({"频率 (Hz)": st.session_state.f_data, "电流 (A)": st.session_state.I_data})
            st.dataframe(df_show, use_container_width=True, height=200)
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                if st.button("🗑️ 清除所有数据"):
                    for key in ['f_data', 'I_data', 'ai_analysis_result', 'final_summary', 'fitting_results']:
                        if key in st.session_state:
                            st.session_state[key] = None
                    st.session_state.experiment_step = 1
                    st.rerun()
            with col_d2:
                if st.button("📥 导出当前数据为CSV"):
                    csv = df_show.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(label="下载CSV文件", data=csv, file_name=f"rlc_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

        # ========== 显示当前数据 ==========
        st.markdown("---")
        st.subheader("📈 当前数据曲线预览")
        f_data = st.session_state.f_data
        I_data = st.session_state.I_data
        unit = st.session_state.get('current_unit', 'A')
        scale = 1000 if unit == 'mA' else 1
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(f_data, I_data * scale, c="blue", s=40, label="实验数据")
        ax.plot(f_data, I_data * scale, "b-", alpha=0.3)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel(f"电流 ({unit})")
        ax.set_title("实验数据曲线")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 本地快速检测
        try:
            local_has_outlier, outliers_mask, local_reasons = enhanced_outlier_check(f_data, I_data)
            if local_has_outlier:
                st.warning(f"⚠️ 本地快速检测发现潜在异常：{', '.join(local_reasons)}")
            else:
                st.success("✅ 本地检测数据趋势正常")
        except:
            pass

        # ========== AI完整异常检测 ==========
        st.markdown("---")
        st.subheader("🤖 AI完整异常检测")
        if st.button("🔍 让AI深度分析数据", type="primary"):
            with st.spinner("AI正在分析数据，检测异常..."):
                result = ai_detect_anomalies(f_data, I_data)
                st.session_state.ai_analysis_result = result

        if st.session_state.ai_analysis_result:
            st.markdown(st.session_state.ai_analysis_result)

            # AI对话
            with st.expander("💬 就异常检测结果与AI对话", expanded=False):
                user_msg = st.text_input("向AI提问（例如：如何验证接线是否松动？）", key="chat_tab3_input")
                if st.button("发送", key="chat_tab3_btn"):
                    if user_msg.strip():
                        with st.spinner("AI思考中..."):
                            response = ai_chat(user_msg, st.session_state.ai_analysis_context)
                            st.markdown(response)

            # 进入异常处理或拟合
            has_anomaly_flag = "异常是否存在：是" in st.session_state.ai_analysis_result or "存在异常：是" in st.session_state.ai_analysis_result
            if has_anomaly_flag:
                st.warning("⚠️ AI检测到数据存在异常，建议先排查问题再进行拟合分析。")
                st.info("💡 您可以：1）重新测量疑似异常的数据点；2）在「单点实时判断」标签页单独分析某一点；3）在「连线操作检测」标签页检查接线是否有误。")
            else:
                st.success("✅ AI判断数据正常，可以进行拟合分析。")

        # ========== 拟合分析 ==========
        st.markdown("---")
        st.subheader("📈 参数拟合分析")
        if st.button("⚙️ 开始拟合分析", type="primary") or st.session_state.fitting_results is not None:
            try:
                use_man_f0 = st.session_state.get('use_manual_f0', False)
                man_f0 = st.session_state.get('manual_f0', 3000.0)
                use_man_f1 = st.session_state.get('use_manual_f1', False)
                man_f1 = st.session_state.get('manual_f1', 2700.0)
                use_man_f2 = st.session_state.get('use_manual_f2', False)
                man_f2 = st.session_state.get('manual_f2', 3300.0)

                L_fixed = st.session_state.design_L
                if use_man_f0:
                    f0_fit = man_f0
                    closest = np.argmin(np.abs(f_data - f0_fit))
                    I0 = I_data[closest]
                else:
                    max_i_idx = np.argmax(I_data)
                    f0_fit = f_data[max_i_idx]
                    I0 = I_data[max_i_idx]

                f1_man = man_f1 if use_man_f1 else None
                f2_man = man_f2 if use_man_f2 else None
                f1, f2, I_half = calculate_half_power_points(f_data, I_data, f0_fit, I0, f1_man, f2_man)
                BW = f2 - f1
                Q_fit = f0_fit / BW if BW != 0 else 0
                C_fit = 1 / (4 * np.pi**2 * f0_fit**2 * L_fixed)
                R_fit = (2 * np.pi * f0_fit * L_fixed) / Q_fit if Q_fit != 0 else 0

                # 绘制理想拟合曲线
                f_fit = np.linspace(f_data.min(), f_data.max(), 1000)
                omega = 2 * np.pi * f_fit
                XL = omega * L_fixed
                XC = 1 / (omega * C_fit)
                Z = np.sqrt(R_fit**2 + (XL - XC)**2)
                I_fit = (I0 * R_fit) / Z
                I_fit_on_data = np.interp(f_data, f_fit, I_fit)
                ss_res = np.sum((I_data - I_fit_on_data)**2)
                ss_tot = np.sum((I_data - np.mean(I_data))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

                st.session_state.fitting_results = {'f0': f0_fit, 'Q': Q_fit, 'BW': BW, 'r2': r2, 'C': C_fit, 'R': R_fit, 'I0': I0, 'f1': f1, 'f2': f2}

                # 显示参数
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("谐振频率 f0", f"{f0_fit:.1f} Hz")
                with c2: st.metric("品质因数 Q", f"{Q_fit:.2f}")
                with c3: st.metric("带宽 BW", f"{BW:.1f} Hz")
                with c4: st.metric("拟合优度 R²", f"{r2:.4f}")

                c5, c6, c7 = st.columns(3)
                with c5: st.metric("实际电容 C", f"{C_fit*1e6:.4f} μF")
                with c6: st.metric("等效电阻 R", f"{R_fit:.2f} Ω")
                with c7: st.metric("固定电感 L", f"{L_fixed*1000:.3f} mH")

                # 绘制拟合图
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.scatter(f_data, I_data * scale, c="blue", s=50, label="实验数据", zorder=5)
                ax.plot(f_fit, I_fit * scale, "g-", linewidth=2, label="拟合曲线")
                ax.plot([f0_fit], [I0 * scale], "ro", markersize=12, label=f"谐振点 ({f0_fit:.0f} Hz)")
                ax.plot([f1, f2], [I_half * scale, I_half * scale], "ms", markersize=10, label=f"半功率点 ({f1:.0f}, {f2:.0f} Hz)")
                ax.set_xlabel("频率 (Hz)")
                ax.set_ylabel(f"电流 ({unit})")
                ax.set_title("RLC串联谐振拟合分析")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)

                st.success("✅ 拟合完成！")

                # 生成实验总结
                if st.button("📝 生成AI实验总结报告", type="primary"):
                    with st.spinner("AI正在撰写实验总结..."):
                        has_anomaly = st.session_state.modification_applied or ("异常是否存在：是" in (st.session_state.ai_analysis_result or ""))
                        summary = ai_generate_summary(f_data, I_data, st.session_state.fitting_results, st.session_state.student_feedback, has_anomaly=has_anomaly, anomaly_info=st.session_state.ai_analysis_result if has_anomaly else "")
                        st.session_state.final_summary = summary

                if st.session_state.final_summary:
                    st.markdown("---")
                    st.subheader("📑 AI实验总结报告")
                    st.markdown(st.session_state.final_summary)

            except Exception as e:
                st.error(f"拟合过程出错：{str(e)}")

# ===================== Tab 4: 单点实时判断 =====================
with tab4:
    st.header("🔍 单点数据实时判断（双基准智能校验）")
    st.markdown("**核心改进**：不再仅依赖可能有问题的实测数据做判断，而是同时使用「RLC理论谐振曲线」和「过滤异常点后的实测趋势」作为双重判断基准，即使已有数据中存在异常点，也不会被误导。")

    # 元件参数确认区
    st.subheader("⚙️ 确认元件参数（用于计算理论基准）")
    col_L, col_C, col_R, col_V = st.columns(4)
    with col_L:
        chk_L = st.number_input("电感 L (mH)", 0.01, 10000.0, st.session_state.design_L * 1000, 0.1, format="%.3f", key="chk_L") / 1000
    with col_C:
        chk_C = st.number_input("电容 C (μF)", 0.001, 1000.0, st.session_state.design_C * 1e6, 0.001, format="%.4f", key="chk_C") / 1e6
    with col_R:
        chk_R = st.number_input("电阻 R (Ω)", 0.01, 5000.0, float(st.session_state.design_R), 0.01, format="%.2f", key="chk_R")
    with col_V:
        chk_Vin = st.number_input("信号源电压 Vin (V)", 0.1, 50.0, float(st.session_state.design_Vin), 0.1, format="%.1f", key="chk_Vin")

    # 计算理论谐振参数
    theoretical_f0 = 1 / (2 * np.pi * np.sqrt(chk_L * chk_C))
    theoretical_Q = (1 / chk_R) * np.sqrt(chk_L / chk_C)
    st.info(f"📊 基于上述元件参数计算：理论谐振频率 ≈ {theoretical_f0:.1f} Hz，理论品质因数 Q ≈ {theoretical_Q:.2f}")

    # 计算理论曲线
    f_min_theory = max(100.0, theoretical_f0 * 0.2)
    f_max_theory = min(500000.0, theoretical_f0 * 5.0)
    theory_freq, theory_I, _, _, _, _ = compute_theoretical_curve(chk_L, chk_C, chk_R, chk_Vin, f_min_theory, f_max_theory, num_points=200)

    # 数据点输入
    st.markdown("---")
    st.subheader("✏️ 输入待检测的数据点")
    col_f, col_i, col_btn = st.columns([2, 2, 1])
    with col_f:
        check_freq = st.number_input("频率 (Hz)", 1.0, 1000000.0, theoretical_f0, 1.0, format="%.1f", key="chk_freq")
    with col_i:
        check_current = st.number_input("电流", 0.0, 10000.0, 0.1, 0.001, format="%.4f", key="chk_current", help="如果选择mA为单位，数值要对应缩小")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        check_btn = st.button("🔍 AI检查此数据点", type="primary", use_container_width=True)

    # 本地双重基准检测
    check_current_si = check_current / 1000.0 if st.session_state.get('current_unit', 'A') == 'mA' else check_current
    expected_I_theory = compute_theoretical_current(check_freq, chk_L, chk_C, chk_R, chk_Vin)
    dev_vs_theory = abs(check_current_si - expected_I_theory) / expected_I_theory * 100 if expected_I_theory > 0 else 0
    has_hist_data = st.session_state.f_data is not None and len(st.session_state.f_data) >= 3

    if has_hist_data:
        clean_freq, clean_current, outlier_mask = filter_outlier_points(st.session_state.f_data, st.session_state.I_data)
        if len(clean_freq) >= 2:
            sorted_idx_clean = np.argsort(clean_freq)
            clean_freq_sorted = clean_freq[sorted_idx_clean]
            clean_I_sorted = clean_current[sorted_idx_clean]
            if check_freq < clean_freq_sorted[0] or check_freq > clean_freq_sorted[-1]:
                nearest_idx = 0 if check_freq < clean_freq_sorted[0] else len(clean_freq_sorted) - 1
                expected_from_clean = clean_I_sorted[nearest_idx]
            else:
                expected_from_clean = np.interp(check_freq, clean_freq_sorted, clean_I_sorted)
            dev_vs_clean = abs(check_current_si - expected_from_clean) / expected_from_clean * 100 if expected_from_clean > 0 else 0
        else:
            expected_from_clean = None
            dev_vs_clean = None
    else:
        clean_freq, clean_current = None, None
        outlier_mask = None
        expected_from_clean = None
        dev_vs_clean = None

    # 结果指标卡
    st.markdown("---")
    st.subheader("📋 本地双重基准预检测")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        st.metric("理论预期电流", f"{expected_I_theory:.4f} A", help="基于RLC参数的物理理论计算值，最可靠的基准")
    with col_b2:
        if dev_vs_theory < 20:
            status1 = "✅ 与理论一致"
        elif dev_vs_theory < 50:
            status1 = "⚠️ 偏离理论"
        else:
            status1 = "❌ 严重偏离理论"
        st.metric("与理论偏差", f"{dev_vs_theory:.1f}%", help=status1)
        st.caption(status1)
    with col_b3:
        if expected_from_clean is not None:
            st.metric("实测趋势预期", f"{expected_from_clean:.4f} A", help="基于过滤掉异常点后的历史数据插值预测")
        else:
            st.metric("实测趋势预期", "暂无", help="暂无足够历史数据")
    with col_b4:
        if dev_vs_clean is not None:
            if dev_vs_clean < 20:
                status2 = "✅ 与趋势一致"
            elif dev_vs_clean < 50:
                status2 = "⚠️ 偏离趋势"
            else:
                status2 = "❌ 严重偏离趋势"
            st.metric("与趋势偏差", f"{dev_vs_clean:.1f}%", help=status2)
            st.caption(status2)
        else:
            st.metric("与趋势偏差", "暂无")

    if has_hist_data and outlier_mask is not None and np.sum(outlier_mask) > 0:
        st.info(f"🛡️ 已自动从历史数据中排除 {int(np.sum(outlier_mask))} 个异常点，避免它们干扰对当前点的判断")

    # 综合可视化：理论曲线 + 干净实测点 + 被过滤异常点 + 当前待检点
    st.markdown("---")
    st.subheader("📈 可视化对比（理论曲线 vs 实测数据 vs 当前数据点）")
    unit = st.session_state.get('current_unit', 'A')
    scale = 1000 if unit == 'mA' else 1
    fig, ax = plt.subplots(figsize=(12, 5))
    # 理论曲线
    ax.plot(theory_freq, theory_I * scale, "b-", linewidth=2, alpha=0.6, label="理论谐振曲线 (物理基准)")
    # 干净的历史实测数据
    if has_hist_data and clean_freq is not None and len(clean_freq) >= 2:
        ax.scatter(clean_freq, clean_current * scale, c="green", s=50, marker="o", alpha=0.8, label=f"正常历史数据 ({len(clean_freq)}点)", zorder=3)
    # 被过滤的异常点
    if has_hist_data and outlier_mask is not None and np.sum(outlier_mask) > 0:
        outlier_freqs = st.session_state.f_data[outlier_mask]
        outlier_values = st.session_state.I_data[outlier_mask]
        ax.scatter(outlier_freqs, outlier_values * scale, c="orange", s=80, marker="x", linewidths=2, label=f"被排除的异常点 ({int(np.sum(outlier_mask))}个)", zorder=4)
    # 原始历史数据（浅色）
    if has_hist_data:
        ax.scatter(st.session_state.f_data, st.session_state.I_data * scale, c="gray", s=20, marker=".", alpha=0.3, label="原始历史数据（含异常）", zorder=2)
    # 当前待检测点
    ax.axvline(x=check_freq, color="purple", linestyle=":", linewidth=1.5, alpha=0.7, zorder=1)
    ax.scatter([check_freq], [check_current_si * scale], c="red", s=200, marker="*", linewidths=2, edgecolors="black", label="当前待检测数据点", zorder=10)
    ax.annotate(f"({check_freq:.0f}Hz, {check_current_si*scale:.4f})", (check_freq, check_current_si * scale), textcoords="offset points", xytext=(15, 15), fontsize=9, fontweight="bold", color="red")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel(f"电流 ({unit})")
    ax.set_title("理论曲线 vs 历史数据 vs 当前待检测点（红色星号）")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    st.pyplot(fig)
    st.info("💡 看图方法：如果红色星号接近蓝色理论曲线或绿色正常数据点 → 正常；如果与两者都偏离很大 → 需要检查")

    # AI检查
    if check_btn:
        check_current_si_ai = check_current_si
        with st.spinner("AI正在基于双重基准分析这个数据点..."):
            if has_hist_data:
                data_summary = f"已测量 {len(st.session_state.f_data)} 个数据点，其中 {int(np.sum(outlier_mask)) if outlier_mask is not None else 0} 个被自动识别为异常并排除，使用 {len(clean_freq)} 个正常点做趋势参考，频率范围 {st.session_state.f_data.min():.0f}-{st.session_state.f_data.max():.0f} Hz"
            else:
                data_summary = "刚开始测量，历史数据点不足，主要依赖理论曲线作为判断基准"
            result = ai_single_point_check(
                check_freq, check_current_si_ai,
                theoretical_current=expected_I_theory,
                expected_from_data=expected_from_clean,
                deviation_vs_theory=dev_vs_theory,
                deviation_vs_data=dev_vs_clean,
                context_data=data_summary,
                L=chk_L, C=chk_C, R=chk_R, Vin=chk_Vin,
                filtered_count=int(np.sum(outlier_mask)) if outlier_mask is not None else 0
            )
            st.session_state.single_point_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "freq": check_freq,
                "current": check_current_si_ai,
                "result": result
            })
        st.markdown("---")
        st.subheader("🤖 AI综合分析结果")
        st.markdown(result)

        # 询问是否添加到数据
        col_add1, col_add2 = st.columns([1, 1])
        with col_add1:
            if st.button("✅ 没问题，添加到实验数据", use_container_width=True):
                if st.session_state.f_data is None:
                    st.session_state.f_data = np.array([check_freq])
                    st.session_state.I_data = np.array([check_current_si_ai])
                else:
                    st.session_state.f_data = np.append(st.session_state.f_data, check_freq)
                    st.session_state.I_data = np.append(st.session_state.I_data, check_current_si_ai)
                sorted_idx = np.argsort(st.session_state.f_data)
                st.session_state.f_data = st.session_state.f_data[sorted_idx]
                st.session_state.I_data = st.session_state.I_data[sorted_idx]
                st.success(f"✅ 已添加数据点：f={check_freq:.1f}Hz, I={check_current_si_ai:.4f}A")
        with col_add2:
            if st.button("🔄 重新测量，不添加", use_container_width=True):
                st.info("已标记跳过此数据点。建议按照AI的建议检查后重新测量。")

    # 历史记录
    if st.session_state.single_point_history:
        with st.expander(f"📜 单点判断历史记录（共 {len(st.session_state.single_point_history)} 次）", expanded=False):
            for idx, record in enumerate(reversed(st.session_state.single_point_history)):
                st.markdown(f"**[{record['time']}]** f={record['freq']:.1f}Hz, I={record['current']:.4f}A")
                st.markdown(record['result'])
                st.markdown("---")

# ===================== Tab 5: 连线操作检测 =====================
with tab5:
    st.header("📷 连线与操作检测")
    st.markdown("上传实验台照片或描述接线情况，让AI检查是否存在接线错误、操作不当等问题。")

    # 功能选择
    check_mode = st.radio("选择检查方式", ["📝 文字描述接线情况", "📷 上传实验台照片（需人工描述关键特征）"], horizontal=True)

    if check_mode == "📝 文字描述接线情况":
        st.subheader("📝 描述你的接线与仪器设置")
        st.info("💡 请仔细描述你的接线方式，例如：元件是如何连接的、毫伏表接在哪里、信号源设置了什么参数等。描述越详细，AI判断越准确。")

        # 快捷选项
        st.markdown("#### 快捷情况描述")
        quick_cases = [
            "我刚开始接线，还不确定怎么连，请告诉我正确的接线方法。",
            "我的毫伏表指针几乎不动，可能是什么问题？",
            "我测得的谐振频率和理论计算的差很多，可能是哪里错了？",
            "测量数据很乱、跳动很大，可能是什么原因？",
            "接线柱看起来都拧紧了，但数据还是有问题。"
        ]
        for idx, case in enumerate(quick_cases):
            if st.button(f"💡 {case}", key=f"quick_case_{idx}", use_container_width=True):
                with st.spinner("AI正在分析..."):
                    result = ai_wiring_check(manual_input=case)
                    st.session_state.wiring_check_result = result
                st.markdown("---")
                st.subheader("🤖 AI检查结果")
                st.markdown(result)

        st.markdown("#### 详细描述")
        wiring_desc = st.text_area("详细描述你的接线情况、仪器设置、以及观察到的现象：", height=150, placeholder="例如：\n我把信号发生器输出接了电阻R，然后R的另一端接电感L，L的另一端接电容C，C的另一端接回信号发生器形成回路。\n毫伏表1接在信号发生器两端，毫伏表2并联在电阻R两端测量电流。\n信号发生器设置为正弦波，电压1V，频率可以调节。\n我发现测量的电流值比预期小很多...")

        if st.button("🔍 AI检查接线与操作", type="primary"):
            if wiring_desc.strip():
                with st.spinner("AI正在分析你的接线描述..."):
                    result = ai_wiring_check(manual_input=wiring_desc)
                    st.session_state.wiring_check_result = result
                st.markdown("---")
                st.subheader("🤖 AI检查结果")
                st.markdown(result)
            else:
                st.warning("⚠️ 请先描述你的接线情况。")

    else:
        st.subheader("📷 图像辅助接线检查")
        st.info("💡 上传实验台照片，然后在下方详细描述照片中可观察到的关键特征（元件位置、接线走向、仪器读数等），AI会基于这些信息进行分析。")

        # 上传图片
        uploaded_image = st.file_uploader("上传实验台照片（可选）", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_image is not None:
            st.image(uploaded_image, caption="上传的实验台照片", use_column_width=True)
            st.success("✅ 照片上传成功！请在下方描述照片中的关键特征。")

        # 让学生描述图片
        st.markdown("#### 请描述图片中的关键特征")
        st.markdown("""
        请仔细观察照片，尽量描述以下信息：
        - 电阻、电感、电容是如何串联的？
        - 毫伏表/示波器接在哪里？
        - 接线柱是否都拧紧了？
        - 仪器的设置参数（电压、频率、量程）是多少？
        - 有没有发现松动、脱焊或接触不良？
        """)

        image_desc = st.text_area("照片特征描述：", height=180, placeholder="从照片上看：\n1. 元件是按R-L-C顺序串联的...\n2. 毫伏表1接在信号源两端...\n3. 毫伏表2并联在R两端...\n4. 接线柱看起来都拧紧了...")

        if st.button("🔍 AI基于描述检查接线", type="primary"):
            if image_desc.strip():
                with st.spinner("AI正在分析..."):
                    result = ai_wiring_check(image_description=image_desc)
                    st.session_state.wiring_check_result = result
                st.markdown("---")
                st.subheader("🤖 AI检查结果")
                st.markdown(result)
            else:
                st.warning("⚠️ 请先描述图片特征。")

    # 通用：操作检查清单
    st.markdown("---")
    st.subheader("✅ 接线与操作检查清单")
    st.markdown("实验前/测量中请逐一确认以下事项：")
    checklist_items = [
        "🔌 电路是否形成闭合回路（没有开路）？",
        "📏 电阻R、电感L、电容C是否按正确顺序串联？",
        "⚡ 毫伏表是否并联在被测元件两端（而非串联）？",
        "🌐 所有仪器的接地端是否连接在一起？",
        "🔧 每个接线柱都拧紧了吗？（用手轻轻晃动检查）",
        "🎚️ 信号发生器是否设置为正弦波输出？",
        "📊 信号发生器的输出电压是否在测量过程中保持恒定？",
        "🔋 毫伏表的量程选择是否合适（读数在量程的1/3~2/3之间）？",
        "⏳ 仪器读数是否稳定后再记录（避免频率调节过快）？",
        "📝 数据记录的单位是否正确（A vs mA）？",
    ]
    for i, item in enumerate(checklist_items):
        st.checkbox(item, value=False, key=f"checklist_{i}")

    st.markdown("---")
    st.markdown("""
    <div style='background: #fff3e0; padding: 16px; border-radius: 10px;'>
    <b>💡 重要提示：</b>
    在实际实验教学中，接线错误是导致数据异常的最常见原因。建议学生在开始测量前，主动让老师或同学检查一次接线。
    使用本系统的AI连线检测功能时，<b>描述越详细准确，AI判断越可靠</b>。
    </div>
    """, unsafe_allow_html=True)

# ===================== 底部信息 =====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<small>🔬 RLC串联谐振实验AI智能辅助系统 | 基于豆包大模型 | 专为物理实验教学而设计</small>
</div>
""", unsafe_allow_html=True)
