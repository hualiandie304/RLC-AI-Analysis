# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import requests
import json


# ==============================================
# 🔴 修复：图表中文乱码（最终稳定版）
# ==============================================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# ==============================================

st.set_page_config(page_title="RLC串联谐振实验分析系统", layout="wide")

# ===================== 全局变量 =====================
API_URL_RESPONSES = "https://ark.cn-beijing.volces.com/api/v3/responses"
API_URL_CHAT = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

# 常用模型/endpoint列表
COMMON_ENDPOINTS = [
    {"name": "doubao-seed-2-0-pro-260215", "label": "豆包 Seed 2.0 Pro (260215)"},
    {"name": "doubao-seed-2-0-pro", "label": "豆包 Seed 2.0 Pro"},
    {"name": "doubao-3-5-mini", "label": "豆包 3.5 Mini"},
    {"name": "doubao-3-5-turbo", "label": "豆包 3.5 Turbo"},
    {"name": "doubao-3-5-pro", "label": "豆包 3.5 Pro"},
]

# ===================== 会话状态初始化 =====================
if 'experiment_step' not in st.session_state:
    st.session_state.experiment_step = 1  # 1:数据上传, 2:AI异常检测, 3:异常处理, 4:拟合成功, 5:实验总结
if 'f_data' not in st.session_state:
    st.session_state.f_data = None
if 'I_data' not in st.session_state:
    st.session_state.I_data = None
if 'ai_analysis_result' not in st.session_state:
    st.session_state.ai_analysis_result = None
if 'student_feedback' not in st.session_state:
    st.session_state.student_feedback = ""
if 'modification_applied' not in st.session_state:
    st.session_state.modification_applied = False
if 'final_summary' not in st.session_state:
    st.session_state.final_summary = None
if 'fitting_results' not in st.session_state:
    st.session_state.fitting_results = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'api_status' not in st.session_state:
    st.session_state.api_status = "not_checked"  # not_checked, checking, success, error
if 'api_status_message' not in st.session_state:
    st.session_state.api_status_message = ""
if 'endpoint' not in st.session_state:
    st.session_state.endpoint = ""
if 'use_custom_endpoint' not in st.session_state:
    st.session_state.use_custom_endpoint = False  # 🔴 修复：默认显示预设列表
if 'api_format' not in st.session_state:
    st.session_state.api_format = "chat"  # "chat" or "responses"
if 'timeout' not in st.session_state:
    st.session_state.timeout = 30
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'ai_analysis_context' not in st.session_state:
    st.session_state.ai_analysis_context = ""

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
    sigma_outliers = np.abs(current - mean_I) > 3 * std_I
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

# ===================== API验证函数 =====================
def check_api_connection(api_key, endpoint, api_format="chat", timeout=10):
    """检查API连接是否正常"""
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
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "请回复：连接测试成功"}]
                }
            ]
        }
    
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=timeout)
        resp.raise_for_status()
        resp_json = resp.json()
        
        # 尝试解析响应
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
    
    # 去除密钥前后空格
    if api_key:
        api_key = api_key.strip()
    
    if not api_key or len(api_key) < 10:
        return "❌ 错误：未配置API密钥！\n\n请在侧边栏输入豆包API密钥。"
    
    if not st.session_state.endpoint:
        return "❌ 错误：未配置Endpoint！\n\n请在侧边栏输入您的推理接入点Endpoint。"
    
    # 选择API格式
    if st.session_state.api_format == "chat":
        api_url = API_URL_CHAT
        data = {
            "model": st.session_state.endpoint,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens
        }
    else:
        api_url = API_URL_RESPONSES
        data = {
            "model": st.session_state.endpoint,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    debug_info = f"""
🔍 [调试信息]
- API URL: {api_url}
- API格式: {st.session_state.api_format}
- Endpoint: {st.session_state.endpoint}
- 密钥长度: {len(api_key)}
- 密钥前缀: {api_key[:15]}...
"""
    
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
        
        # 尝试两种响应格式
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

# ===================== AI功能函数 =====================
def ai_detect_anomalies(freq, current):
    
    full_data = []
    for f, i in zip(freq, current):
        full_data.append(f"f={f:.1f}Hz, I={i:.2e}A")
    full_data_str = "\n".join(full_data)
    
    # 计算本地分析结果，给AI提供参考
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
    
    prompt = f"""你是资深的物理实验指导专家，专门帮助学生分析RLC串联谐振实验数据。请严格按照以下要求输出。

【实验数据】
{full_data_str}

【背景知识】
RLC串联谐振电流-频率曲线特性：
1. 曲线应平滑、单峰、先升后降
2. 谐振峰两侧应对称
3. 电流值应为正数
4. 常见异常原因（按可能性从高到低排序）：
   1. 操作失误（接线松动、频率调节过快、仪器读数未稳定）
   2. 参数设置错误（恒流源/信号源参数设置不当）
   3. 数据记录错误（抄写错误、单位混淆）
   4. 仪器问题（仪器故障、量程选择不当）
   5. 环境干扰（电源波动、电磁干扰）

【{local_analysis}】

【输出格式要求】
请使用以下格式输出，每个部分单独成行，保持清晰的分段：

【异常检测结果】

1. 异常是否存在：是/否

2. 异常点（如有）：列出具体频率值

3. 判断依据：详细说明判断理由

【异常原因分析】

4. 可能原因分析（按可能性排序，标注概率）：
   原因1（概率XX%）：具体说明
   原因2（概率XX%）：具体说明
   ...

【验证方法】

5. 验证方法（从最可能的开始，列出具体步骤）：
   方法1：具体操作步骤
   方法2：具体操作步骤
   ...

【修改建议】

6. 修改建议：针对最可能的原因给出具体方案

【重要要求】
- 每个大标题用【】括起来
- 每个编号的内容单独成行
- 保持适当的空行，不要太密集
- 概率总和为100%
- 验证方法要具体可操作
- 绝对不要使用任何数学公式或LaTeX符号
- 用通俗易懂的中文描述物理概念
- 内容要深入具体，不要太笼统
"""
    # 保存上下文用于对话
    st.session_state.ai_analysis_context = prompt
    return call_ai(prompt, max_tokens=1800)

def ai_chat(user_message, context=""):
    """AI对话功能"""
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
    prompt = f"""你是资深物理实验教师，为RLC实验写总结，要求格式清晰、内容充实、有适当空行。

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
请按以下结构撰写总结，每个大标题用【】括起来，保持适当空行：

【实验概述】

简要说明实验目的和完成情况

【异常原因深入剖析】

详细分析本次异常的根本原因，以及该异常是如何影响实验结果的（从物理原理角度解释，用通俗易懂的语言）

【实验结果分析】

分析拟合结果、谐振特性等

【实验收获与改进建议】

总结学生从本次异常处理中学到的经验，以及未来如何避免类似问题，给出具体可操作的建议
"""
    else:
        prompt += """
请按以下结构撰写总结，每个大标题用【】括起来，保持适当空行：

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
- 保持适当的空行，排版清晰
"""
    return call_ai(prompt, max_tokens=2200)

# ===================== 侧边栏配置 =====================
with st.sidebar:
    st.subheader("⚙️ 设置")
    
    st.session_state.debug_mode = st.checkbox(
        "启用调试模式",
        value=st.session_state.debug_mode,
        help="显示详细的API调用信息，便于排查问题"
    )
    
    st.markdown("---")
    st.subheader("🔑 API密钥配置")
    
    current_key = get_api_key()
    if current_key and len(current_key) >= 10:
        st.success(f"✅ API密钥已配置 (前10位: {current_key[:10]}...)")
    else:
        st.warning("⚠️ 请配置API密钥")
    
    new_key = st.text_input(
        "豆包API密钥",
        type="password",
        placeholder="ark-...",
        value=st.session_state.api_key if st.session_state.api_key else "",
        key="api_key_input"
    )
    
    if new_key and new_key != st.session_state.api_key:
        st.session_state.api_key = new_key
        st.success("✅ API密钥已更新")
        st.session_state.api_status = "not_checked"
        st.rerun()



    
    st.markdown("---")
    st.subheader("🎯 Endpoint配置")
    st.info("💡 需要在火山引擎控制台创建推理接入点获取endpoint")

# ===================== 修复：云端初始化状态 =====================
    use_preset = st.checkbox("使用预设endpoint", value=not st.session_state.use_custom_endpoint)
    st.session_state.use_custom_endpoint = not use_preset

    if use_preset:
        preset_options = [m["label"] for m in COMMON_ENDPOINTS]
        preset_names = [m["name"] for m in COMMON_ENDPOINTS]
    
        try:
            current_idx = preset_names.index(st.session_state.endpoint) if st.session_state.endpoint in preset_names else 0
        except:
            current_idx = 0
    
        selected_label = st.selectbox(
            "选择预设endpoint",
            preset_options,
            index=current_idx,
            help="选择预设的豆包endpoint"
        )
    
        selected_name = [m["name"] for m in COMMON_ENDPOINTS if m["label"] == selected_label][0]
    
        if selected_name != st.session_state.endpoint:
            st.session_state.endpoint = selected_name
            st.session_state.api_status = "not_checked"
            st.rerun()
    else:
        custom_endpoint = st.text_input(
            "输入您的推理接入点Endpoint",
            value=st.session_state.endpoint if st.session_state.endpoint else "",
            placeholder="例如: ep-20240101000000-abcde",
            key="endpoint_input"
        )
    
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
    
    # 显示当前状态
    if st.session_state.api_status == "checking":
        st.info("🔄 正在检查API连接...")
    elif st.session_state.api_status == "success":
        st.success(st.session_state.api_status_message)
    elif st.session_state.api_status == "error":
        st.error(st.session_state.api_status_message)
    else:
        st.info("⚪ 点击下方按钮检查连接")
    
    # 检查按钮
    if st.button("检查API连接", type="primary", key="check_api_btn"):
        st.session_state.api_status = "checking"
        st.rerun()
    
    # 执行检查
    if st.session_state.api_status == "checking":
        with st.spinner("正在检查连接..."):
            success, message = check_api_connection(
                current_key,
                st.session_state.endpoint,
                st.session_state.api_format,
                min(15, st.session_state.timeout)
            )
        if success:
            st.session_state.api_status = "success"
        else:
            st.session_state.api_status = "error"
        st.session_state.api_status_message = message
        st.rerun()
    
    st.markdown("---")
    st.subheader("⚙️ API高级设置")
    
    # API格式选择
    api_format_option = st.radio(
        "API格式",
        ["Chat Completions (推荐)", "Responses"],
        index=0 if st.session_state.api_format == "chat" else 1,
        help="选择API调用格式"
    )
    new_format = "chat" if "Chat" in api_format_option else "responses"
    if new_format != st.session_state.api_format:
        st.session_state.api_format = new_format
        st.session_state.api_status = "not_checked"
        st.rerun()
    
    # 超时设置
    new_timeout = st.slider(
        "请求超时时间(秒)",
        10, 120, st.session_state.timeout,
        help="设置API请求超时时间"
    )
    if new_timeout != st.session_state.timeout:
            st.session_state.timeout = new_timeout
            st.rerun()
    
    st.markdown("---")
    st.subheader("⚙️ 实验设置")
    current_unit = st.selectbox("电流单位", ["A", "mA"])
    L_fixed = st.number_input("固定电感 L (H)", 0.0001, 0.1, 0.01, format="%.4f")
    
    st.markdown("---")
    st.subheader("手动修正选项")
    use_manual_f0 = st.checkbox("手动设置谐振频率 f0", value=False)
    manual_f0 = st.number_input("谐振频率 f0 (Hz)", 0.0, 100000.0, 3000.0, format="%.1f", disabled=not use_manual_f0)
    use_manual_f1 = st.checkbox("手动设置低频半功率点 f1", value=False)
    manual_f1 = st.number_input("低频半功率点 f1 (Hz)", 0.0, 100000.0, 2700.0, format="%.1f", disabled=not use_manual_f1)
    use_manual_f2 = st.checkbox("手动设置高频半功率点 f2", value=False)
    manual_f2 = st.number_input("高频半功率点 f2 (Hz)", 0.0, 100000.0, 3300.0, format="%.1f", disabled=not use_manual_f2)

# ===================== 配置说明 =====================
with st.expander("⚙️ 系统状态与配置说明", expanded=False):
    st.subheader("📋 使用说明")
    
    current_key = get_api_key()
    api_ok = st.session_state.api_status == "success"
    
    if api_ok:
        st.success("✅ API连接正常，可以使用AI功能")
    elif st.session_state.api_status == "error":
        st.error("❌ API连接失败，请检查配置")
    else:
        st.info("⚪ 请在侧边栏配置并检查API连接")
    
    st.markdown("""
    **快速开始：**
    1. 在侧边栏配置API密钥和Endpoint
    2. 点击"检查API连接"验证配置
    3. 点击下方"生成示例数据"或上传Excel数据
    4. 开始使用AI分析
    
    **获取豆包API密钥和Endpoint：**
    1. 访问 [火山引擎控制台](https://console.volcengine.com/ark)
    2. 注册/登录账号
    3. 进入"推理接入点"创建新接入点
    4. 选择模型并创建接入点
    5. 获取 Endpoint（格式：ep-20240101000000-abcde）
    6. 在"API访问"获取 API Key（格式：ark-xxxxxxxxxx）
    
    **⚠️ 重要提示：**
    - 输入API密钥时，请确保不要带前后空格
    - 如果超时，请在侧边栏增加超时时间
    - 推荐使用"Chat Completions"格式
    """)

# ===================== 主界面 =====================
st.title("🔬 RLC串联谐振实验分析系统")

current_key = get_api_key()
api_ok = st.session_state.api_status == "success"
if not api_ok:
    st.warning("⚠️ API连接未验证，请在侧边栏配置并检查连接")

steps = ["1️⃣ 数据上传", "2️⃣ 检测/拟合", "3️⃣ 异常处理", "4️⃣ 拟合分析", "5️⃣ 实验总结"]
current_step = st.session_state.experiment_step
st.progress((current_step - 1) / 4)
st.markdown(f"**当前阶段：{steps[current_step - 1]}**")

# ===================== 示例数据生成 =====================
with st.expander("📁 没有数据？生成示例数据", expanded=True):
    st.subheader("生成示例RLC数据")
    col1, col2, col3 = st.columns(3)
    with col1:
        sample_f0 = st.number_input("示例谐振频率 (Hz)", 100.0, 10000.0, 3000.0)
    with col2:
        sample_Q = st.number_input("示例品质因数 Q", 1.0, 100.0, 30.0)
    with col3:
        sample_noise = st.slider("添加噪声水平", 0.0, 0.2, 0.02, 0.01)
    
    add_anomaly = st.checkbox("添加模拟异常点", value=False)
    
    if st.button("生成示例数据", type="primary"):
        num_points = 50
        freq = np.linspace(sample_f0 * 0.5, sample_f0 * 1.5, num_points)
        omega = 2 * np.pi * freq
        omega0 = 2 * np.pi * sample_f0
        I = 1 / np.sqrt(1 + (sample_Q * (omega/omega0 - omega0/omega))**2)
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

# ===================== 实验流程 =====================
if st.session_state.experiment_step >= 1:
    st.header("📊 步骤1：上传实验数据")
    
    if st.session_state.f_data is not None:
        st.info("📁 已有数据加载中...")
        f_data = st.session_state.f_data
        I_data = st.session_state.I_data
        
        fig, ax = plt.subplots(figsize=(10, 4))
        if current_unit == "mA":
            ax.scatter(f_data, I_data*1000, c="blue", label="Experiment Data")
            ax.set_ylabel("Current (mA)")
        else:
            ax.scatter(f_data, I_data, c="blue", label="Experiment Data")
            ax.set_ylabel("Current (A)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title("Data Curve")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # 本地快速检测
        try:
            local_has_outlier, outliers_mask, local_reasons = enhanced_outlier_check(f_data, I_data)
            if local_has_outlier:
                st.warning(f"⚠️ 本地检测发现可能异常：{', '.join(local_reasons)}")
            else:
                st.success("✅ 本地检测数据正常，可直接拟合")
        except:
            pass
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🤖 使用AI完整检测", type="primary"):
                st.session_state.experiment_step = 2
                st.rerun()
        with col2:
            if st.button("📊 直接开始拟合", type="secondary"):
                st.session_state.experiment_step = 4
                st.rerun()
        with col3:
            if st.button("🗑️ 清除数据"):
                for key in ['f_data', 'I_data', 'ai_analysis_result', 'final_summary', 'fitting_results']:
                    if key in st.session_state:
                        st.session_state[key] = None
                st.session_state.experiment_step = 1
                st.rerun()
    else:
        uploaded_file = st.file_uploader("上传Excel（第一列频率，第二列电流，按频率从小到大排列）", type=["xlsx", "xls"], key="data_upload")
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                f_data = df.iloc[:, 0].values
                I_data = df.iloc[:, 1].values
                if current_unit == "mA":
                    I_data = I_data / 1000.0
                st.session_state.f_data = f_data
                st.session_state.I_data = I_data
                st.success("✅ 数据上传成功！")
                st.rerun()
            except Exception as e:
                st.error(f"数据读取失败：{str(e)}")

if st.session_state.experiment_step >= 2 and st.session_state.f_data is not None:
    st.header("🤖 步骤2：AI异常检测")
    
    # AI分析结果
    if st.session_state.ai_analysis_result is None:
        with st.spinner("AI正在分析数据，请稍候..."):
            result = ai_detect_anomalies(st.session_state.f_data, st.session_state.I_data)
            st.session_state.ai_analysis_result = result
    st.markdown(st.session_state.ai_analysis_result)
    
    # AI对话窗口
    st.markdown("---")
    st.subheader("💬 AI对话助手")
    st.info("有问题？随时问AI！可以询问异常原因、验证方法、物理原理等。")
    
    # 显示历史消息
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 用户输入
    if prompt := st.chat_input("问AI关于实验的问题...", key="chat_step2"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("AI正在思考..."):
                response = ai_chat(prompt, st.session_state.ai_analysis_context)
            st.markdown(response)
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
    
    st.markdown("---")
    
    has_anomaly = "存在异常：是" in st.session_state.ai_analysis_result or "异常是否存在：是" in st.session_state.ai_analysis_result
    if has_anomaly:
        st.warning("⚠️ 检测到异常数据！")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 进入异常处理流程", type="primary"):
                st.session_state.experiment_step = 3
                st.rerun()
        with col2:
            if st.button("🔄 重新分析", type="secondary"):
                st.session_state.ai_analysis_result = None
                st.rerun()
    else:
        st.success("✅ 数据正常，可继续拟合！")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 开始拟合分析", type="primary"):
                st.session_state.experiment_step = 4
                st.rerun()
        with col2:
            if st.button("🔄 重新分析", type="secondary"):
                st.session_state.ai_analysis_result = None
                st.rerun()

if st.session_state.experiment_step >= 3 and ("存在异常：是" in (st.session_state.ai_analysis_result or "") or "异常是否存在：是" in (st.session_state.ai_analysis_result or "")):
    st.header("🔧 步骤3：异常处理")
    
    # AI对话窗口
    st.subheader("💬 AI对话助手")
    st.info("定位异常时有问题？问AI！可以询问验证方法、物理原理等。")
    
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("问AI关于异常处理的问题...", key="chat_step3"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("AI正在思考..."):
                response = ai_chat(prompt, st.session_state.ai_analysis_context)
            st.markdown(response)
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
    
    st.markdown("---")
    
    st.subheader("📝 学生反馈与定位")
    st.session_state.student_feedback = st.text_area(
        "请描述您定位到的异常原因和验证过程：",
        value=st.session_state.student_feedback,
        height=150,
        placeholder="例如：我通过检查接线发现谐振峰附近的接线柱松动，重新连接后数据恢复正常..."
    )
    st.subheader("🔄 重新上传修正后的数据")
    new_uploaded_file = st.file_uploader("上传修正后的Excel文件", type=["xlsx", "xls"], key="fixed_data_upload")
    if new_uploaded_file is not None:
        try:
            df = pd.read_excel(new_uploaded_file)
            f_data = df.iloc[:, 0].values
            I_data = df.iloc[:, 1].values
            if current_unit == "mA":
                I_data = I_data / 1000.0
            st.session_state.f_data = f_data
            st.session_state.I_data = I_data
            st.session_state.modification_applied = True
            st.success("✅ 修正后数据上传成功！")
            fig, ax = plt.subplots(figsize=(10, 4))
            if current_unit == "mA":
                ax.scatter(f_data, I_data*1000, c="green", label="Corrected Data")
                ax.set_ylabel("Current (mA)")
            else:
                ax.scatter(f_data, I_data, c="green", label="Corrected Data")
                ax.set_ylabel("Current (A)")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_title("Corrected Data Curve")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            if st.button("✅ 确认修正，重新检测", type="primary"):
                st.session_state.ai_analysis_result = None
                st.session_state.experiment_step = 2
                st.rerun()
        except Exception as e:
            st.error(f"数据读取失败：{str(e)}")

if st.session_state.experiment_step >= 4 and st.session_state.f_data is not None:
    st.header("📈 步骤4：拟合分析")
    f_data = st.session_state.f_data
    I_data = st.session_state.I_data
    try:
        local_has_outlier, outliers_mask, local_reasons = enhanced_outlier_check(f_data, I_data)
        if local_has_outlier:
            st.warning("⚠️ 本地检测仍发现异常，请检查数据")
            for reason in local_reasons:
                st.markdown(reason)
        f_clean = f_data
        I_clean = I_data
        st.subheader("RLC 谐振参数计算")
        if use_manual_f0:
            f0_fit = manual_f0
            closest = np.argmin(np.abs(f_clean - f0_fit))
            I0 = I_clean[closest]
        else:
            max_i_idx = np.argmax(I_clean)
            f0_fit = f_clean[max_i_idx]
            I0 = I_clean[max_i_idx]
        f1_man = manual_f1 if use_manual_f1 else None
        f2_man = manual_f2 if use_manual_f2 else None
        f1, f2, I_half = calculate_half_power_points(f_clean, I_clean, f0_fit, I0, f1_man, f2_man)
        BW = f2 - f1
        Q_fit = f0_fit / BW if BW != 0 else 0
        C_fit = 1 / (4 * np.pi**2 * f0_fit**2 * L_fixed)
        R_fit = (2 * np.pi * f0_fit * L_fixed) / Q_fit if Q_fit != 0 else 0
        f_fit = np.linspace(f_clean.min(), f_clean.max(), 1000)
        omega = 2 * np.pi * f_fit
        XL = omega * L_fixed
        XC = 1 / (omega * C_fit)
        Z = np.sqrt(R_fit**2 + (XL - XC)**2)
        I_fit = (I0 * R_fit) / Z
        I_fit_on_data = np.interp(f_clean, f_fit, I_fit)
        r2 = 1 - np.sum((I_clean - I_fit_on_data)**2) / np.sum((I_clean - np.mean(I_clean))**2) if np.sum((I_clean - np.mean(I_clean))**2) != 0 else 0
        st.session_state.fitting_results = {
            'f0': f0_fit,
            'Q': Q_fit,
            'BW': BW,
            'r2': r2,
            'C': C_fit,
            'R': R_fit
        }
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("电感 L", f"{L_fixed*1000:.2f} mH")
        with col2: st.metric("电容 C", f"{C_fit*1e6:.2f} μF")
        with col3: st.metric("电阻 R", f"{R_fit:.2f} Ω")
        with col4: st.metric("Q值", f"{Q_fit:.2f}")
        col5, col6, col7 = st.columns(3)
        with col5: st.metric("谐振频率 f0", f"{f0_fit:.2f} Hz")
        with col6: st.metric("带宽 BW", f"{BW:.2f} Hz")
        with col7: st.metric("R²", f"{r2:.4f}")
        st.subheader("谐振曲线")
        fig, ax = plt.subplots(figsize=(10, 5))
        if current_unit == "mA":
            ax.scatter(f_clean, I_clean*1000, c="blue", label="Experiment Data")
            ax.plot(f_fit, I_fit*1000, "g-", label="Fitting Curve")
            ax.plot(f0_fit, I0*1000, "ro", label="Resonance Point")
            ax.plot([f1, f2], [I_half*1000, I_half*1000], "ms", label="Half-Power Points")
            ax.set_ylabel("Current (mA)")
        else:
            ax.scatter(f_clean, I_clean, c="blue", label="Experiment Data")
            ax.plot(f_fit, I_fit, "g-", label="Fitting Curve")
            ax.plot(f0_fit, I0, "ro", label="Resonance Point")
            ax.plot([f1, f2], [I_half, I_half], "ms", label="Half-Power Points")
            ax.set_ylabel("Current (A)")
        ax.set_xlabel("Frequency (Hz)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        st.success("✅ 拟合完成！")
        if st.button("📝 生成实验总结", type="primary"):
            st.session_state.experiment_step = 5
            st.rerun()
    except Exception as e:
        st.error(f"拟合过程出错：{str(e)}")

if st.session_state.experiment_step >= 5 and st.session_state.fitting_results is not None:
    st.header("📝 步骤5：实验总结")
    
    has_anomaly = st.session_state.modification_applied or ("存在异常：是" in (st.session_state.ai_analysis_result or "") or "异常是否存在：是" in (st.session_state.ai_analysis_result or ""))
    if st.session_state.final_summary is None:
        with st.spinner("AI正在生成实验总结，请稍候..."):
            summary = ai_generate_summary(
                st.session_state.f_data,
                st.session_state.I_data,
                st.session_state.fitting_results,
                st.session_state.student_feedback,
                has_anomaly=has_anomaly,
                anomaly_info=st.session_state.ai_analysis_result if has_anomaly else ""
            )
            st.session_state.final_summary = summary
    st.markdown(st.session_state.final_summary)
    
    # AI对话窗口
    st.markdown("---")
    st.subheader("💬 AI对话助手")
    st.info("对实验有疑问？问AI！可以询问物理原理、拓展知识等。")
    
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("问AI关于实验的问题...", key="chat_step5"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("AI正在思考..."):
                context = f"实验总结：{st.session_state.final_summary}"
                response = ai_chat(prompt, context)
            st.markdown(response)
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
    
    st.markdown("---")
    st.success("🎉 实验完成！")
    if st.button("🔄 开始新实验"):
        for key in list(st.session_state.keys()):
            if key not in ['debug_mode', 'endpoint', 'api_format', 'timeout', 'api_key', 'use_custom_endpoint', 'api_status', 'api_status_message']:
                del st.session_state[key]
        st.session_state.experiment_step = 1
        st.rerun()
