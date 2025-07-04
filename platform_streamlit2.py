import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time  # 引入time模块用于实现延时

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 16})

# --- 全局固定参数 ---
MAX_ERROR = 50
SLOPE_THRESHOLD = 2

# --- 应用配置 ---
st.set_page_config(page_title="油气序列分段与分析工具", layout="wide")
st.title("油气序列分段与分析工具")


# --- 核心功能函数 ---

# 【新增】打字机效果函数
def stream_text_to_placeholder(text, placeholder, delay=0.02):
    """
    将文本以打字机效果流式输出到Streamlit的占位符中。
    - text: 要显示的完整文本。
    - placeholder: st.empty() 创建的占位符对象。
    - delay: 每个字符显示的延迟时间（秒）。
    """
    displayed_text = ""
    # 添加一个模拟的光标
    cursor = "▌"
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text + cursor, unsafe_allow_html=True)
        time.sleep(delay)
    # 循环结束后，显示完整文本并移除光标
    placeholder.markdown(displayed_text, unsafe_allow_html=True)


def piecewise_linear_representation(x, y, max_error):
    """分段线性表示算法"""
    segments = []
    n = len(x)
    i = 0
    while i < n - 1:
        j = i + 1
        while j < n:
            if len(x[i:j + 1]) < 2:
                j += 1
                continue

            coeff = np.polyfit(x[i:j + 1], y[i:j + 1], 1)
            fit = np.poly1d(coeff)
            error = np.max(np.abs(y[i:j + 1] - fit(x[i:j + 1])))

            if error > max_error:
                break
            j += 1

        end_point = j - 1
        if end_point <= i:
            end_point = i + 1

        segments.append((i, end_point))
        i = end_point

        if i == n - 2 and end_point == n - 2:
            segments.append((n - 2, n - 1))
            break

    return segments


def generate_analysis_report(df, segments, line_name, anomaly_detected, slope_threshold):
    """生成详细的、非机械化的分析报告"""
    report_lines = []

    report_lines.append(f"## {line_name} 序列分段分析报告")
    report_lines.append("---")
    start_report_date = df['date'].iloc[0].strftime('%Y-%m-%d')
    end_report_date = df['date'].iloc[-1].strftime('%Y-%m-%d')
    report_lines.append(f"**分析周期:** 从 {start_report_date} 到 {end_report_date}")
    report_lines.append(f"**分析数据列:** {line_name}")
    report_lines.append(f"**总分段数:** {len(segments)} 段")
    report_lines.append("\n### 总体趋势分析")

    overall_start_val = df[line_name].iloc[0]
    overall_end_val = df[line_name].iloc[-1]
    overall_change = overall_end_val - overall_start_val
    if overall_change > 0:
        overall_trend = f"在整个分析周期内，数值整体呈现 **上升** 趋势，从 {overall_start_val:.2f} 增长到 {overall_end_val:.2f}。"
    elif overall_change < 0:
        overall_trend = f"在整个分析周期内，数值整体呈现 **下降** 趋势，从 {overall_start_val:.2f} 减少到 {overall_end_val:.2f}。"
    else:
        overall_trend = f"在整个分析周期内，数值整体保持稳定，维持在 {overall_start_val:.2f} 水平。"
    report_lines.append(overall_trend)
    report_lines.append("\n### 各分段详细解读")

    for idx, (start, end) in enumerate(segments):
        seg_start_date = df['date'].iloc[start].strftime('%Y-%m-%d')
        seg_end_date = df['date'].iloc[end].strftime('%Y-%m-%d')
        seg_start_val = df[line_name].iloc[start]
        seg_end_val = df[line_name].iloc[end]
        duration = (df['date'].iloc[end] - df['date'].iloc[start]).days + 1

        if end > start:
            slope = (seg_end_val - seg_start_val) / (end - start)
        else:
            slope = 0

        if abs(slope) < 0.5:
            trend_desc = "趋势平稳期"
            analysis = f"数值在此阶段表现稳定，波动较小，维持在 {seg_start_val:.2f} 附近。"
        elif slope > slope_threshold:
            trend_desc = "快速增长期"
            analysis = f"数值呈现快速增长，从 {seg_start_val:.2f} 上升至 {seg_end_val:.2f}，表明了积极的变化。"
        elif slope > 0:
            trend_desc = "缓慢增长期"
            analysis = f"数值在此阶段平缓上升，从 {seg_start_val:.2f} 增长到 {seg_end_val:.2f}。"
        elif slope < -slope_threshold:
            trend_desc = "快速下降期"
            analysis = f"数值出现显著下降，从 {seg_start_val:.2f} 锐减至 {seg_end_val:.2f}，可能需要关注其原因。"
        else:
            trend_desc = "缓慢下降期"
            analysis = f"数值在此阶段平缓回落，从 {seg_start_val:.2f} 减少到 {seg_end_val:.2f}。"

        report_lines.append(f"\n**分段 {idx + 1}: {trend_desc} ({seg_start_date} to {seg_end_date})**")
        report_lines.append(f"- **持续时间:** {duration} 天")
        report_lines.append(f"- **数值变化:** 从 {seg_start_val:.2f} 到 {seg_end_val:.2f}")
        report_lines.append(f"- **分析解读:** {analysis}")

    report_lines.append("\n### 结论与建议")
    if anomaly_detected:
        anomaly_text = "重要提醒：分析显示，数据末端出现了异常波动（最后一个分段过短），这通常意味着近期产量或压力发生了急剧变化。建议立即核查相关生产动态或设备状况，并采取相应措施。"
        report_lines.append(f"**<font color='red'>【异常预警】</font>** {anomaly_text}")
    else:
        report_lines.append("当前序列整体变化符合分段趋势，未检测到末端突变异常。建议持续监控数据变化。")

    return "\n".join(report_lines)


# --- Streamlit UI 布局 ---

if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_filename' not in st.session_state:
    st.session_state.original_filename = None

with st.sidebar:
    st.header("1. 数据上传与设置")
    uploaded_file = st.file_uploader("选择CSV文件", type="csv")

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.original_filename:
            try:
                df = pd.read_csv(uploaded_file)
                if 'date' not in df.columns:
                    st.error("CSV文件中必须包含 'date' 列。")
                    st.session_state.df = None
                else:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values(by='date').reset_index(drop=True)
                    st.session_state.df = df
                    st.session_state.original_filename = uploaded_file.name
                    st.success(f"文件 '{uploaded_file.name}' 加载成功！")
            except Exception as e:
                st.error(f"加载文件时出错: {e}")
                st.session_state.df = None

    if st.session_state.df is not None:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        line_name = st.selectbox("选择要分析的数据列:", options=numeric_cols)

        st.header("2. 日期选择")
        dates_options = st.session_state.df['date'].dt.strftime('%Y-%m-%d').tolist()
        start_date = st.selectbox("选择分段开始日期:", options=dates_options)

if st.session_state.df is None:
    st.info("请在左侧侧边栏上传一个CSV文件以开始分析。")
    st.markdown("---")
    st.markdown("...")  # 省略提示信息
else:
    df = st.session_state.df

    with st.expander("📝 添加新数据点 (可选)"):
        col1, col2 = st.columns(2)
        with col1:
            new_value = st.text_input("输入新数值:", key="new_val_input")
        with col2:
            st.write("")
            st.write("")
            if st.button("确定新增", key="add_data_btn"):
                if new_value and 'line_name' in locals() and line_name:
                    try:
                        current_df = st.session_state.df.copy()
                        last_date = current_df['date'].iloc[-1]
                        new_date = last_date + pd.Timedelta(days=1)

                        new_row_data = {'date': new_date}
                        for col in current_df.columns:
                            if col != 'date':
                                new_row_data[col] = np.nan
                        new_row_data[line_name] = float(new_value)

                        new_row = pd.DataFrame([new_row_data])

                        st.session_state.df = pd.concat([current_df, new_row], ignore_index=True)
                        st.success(f"新增成功！日期: {new_date.strftime('%Y-%m-%d')}, 数值: {float(new_value)}")
                        st.rerun()

                    except ValueError:
                        st.error("请输入有效的数值！")
                    except Exception as e:
                        st.error(f"发生错误: {str(e)}")
                else:
                    st.warning("请输入数值并选择要分析的数据列。")

    st.markdown("---")
    st.subheader("数据预览与下载")
    st.dataframe(st.session_state.df.tail())

    csv_buffer = io.StringIO()
    st.session_state.df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')

    st.download_button(
        label="📥 下载修改后的CSV文件",
        data=csv_buffer.getvalue(),
        file_name=f"updated_{st.session_state.original_filename}",
        mime="text/csv"
    )

    st.markdown("---")
    if st.button("🚀 开始分段与生成报告", type="primary"):
        with st.spinner("正在进行计算分析，请稍候..."):
            # 这部分计算逻辑不变
            selected_start_date = pd.to_datetime(start_date)
            analysis_df = st.session_state.df[st.session_state.df['date'] >= selected_start_date].copy().reset_index(
                drop=True)

            if analysis_df.empty or len(analysis_df) < 2:
                st.error("所选日期范围内数据不足，无法进行分析。")
            else:
                dates = analysis_df['date'].values
                daily_values = analysis_df[line_name].values
                segments = piecewise_linear_representation(np.arange(len(dates)), daily_values, MAX_ERROR)

                anomaly_detected = False
                if segments and len(segments) > 1:
                    last_seg_start, last_seg_end = segments[-1]
                    if last_seg_end - last_seg_start <= 1:
                        anomaly_detected = True
                        st.warning("【异常预警】数据末端出现剧烈波动，最后一个分段过短，请关注最新动态！")

        # spinner 结束后，开始显示结果
        st.header("分析结果")

        # 1. 绘制图表 (这部分是瞬时完成的)
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(dates, daily_values, 'o-', label='original_data', color='cornflowerblue', markersize=3, alpha=0.7)
        segment_point_indices = []
        if segments:
            segment_point_indices.append(segments[0][0])
            for start, end in segments:
                segment_point_indices.append(end)
            segment_point_indices = sorted(list(set(segment_point_indices)))

        for idx, (start, end) in enumerate(segments):
            seg_dates, seg_values = dates[start:end + 1], daily_values[start:end + 1]
            if anomaly_detected and idx == len(segments) - 1:
                ax.plot(seg_dates, seg_values, color='orangered', linewidth=3, linestyle='--', label=f'Abnormal_segement')
            else:
                label = 'Trend_segement' if idx == 0 else "_nolegend_"
                ax.plot(seg_dates, seg_values, color='red', linewidth=3, label=label)

        if segment_point_indices:
            point_dates, point_values = dates[segment_point_indices], daily_values[segment_point_indices]
            ax.scatter(point_dates, point_values, color='green', s=100, zorder=5, label='分段点')

        # --- 【核心修改】在分段点上标注日期 ---
        # 动态计算文本的垂直偏移量，使其能自适应不同数值范围的图表
        y_min, y_max = ax.get_ylim()
        y_offset = (y_max - y_min) * 0.02  # 偏移量为Y轴范围的2%

        for date, value in zip(point_dates, point_values):
            # 使用 月-日 格式避免标签过长导致重叠
            date_str = pd.to_datetime(date).strftime('%m-%d')
            ax.text(
                x=date,
                y=value + y_offset,  # 在点的上方显示文本
                s=date_str,
                ha='center',  # 水平居中对齐
                va='bottom',  # 垂直底部对齐
                fontsize=9,
                color='dimgray',  # 使用深灰色，避免喧宾夺主
                fontweight='bold'
            )
        # --- 【修改结束】 ---

        ax.set_title(f"{line_name} Segement_result", fontsize=16)
        ax.set_xlabel("date", fontsize=12)
        ax.set_ylabel(line_name, fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # 2. 生成并以打字机效果显示报告
        st.subheader("📜 分析报告")
        report_content = generate_analysis_report(analysis_df, segments, line_name, anomaly_detected, SLOPE_THRESHOLD)

        # 【核心修改】创建占位符并调用打字机函数
        report_placeholder = st.empty()
        stream_text_to_placeholder(report_content, report_placeholder, delay=0.015)  # 调整 delay 可以改变打字速度

        # 3. 提供报告下载 (下载按钮在报告“打完”后出现)
        st.download_button(
            label="📥 下载分析报告 (.txt)",
            data=report_content.encode('utf-8'),
            file_name=f"{line_name}_analysis_report.txt",
            mime="text/plain"
        )
