import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# 基本设置
st.set_page_config(page_title="油气序列分段工具", layout="wide")
st.title("\U0001f4c8 油气序列分段工具")

# 文件上传
uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])

# 初始化存储区
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.file_path = None
    st.session_state.line_name = ""

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        st.session_state.df = df
        st.session_state.file_path = uploaded_file.name
        st.success(f"\u6587\u4ef6\u6210\u529f\u4e0a\u4f20：{uploaded_file.name}")
    except Exception as e:
        st.error(f"\u65ad\u8a00\u6587\u4ef6\u5931\u8d25: {str(e)}")

if st.session_state.df is not None:
    df = st.session_state.df

    with st.sidebar:
        st.header(":wrench: 输入参数")

        line_name = st.text_input("\u5206\u6bb5\u6570\u636e\u7c7b\u578b", value=st.session_state.line_name)
        max_error = st.number_input("\u6700\u5927\u8bef\u5dee (max_error)", min_value=0.0, value=5.0)
        slope_threshold = st.number_input("\u659c率变化阈值 (slope_threshold)", min_value=0.0, value=0.5)

        start_date = st.selectbox("\u5206\u6bb5\u5f00\u59cb\u65e5期", df['date'].dt.strftime('%Y-%m-%d'))

        st.session_state.line_name = line_name

        add_data = st.checkbox("\u662f否添加新\u6570\u636e")
        if add_data:
            new_value = st.text_input("\u65b0增值")
            if st.button("\u786e定新增\u6570\u503c"):
                try:
                    value = float(new_value)
                    new_date = df['date'].iloc[-1] + timedelta(days=1)
                    new_row = pd.DataFrame({'date': [new_date], line_name: [value]})
                    df = pd.concat([df, new_row], ignore_index=True)
                    df = df.sort_values('date')
                    st.session_state.df = df
                    st.success(f"\u5df2\u6dfb\u52a0\u65b0\u503c {value} 日期 {new_date.date()}")
                except Exception as e:
                    st.error(f"\u65b0\u589e\u6570\u636e\u5931\u8d25：{str(e)}")

    if st.button("\u5f00\u59cb\u5206\u6bb5"):
        try:
            df = df[df['date'] >= pd.to_datetime(start_date)]
            x = np.arange(len(df))
            y = df[line_name].values

            segments = []
            start = 0
            while start < len(x) - 1:
                end = start + 1
                while end < len(x):
                    coeff = np.polyfit(x[start:end + 1], y[start:end + 1], 1)
                    fit = np.poly1d(coeff)
                    error = np.max(np.abs(y[start:end + 1] - fit(x[start:end + 1])))
                    if error > max_error:
                        break
                    end += 1
                segments.append((start, end - 1))
                start = end - 1

            # merge segments by slope
            merged_segments = []
            prev_start, prev_end = segments[0]
            prev_slope = np.polyfit(x[prev_start:prev_end + 1], y[prev_start:prev_end + 1], 1)[0]

            for i in range(1, len(segments)):
                cur_start, cur_end = segments[i]
                cur_slope = np.polyfit(x[cur_start:cur_end + 1], y[cur_start:cur_end + 1], 1)[0]

                if abs(cur_slope - prev_slope) < slope_threshold:
                    prev_end = cur_end
                else:
                    merged_segments.append((prev_start, prev_end, prev_slope))
                    prev_start, prev_end = cur_start, cur_end
                    prev_slope = cur_slope
            merged_segments.append((prev_start, prev_end, prev_slope))

            # plot
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(df['date'], y, label='\u539f\u59cb\u6570\u636e', color='blue')

            segment_info = []
            for i, (s, e, slope) in enumerate(merged_segments):
                xs = df['date'].iloc[s:e + 1]
                ys = y[s:e + 1]
                fit = np.poly1d(np.polyfit(np.arange(s, e + 1), ys, 1))
                ax.plot(xs, fit(np.arange(s, e + 1)), color='red', linewidth=2)

                ax.scatter(xs.iloc[0], ys[0], color='black')
                ax.scatter(xs.iloc[-1], ys[-1], color='black')
                ax.annotate(xs.iloc[0].strftime('%Y-%m-%d'), (xs.iloc[0], ys[0]), fontsize=8)
                ax.annotate(xs.iloc[-1].strftime('%Y-%m-%d'), (xs.iloc[-1], ys[-1]), fontsize=8)

                segment_info.append(f"\u6bb5 {i + 1}:\n  - \u8d77点: {xs.iloc[0].strftime('%Y-%m-%d')} ({ys[0]:.2f})\n  - \u7ed3束: {xs.iloc[-1].strftime('%Y-%m-%d')} ({ys[-1]:.2f})\n  - \u659c率: {slope:.4f}\n")

            ax.set_title('Time Series Segmentation')
            ax.grid(True)
            st.pyplot(fig)

            st.text_area("\u5206\u6bb5信息", value="\n".join(segment_info), height=300)

        except Exception as e:
            st.error(f"\u5206\u6bb5\u5931\u8d25：{str(e)}")
