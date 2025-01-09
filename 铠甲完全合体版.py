# 胡凯欣
# 开发时间:2025/1/9 14:28
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号问题

# 标题
st.title("基于Cox模型风险值计算与生存曲线可视化")
st.write("通过输入变量值计算风险值并展示生存曲线。")

# 选择模型
model_choice = st.sidebar.selectbox("选择要使用的模型", ["临床风险模型", "联合模型"])

# 加载模型
if model_choice == "临床风险模型":
    model_file_path = r"C:\Users\25845\Desktop\模型应用\crs_model.pkl"
    features = ['淋巴细胞比例', '血小板分布宽度', '球蛋白', '氯', '糖类抗原.242', 'grade', 'T', 'N', 'M']
else:
    model_file_path = r"C:\Users\25845\Desktop\模型应用\cmb_model.pkl"
    features = ['crs', 'prs']

cox_model = joblib.load(model_file_path)

# 用户输入特征值
st.header(f"输入特征值（{model_choice}）")

# 如果用户之前的输入存在，则使用它
if 'user_input' not in st.session_state:
    st.session_state.user_input = {}

for feature in features:
    if feature in ["T", "N", "M", "grade"]:  # 分类变量选择框
        if feature == "T":
            st.session_state.user_input[feature] = st.sidebar.selectbox(f"选择 {feature} 的值：", [1, 2, 3, 4])
        elif feature == "N":
            st.session_state.user_input[feature] = st.sidebar.selectbox(f"选择 {feature} 的值：", [0, 1, 2])
        elif feature == "M":
            st.session_state.user_input[feature] = st.sidebar.selectbox(f"选择 {feature} 的值：", [0, 1])
        elif feature == "grade":
            st.session_state.user_input[feature] = st.sidebar.selectbox(f"选择 {feature} 的值（1为高分化，2为低分化）：", [1, 2])
    else:  # 数值输入框
        if feature == '淋巴细胞比例':
            st.session_state.user_input[feature] = st.sidebar.number_input(f"输入 {feature} 的值 (%)：", value=st.session_state.user_input.get(feature, 0.0))
        elif feature == '血小板分布宽度':
            st.session_state.user_input[feature] = st.sidebar.number_input(f"输入 {feature} 的值(%)：", value=st.session_state.user_input.get(feature, 0.0))
        elif feature == '球蛋白':
            st.session_state.user_input[feature] = st.sidebar.number_input(f"输入 {feature} 的值(g/L)：", value=st.session_state.user_input.get(feature, 0.0))
        elif feature == '氯':
            st.session_state.user_input[feature] = st.sidebar.number_input(f"输入 {feature} 的值(mmol/L)：", value=st.session_state.user_input.get(feature, 0.0))
        elif feature == '糖类抗原.242':
            st.session_state.user_input[feature] = st.sidebar.number_input(f"输入 {feature} 的值(U/ml)：", value=st.session_state.user_input.get(feature, 0.0))
        elif feature == 'crs':
            st.session_state.user_input[feature] = st.sidebar.number_input(f"输入 {feature} 的值：", format="%0.6f",
                                                                           value=st.session_state.user_input.get(
                                                                               feature, 0.0))
        elif feature == 'prs':
            st.session_state.user_input[feature] = st.sidebar.number_input(f"输入 {feature} 的值：", format="%0.6f",
                                                                           value=st.session_state.user_input.get(
                                                                               feature, 0.0))

        else:
            st.session_state.user_input[feature] = st.sidebar.number_input(f"输入 {feature} 的值：", value=st.session_state.user_input.get(feature, 0.0))

# 转换为 DataFrame
input_df = pd.DataFrame([st.session_state.user_input])

# 计算风险值和可视化生存曲线
if st.button("计算风险值"):
    try:
        # 计算风险值
        risk_score = cox_model.predict_partial_hazard(input_df).values[0]
        st.success(f"计算的风险值为：{risk_score:.4f}")

        # 生存曲线可视化
        st.subheader("生存曲线")
        survival_curve = cox_model.predict_survival_function(input_df)

        # 使用 matplotlib 绘制曲线
        plt.figure(figsize=(6, 4))
        for i, curve in enumerate(survival_curve.T.values):
            plt.step(survival_curve.index, curve, where="post", label=f"曲线 {i + 1}")
        plt.title("预测的生存曲线")
        plt.xlabel("时间")
        plt.ylabel("生存概率")
        plt.grid(True)
        plt.legend(loc="best")
        st.pyplot(plt)

        # 生存概率表（接近365天, 1095天, 1825天）
        st.subheader("生存概率表（接近365天, 1095天, 1825天）")

        try:
            # 寻找接近365、1095、1825的索引
            closest_365 = survival_curve.index[np.abs(survival_curve.index - 365).argmin()]
            closest_1095 = survival_curve.index[np.abs(survival_curve.index - 1095).argmin()]
            closest_1825 = survival_curve.index[np.abs(survival_curve.index - 1825).argmin()]

            survival_365 = survival_curve.loc[closest_365].values[0]
            survival_1095 = survival_curve.loc[closest_1095].values[0]
            survival_1825 = survival_curve.loc[closest_1825].values[0]

            survival_table = pd.DataFrame({
                '时间': [closest_365, closest_1095, closest_1825],
                '生存概率': [survival_365, survival_1095, survival_1825]
            })

            st.write(survival_table)

        except KeyError as e:
            st.error(f"生存概率计算失败，错误信息：{e}")


    except Exception as e:
        st.error(f"风险值计算失败，错误信息：{e}")
