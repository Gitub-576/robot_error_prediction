
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    prepare_app,
    preprocess_data,
    FEATURE_COLUMNS,
    SENSORS
)


# =====================================
# 페이지 설정
# =====================================

st.set_page_config(
    page_title="데이터 탐색",
    page_icon="📊",
    layout="wide"
)

st.title("📊 데이터 탐색")

st.markdown("""
로봇의 힘과 토크 센서 데이터가 어떤 분포를 가지는지 확인하고,
Class에 따라 측정값에 차이가 나타나는지 분석합니다.
""")

df = prepare_app()

if df is None:
    st.stop()

data = preprocess_data(df)

classes = sorted(data["Class"].unique(), key=str)


# =====================================
# 1. Class별 데이터 개수
# =====================================

st.divider()

st.subheader("1. Class별 데이터 개수 비교")

counts = (
    data["Class"]
    .value_counts()
    .sort_index(key=lambda x: x.astype(str))
    .rename_axis("Class")
    .reset_index(name="데이터 개수")
)

fig = px.bar(
    counts,
    x="Class",
    y="데이터 개수",
    color="Class",
    text="데이터 개수",
    title="Class별 데이터 분포"
)

fig.update_layout(
    showlegend=False,
    xaxis_title="Class",
    yaxis_title="데이터 개수"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "Class별 데이터 개수를 비교하여 특정 유형에 데이터가 "
    "집중되어 있는지 확인할 수 있습니다."
)


# =====================================
# 2. 힘 측정값 분포
# =====================================

st.divider()

st.subheader("2. 힘(Force) 측정값 분포")

force_sensors = ["Fx", "Fy", "Fz"]

selected_force = st.selectbox(
    "분석할 힘 센서 선택",
    force_sensors,
    format_func=lambda x: {
        "Fx": "X축 힘 (Fx)",
        "Fy": "Y축 힘 (Fy)",
        "Fz": "Z축 힘 (Fz)"
    }[x]
)

force_time = st.slider(
    "힘 측정 시점",
    min_value=1,
    max_value=15,
    value=1
)

force_column = f"{selected_force}{force_time}"

fig = px.histogram(
    data,
    x=force_column,
    color="Class",
    barmode="overlay",
    opacity=0.7,
    marginal="box",
    title=f"{force_column} 측정값 분포"
)

fig.update_layout(
    xaxis_title="힘 측정값",
    yaxis_title="데이터 개수"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    f"{force_column}의 분포를 통해 해당 시점에서 "
    "힘 측정값이 어떤 범위에 집중되는지 확인할 수 있습니다."
)


# =====================================
# 3. 토크 측정값 분포
# =====================================

st.divider()

st.subheader("3. 토크(Torque) 측정값 분포")

torque_sensors = ["Tx", "Ty", "Tz"]

selected_torque = st.selectbox(
    "분석할 토크 센서 선택",
    torque_sensors,
    format_func=lambda x: {
        "Tx": "X축 토크 (Tx)",
        "Ty": "Y축 토크 (Ty)",
        "Tz": "Z축 토크 (Tz)"
    }[x]
)

torque_time = st.slider(
    "토크 측정 시점",
    min_value=1,
    max_value=15,
    value=1
)

torque_column = f"{selected_torque}{torque_time}"

fig = px.histogram(
    data,
    x=torque_column,
    color="Class",
    barmode="overlay",
    opacity=0.7,
    marginal="box",
    title=f"{torque_column} 측정값 분포"
)

fig.update_layout(
    xaxis_title="토크 측정값",
    yaxis_title="데이터 개수"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    f"{torque_column}의 분포를 통해 해당 시점의 "
    "토크 측정값이 Class별로 어떤 차이를 보이는지 살펴볼 수 있습니다."
)


# =====================================
# 4. Class별 센서 측정값 비교
# =====================================

st.divider()

st.subheader("4. Class별 센서 측정값 비교")

selected_sensor = st.selectbox(
    "비교할 센서 선택",
    SENSORS,
    format_func=lambda x: {
        "Fx": "X축 힘",
        "Fy": "Y축 힘",
        "Fz": "Z축 힘",
        "Tx": "X축 토크",
        "Ty": "Y축 토크",
        "Tz": "Z축 토크"
    }[x],
    key="comparison_sensor"
)

selected_time = st.slider(
    "비교할 측정 시점",
    1,
    15,
    1,
    key="comparison_time"
)

comparison_column = f"{selected_sensor}{selected_time}"

fig = px.box(
    data,
    x="Class",
    y=comparison_column,
    color="Class",
    points="all",
    title=f"Class별 {comparison_column} 측정값 비교"
)

fig.update_layout(
    xaxis_title="Class",
    yaxis_title="센서 측정값"
)

st.plotly_chart(fig, use_container_width=True)

# Class별 평균값
mean_table = (
    data.groupby("Class")[comparison_column]
    .agg(["mean", "median", "std"])
    .reset_index()
)

mean_table.columns = [
    "Class",
    "평균",
    "중앙값",
    "표준편차"
]

st.write("Class별 센서 측정값 통계")

st.dataframe(
    mean_table.round(3),
    use_container_width=True,
    hide_index=True
)

st.info(
    "상자 그림을 통해 Class별 중앙값과 측정값의 분포 범위를 "
    "비교할 수 있습니다. 평균값이나 분포 차이가 크더라도 "
    "그 자체만으로 원인이나 오류 유형을 확정할 수는 없습니다."
)


# =====================================
# 5. 측정 시점별 센서 변화
# =====================================

st.divider()

st.subheader("5. 측정 시점에 따른 센서값 변화")

selected_class = st.selectbox(
    "분석할 Class 선택",
    classes,
    format_func=lambda x: f"Class {x}",
    key="time_class"
)

selected_sensor2 = st.selectbox(
    "분석할 센서 선택",
    SENSORS,
    key="time_sensor"
)

class_data = data[
    data["Class"] == selected_class
]

time_means = []

for i in range(1, 16):

    column = f"{selected_sensor2}{i}"

    time_means.append({
        "측정 시점": i,
        "평균 측정값": class_data[column].mean()
    })

time_df = pd.DataFrame(time_means)

fig = px.line(
    time_df,
    x="측정 시점",
    y="평균 측정값",
    markers=True,
    title=f"Class {selected_class}의 {selected_sensor2} 변화"
)

fig.update_layout(
    xaxis_title="측정 시점",
    yaxis_title="평균 측정값"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "15개 측정 시점의 평균값을 연결하여 "
    "시간의 흐름에 따른 센서값 변화를 살펴볼 수 있습니다."
)
