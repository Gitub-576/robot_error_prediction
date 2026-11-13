
import streamlit as st
import plotly.express as px

from utils import (
    prepare_app,
    preprocess_data,
    FEATURE_COLUMNS,
    get_feature_description
)


# =====================================
# 페이지 설정
# =====================================

st.set_page_config(
    page_title="산업용 로봇 오류 예측 실습실",
    page_icon="🤖",
    layout="wide"
)


# =====================================
# 데이터 불러오기
# =====================================

df = prepare_app()

if df is None:
    st.stop()

data = preprocess_data(df)

classes = sorted(data["Class"].unique(), key=str)

total_rows = len(data)
total_columns = data.shape[1]
class_count = data["Class"].nunique()
missing_count = int(df.isnull().sum().sum())
duplicate_count = int(data.duplicated().sum())


# =====================================
# 제목
# =====================================

st.title("🤖 산업용 로봇 오류 예측 실습실")

st.markdown("""
산업용 로봇에서 측정한 **힘(Force)과 토크(Torque)** 데이터를
머신러닝으로 분석하여 로봇의 작동 결과와 오류 유형을 분류하는 프로그램입니다.

UCI Machine Learning Repository의 Robot Execution Failures
LP1 데이터를 사용합니다.
""")

st.divider()


# =====================================
# 데이터 통계 카드
# =====================================

st.subheader("📊 데이터셋 기본 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "전체 데이터 개수",
        f"{total_rows}개"
    )

with col2:
    st.metric(
        "전체 속성 개수",
        f"{total_columns}개"
    )

with col3:
    st.metric(
        "Class 종류 개수",
        f"{class_count}개"
    )

with col4:
    st.metric(
        "결측치 개수",
        f"{missing_count}개"
    )

st.caption(
    f"중복 데이터: {duplicate_count}개 | "
    "원본 행을 임의로 삭제하지 않고 검사 결과를 표시합니다."
)

st.divider()


# =====================================
# 데이터 미리보기
# =====================================

st.subheader("📋 데이터 미리보기")

st.write("LP1 데이터의 처음 5개 행입니다.")

st.dataframe(
    data.head(),
    use_container_width=True
)


# =====================================
# 속성 설명
# =====================================

st.divider()

st.subheader("📖 데이터 속성 설명")

st.markdown("""
각 행은 로봇의 작동 과정에서 측정된 하나의 사례입니다.

- Fx, Fy, Fz: X, Y, Z축 방향의 힘
- Tx, Ty, Tz: X, Y, Z축 방향의 토크
- 1~15: 측정이 이루어진 순서
- Class: 해당 사례의 실제 분류 결과
""")

description = get_feature_description()

st.dataframe(
    description,
    use_container_width=True,
    hide_index=True
)

st.info(
    "총 90개의 센서 측정 속성과 1개의 Class 속성으로 구성됩니다."
)


# =====================================
# Class별 데이터 분포
# =====================================

st.divider()

st.subheader("📈 Class별 데이터 분포")

class_counts = (
    data["Class"]
    .value_counts()
    .sort_index(key=lambda x: x.astype(str))
    .reset_index()
)

class_counts.columns = ["Class", "데이터 개수"]

class_counts["비율(%)"] = (
    class_counts["데이터 개수"] / total_rows * 100
).round(2)

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        class_counts,
        x="Class",
        y="데이터 개수",
        color="Class",
        text="데이터 개수",
        title="Class별 데이터 개수"
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Class",
        yaxis_title="데이터 개수"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.pie(
        class_counts,
        names="Class",
        values="데이터 개수",
        hole=0.4,
        title="Class별 데이터 비율"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.dataframe(
    class_counts,
    use_container_width=True,
    hide_index=True
)


# =====================================
# 데이터 출처
# =====================================

st.divider()

st.subheader("🔗 데이터 출처")

st.markdown(
    "[UCI Machine Learning Repository - Robot Execution Failures]"
    "(https://archive.ics.uci.edu/dataset/138/robot+execution+failures)"
)

st.caption(
    "Lopes, L. & Camarinha-Matos, L. (1998). "
    "Robot Execution Failures. UCI Machine Learning Repository."
)
