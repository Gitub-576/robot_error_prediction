
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    prepare_app,
    preprocess_data,
    FEATURE_COLUMNS,
    SENSORS,
    initialize_session
)


# =====================================
# 페이지 설정
# =====================================

st.set_page_config(
    page_title="오류 유형 예측",
    page_icon="🤖",
    layout="wide"
)

initialize_session()

st.title("🤖 오류 유형 예측")

st.markdown("""
새로운 로봇의 힘과 토크 센서 측정값을 입력하고
학습된 머신러닝 모델을 이용하여 Class를 예측합니다.
""")

df = prepare_app()

if df is None:
    st.stop()

data = preprocess_data(df)


# =====================================
# 모델 학습 여부 확인
# =====================================

st.divider()

st.subheader("1. 학습된 모델 확인")

model = st.session_state.trained_model

if model is None:

    st.warning("""
    아직 학습된 모델이 없습니다.

    먼저 사이드바에서 '머신러닝 분류 모델' 페이지로 이동한 후
    모델 학습을 완료해 주세요.
    """)

    st.stop()

st.success(
    f"현재 사용 중인 모델: {st.session_state.model_name}"
)

selected_features = st.session_state.selected_features

st.write(
    f"현재 모델이 사용하는 독립 변수: "
    f"{len(selected_features)}개"
)


# =====================================
# 입력 데이터 준비
# =====================================

st.divider()

st.subheader("2. 센서 측정값 입력")

st.markdown("""
90개의 센서 측정값을 입력해 주세요.

아래 입력창의 초기값은 데이터셋에서 계산한 중앙값을 사용합니다.
실제 측정값이 아니므로, 결과 시연용 예시값으로만 사용하세요.
실제 로봇의 측정값을 확보한 경우 해당 값으로 변경하면 됩니다.
""")

st.warning(
    "주의: 현재 입력된 기본값은 실제 로봇에서 측정한 데이터가 아닙니다. "
    "기본값으로 얻은 예측 결과를 실제 로봇의 오류 진단 결과로 "
    "해석해서는 안 됩니다."
)


# =====================================
# 센서값 입력
# =====================================

median_values = data[FEATURE_COLUMNS].median()

input_values = {}

for sensor in SENSORS:

    with st.expander(
        f"{sensor} 센서 측정값 입력",
        expanded=(sensor == "Fx")
    ):

        st.caption(
            f"{sensor} 센서의 1~15번째 측정값을 입력합니다."
        )

        cols = st.columns(3)

        for i in range(1, 16):

            feature = f"{sensor}{i}"

            with cols[(i - 1) % 3]:

                value = st.number_input(
                    feature,
                    value=float(median_values[feature]),
                    step=1.0,
                    format="%.3f",
                    key=f"input_{feature}"
                )

                input_values[feature] = value


# =====================================
# 입력 데이터 미리보기
# =====================================

st.divider()

st.subheader("3. 입력값 확인")

input_df = pd.DataFrame(
    [input_values],
    columns=FEATURE_COLUMNS
)

st.dataframe(
    input_df,
    use_container_width=True
)


# =====================================
# 예측 실행
# =====================================

st.divider()

st.subheader("4. 오류 유형 예측")

if st.button(
    "🔍 Class 예측하기",
    type="primary",
    use_container_width=True
):

    try:

        # 학습 당시 사용한 독립 변수만 선택
        X_new = input_df[
            selected_features
        ]

        # Class 예측
        prediction = model.predict(X_new)[0]

        st.success("예측이 완료되었습니다!")

        st.markdown("### 예측 결과")

        st.metric(
            label="예측된 Class",
            value=str(prediction)
        )

        st.caption(
            "예측된 Class 값은 원본 데이터의 라벨을 그대로 사용합니다."
        )

        # 예측 확률
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X_new)[0]

            probability_df = pd.DataFrame({
                "Class": model.classes_,
                "예측 확률": probabilities
            })

            probability_df["예측 확률(%)"] = (
                probability_df["예측 확률"] * 100
            ).round(2)

            st.divider()

            st.subheader("5. Class별 예측 확률")

            fig = px.bar(
                probability_df,
                x="Class",
                y="예측 확률",
                color="Class",
                text=probability_df["예측 확률(%)"].apply(
                    lambda x: f"{x:.2f}%"
                ),
                range_y=[0, 1],
                title="각 Class에 속할 예측 확률"
            )

            fig.update_layout(
                xaxis_title="Class",
                yaxis_title="예측 확률"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                probability_df[
                    ["Class", "예측 확률(%)"]
                ],
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "예측 확률은 학습된 모델이 계산한 분류 점수입니다. "
                "실제 오류가 발생할 확률이나 안전성의 확률을 "
                "직접 의미하지는 않습니다."
            )

        # 입력값과 예측값 저장
        st.session_state.last_prediction = prediction

    except Exception as e:

        st.error(
            f"예측 중 오류가 발생했습니다: {e}"
        )


# =====================================
# 예측 안내
# =====================================

st.divider()

st.subheader("📌 예측 결과 해석")

st.markdown("""
- Class는 LP1 원본 데이터의 분류값입니다.
- 예측 결과는 학습된 모델이 입력된 센서 패턴을 바탕으로 분류한 값입니다.
- 학습 데이터에 존재하지 않는 새로운 유형을 자동으로 찾아내는 기능은 아닙니다.
- 실제 산업용 로봇에 적용하기 위해서는 별도의 검증과 안전성 평가가 필요합니다.
""")
