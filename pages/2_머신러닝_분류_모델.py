
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from utils import (
    prepare_app,
    preprocess_data,
    FEATURE_COLUMNS,
    SENSORS,
    create_model,
    initialize_session
)


# =====================================
# 페이지 설정
# =====================================

st.set_page_config(
    page_title="머신러닝 분류 모델",
    page_icon="🧠",
    layout="wide"
)

initialize_session()

st.title("🧠 머신러닝 분류 모델")

st.markdown("""
로봇의 힘과 토크 측정값을 학습하여
작동 결과 및 오류 유형을 분류하는 머신러닝 모델을 만듭니다.
""")

df = prepare_app()

if df is None:
    st.stop()

data = preprocess_data(df)

classes = sorted(data["Class"].unique(), key=str)


# =====================================
# 데이터 정보
# =====================================

st.divider()

st.subheader("1. 학습 데이터 정보")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(data)}개")
col2.metric("입력 속성", f"{len(FEATURE_COLUMNS)}개")
col3.metric("Class 종류", f"{len(classes)}개")

st.caption(
    "Class별 데이터 수가 적으므로 학습 및 평가 결과가 "
    "데이터 분할 방식에 따라 달라질 수 있습니다."
)


# =====================================
# 모델 설정
# =====================================

st.divider()

st.subheader("2. 모델 설정")

model_name = st.selectbox(
    "머신러닝 알고리즘 선택",
    ["의사결정나무", "랜덤 포레스트"]
)

test_ratio = st.slider(
    "테스트 데이터 비율",
    min_value=0.2,
    max_value=0.4,
    value=0.25,
    step=0.05,
    format="%.2f"
)


# =====================================
# 독립 변수 선택
# =====================================

st.divider()

st.subheader("3. 독립 변수 선택")

st.markdown("""
모델이 Class를 예측할 때 사용할 센서 측정값을 선택합니다.

기본적으로 90개 속성이 모두 선택되어 있습니다.
""")

if "feature_selection" not in st.session_state:
    st.session_state.feature_selection = FEATURE_COLUMNS.copy()

col1, col2 = st.columns(2)

with col1:

    if st.button("90개 속성 전체 선택", use_container_width=True):
        st.session_state.feature_selection = FEATURE_COLUMNS.copy()
        st.rerun()

with col2:

    if st.button("전체 선택 해제", use_container_width=True):
        st.session_state.feature_selection = []
        st.rerun()


# 센서 종류별 선택
for sensor in SENSORS:

    sensor_columns = [
        f"{sensor}{i}" for i in range(1, 16)
    ]

    with st.expander(f"{sensor} 센서 속성 선택"):

        selected_sensor_columns = st.multiselect(
            f"{sensor} 측정 속성",
            sensor_columns,
            default=[
                col for col in sensor_columns
                if col in st.session_state.feature_selection
            ],
            key=f"select_{sensor}"
        )

        # 현재 선택 결과를 세션 상태에 반영
        current = set(st.session_state.feature_selection)
        current.difference_update(sensor_columns)
        current.update(selected_sensor_columns)

        st.session_state.feature_selection = [
            col for col in FEATURE_COLUMNS if col in current
        ]

selected_features = st.session_state.feature_selection

st.metric(
    "현재 선택된 독립 변수",
    f"{len(selected_features)}개"
)

if len(selected_features) == 0:
    st.warning("최소 1개 이상의 독립 변수를 선택해 주세요.")


# =====================================
# Class별 표본 수 확인
# =====================================

st.divider()

st.subheader("4. Class별 데이터 개수 확인")

class_counts = data["Class"].value_counts().sort_index(
    key=lambda x: x.astype(str)
)

st.dataframe(
    class_counts.rename("데이터 개수"),
    use_container_width=True
)

min_class_count = int(class_counts.min())

if min_class_count < 2:
    st.error(
        "표본 수가 2개 미만인 Class가 있습니다. "
        "Stratified 분할을 수행할 수 없습니다."
    )


# =====================================
# 모델 학습
# =====================================

st.divider()

st.subheader("5. 모델 학습")

st.markdown("""
아래 버튼을 누르면 선택한 독립 변수와 알고리즘을 이용하여
모델을 학습하고 테스트 데이터로 성능을 평가합니다.
""")

if st.button(
    "🚀 머신러닝 모델 학습하기",
    type="primary",
    use_container_width=True
):

    if len(selected_features) == 0:

        st.error("독립 변수를 하나 이상 선택해 주세요.")

    elif min_class_count < 2:

        st.error("Class별 표본 수가 부족합니다.")

    else:

        try:

            X = data[selected_features]
            y = data["Class"]

            # Class가 학습 데이터와 테스트 데이터에
            # 모두 포함될 수 있는지 검사
            n_classes = y.nunique()
            n_samples = len(y)

            n_test = int(np.ceil(n_samples * test_ratio))
            n_train = n_samples - n_test

            if n_test < n_classes or n_train < n_classes:
                raise ValueError(
                    "테스트 또는 학습 데이터가 부족합니다. "
                    "테스트 비율을 조절해 주세요."
                )

            if min_class_count < 2:
                raise ValueError(
                    "각 Class에 최소 2개 이상의 표본이 필요합니다."
                )

            # 학습 데이터와 테스트 데이터 분리
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_ratio,
                random_state=42,
                stratify=y
            )

            # 모델 생성
            model = create_model(model_name)

            # 모델 학습
            with st.spinner("모델을 학습하고 있습니다..."):
                model.fit(X_train, y_train)

            # 예측
            y_pred = model.predict(X_test)

            # 평가 지표
            accuracy = accuracy_score(y_test, y_pred)

            precision = precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            # 모델을 세션에 저장
            st.session_state.trained_model = model

            st.session_state.selected_features = selected_features.copy()

            st.session_state.model_classes = model.classes_

            st.session_state.model_name = model_name

            st.session_state.model_metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1
            }

            # 실제값과 예측값 저장
            results = pd.DataFrame({
                "실제 Class": y_test.values,
                "예측 Class": y_pred
            })

            results["예측 결과"] = np.where(
                results["실제 Class"] == results["예측 Class"],
                "정답",
                "오답"
            )

            st.session_state.test_results = results

            # 예측에 사용할 대표값 계산
            st.session_state.training_data = X_train.copy()

            st.success("모델 학습 및 평가가 완료되었습니다!")

        except Exception as e:

            st.session_state.trained_model = None
            st.session_state.model_metrics = None
            st.session_state.test_results = None
            st.session_state.training_data = None

            st.error(f"모델 학습 중 오류가 발생했습니다: {e}")


# =====================================
# 학습 결과 출력
# =====================================

if st.session_state.trained_model is not None:

    st.divider()

    st.subheader("6. 모델 성능 평가 결과")

    metrics = st.session_state.model_metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "정확도",
        f"{metrics['Accuracy'] * 100:.2f}%"
    )

    col2.metric(
        "정밀도",
        f"{metrics['Precision'] * 100:.2f}%"
    )

    col3.metric(
        "재현율",
        f"{metrics['Recall'] * 100:.2f}%"
    )

    col4.metric(
        "F1-score",
        f"{metrics['F1-score'] * 100:.2f}%"
    )

    st.caption(
        "정밀도, 재현율, F1-score는 Class별 성능을 "
        "표본 수로 가중 평균한 값입니다."
    )


    # =================================
    # 성능 그래프
    # =================================

    st.subheader("평가 지표 시각화")

    metric_df = pd.DataFrame({
        "평가 지표": list(metrics.keys()),
        "점수": list(metrics.values())
    })

    fig = px.bar(
        metric_df,
        x="평가 지표",
        y="점수",
        color="평가 지표",
        text=metric_df["점수"].apply(
            lambda x: f"{x * 100:.1f}%"
        ),
        range_y=[0, 1],
        title="머신러닝 모델 성능"
    )

    fig.update_layout(
        showlegend=False,
        yaxis_title="점수"
    )

    st.plotly_chart(fig, use_container_width=True)


    # =================================
    # 혼동 행렬
    # =================================

    st.divider()

    st.subheader("7. 혼동 행렬 (Confusion Matrix)")

    results = st.session_state.test_results

    labels = list(st.session_state.model_classes)

    cm = confusion_matrix(
        results["실제 Class"],
        results["예측 Class"],
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"실제 {x}" for x in labels],
        columns=[f"예측 {x}" for x in labels]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        title="실제 Class와 예측 Class 비교"
    )

    fig.update_layout(
        xaxis_title="예측한 Class",
        yaxis_title="실제 Class"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "혼동 행렬의 대각선에 있는 숫자는 정확하게 예측한 개수이며, "
        "대각선 밖의 숫자는 다른 Class로 잘못 예측한 개수입니다."
    )


    # =================================
    # 실제값과 예측값 비교
    # =================================

    st.divider()

    st.subheader("8. 실제 Class와 예측 Class 비교")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    correct = (
        results["실제 Class"] == results["예측 Class"]
    ).sum()

    total = len(results)

    st.write(
        f"테스트 데이터 {total}개 중 "
        f"{correct}개를 정확하게 분류했습니다."
    )


    # =================================
    # Class별 상세 평가
    # =================================

    st.divider()

    st.subheader("9. Class별 상세 성능")

    report = classification_report(
        results["실제 Class"],
        results["예측 Class"],
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    report_rows = []

    for label in labels:

        values = report[str(label)]

        report_rows.append({
            "Class": label,
            "정밀도": values["precision"],
            "재현율": values["recall"],
            "F1-score": values["f1-score"],
            "데이터 개수": int(values["support"])
        })

    report_df = pd.DataFrame(report_rows)

    st.dataframe(
        report_df.round(3),
        use_container_width=True,
        hide_index=True
    )


    # =================================
    # 변수 중요도
    # =================================

    st.divider()

    st.subheader("10. 센서 속성 중요도")

    model = st.session_state.trained_model

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "센서 속성": st.session_state.selected_features,
            "중요도": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            "중요도",
            ascending=False
        )

        fig = px.bar(
            importance_df.head(15).sort_values(
                "중요도",
                ascending=True
            ),
            x="중요도",
            y="센서 속성",
            orientation="h",
            title="중요도가 높은 센서 속성 TOP 15"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "변수 중요도는 해당 모델의 분류 과정에서 각 속성이 "
            "기여한 정도를 나타냅니다. 인과관계를 의미하지는 않습니다."
        )


    # =================================
    # 주의사항
    # =================================

    st.divider()

    st.warning("""
    **모델 평가 시 주의사항**

    - LP1 데이터는 전체 88개로 표본 수가 적습니다.
    - 데이터 분할 방식에 따라 평가 점수가 달라질 수 있습니다.
    - 테스트 데이터의 개수가 적어 Class별 평가 결과가 불안정할 수 있습니다.
    - 정확도가 높더라도 새로운 로봇이나 다른 작업 환경에서
      동일한 성능이 보장되는 것은 아닙니다.
    - 이 모델은 실제 산업용 로봇의 안전 제어를 대신하는 프로그램이 아닙니다.
    """)
