
import streamlit as st
import pandas as pd
import numpy as np
import requests

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# =====================================
# 1. 데이터 주소 및 센서 속성
# =====================================

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/robotfailure-mld/lp1.data"
)

SENSORS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

# 센서별 속성 이름
FEATURE_COLUMNS = [
    f"{sensor}{i}"
    for sensor in SENSORS
    for i in range(1, 16)
]

ALL_COLUMNS = ["Class"] + FEATURE_COLUMNS


# =====================================
# 2. 데이터 불러오기
# =====================================

@st.cache_data
def load_data():

    try:
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()

        # 파일 전체를 공백 기준으로 분리
        # 실제 LP1 파일은 Class 1개 + 센서값 90개가
        # 여러 줄에 나누어 저장되어 있음
        tokens = response.text.split()

        # 한 사례는 총 91개의 값으로 구성
        if len(tokens) % 91 != 0:
            raise ValueError(
                f"데이터 값의 개수가 91의 배수가 아닙니다. "
                f"전체 값 개수: {len(tokens)}"
            )

        # 91개씩 묶어서 하나의 사례로 구성
        records = [
            tokens[i:i + 91]
            for i in range(0, len(tokens), 91)
        ]

        # 원본 데이터의 열 순서
        # Class, Fx1, Fy1, Fz1, Tx1, Ty1, Tz1, ...
        original_columns = ["Class"]

        for i in range(1, 16):
            for sensor in SENSORS:
                original_columns.append(f"{sensor}{i}")

        raw = pd.DataFrame(
            records,
            columns=original_columns
        )

        # 센서별 순서로 열 재배열
        df = raw[ALL_COLUMNS].copy()

        # 센서 측정값을 숫자형으로 변환
        for col in FEATURE_COLUMNS:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # Class 결측치가 있는 행 제거
        df = df.dropna(subset=["Class"]).copy()

        if df.empty:
            raise ValueError("사용할 수 있는 데이터가 없습니다.")

        return df

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            "데이터 다운로드에 실패했습니다. "
            "인터넷 연결과 UCI 데이터 주소를 확인해 주세요."
        ) from e

    except Exception as e:
        raise RuntimeError(
            f"데이터 처리 중 오류가 발생했습니다: {e}"
        ) from e


# =====================================
# 3. 데이터 전처리
# =====================================

def preprocess_data(df):

    data = df.copy()

    # 센서값을 숫자형으로 변환
    for col in FEATURE_COLUMNS:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # 센서값 또는 Class가 없는 행 제거
    data = data.dropna(
        subset=FEATURE_COLUMNS + ["Class"]
    )

    return data


# =====================================
# 4. 센서 속성 설명
# =====================================

def get_feature_description():

    descriptions = []

    sensor_names = {
        "Fx": "X축 힘 측정값",
        "Fy": "Y축 힘 측정값",
        "Fz": "Z축 힘 측정값",
        "Tx": "X축 토크 측정값",
        "Ty": "Y축 토크 측정값",
        "Tz": "Z축 토크 측정값"
    }

    for sensor in SENSORS:
        for i in range(1, 16):
            descriptions.append({
                "속성": f"{sensor}{i}",
                "설명": sensor_names[sensor],
                "측정 시점": f"{i}번째 측정"
            })

    return pd.DataFrame(descriptions)


# =====================================
# 5. 머신러닝 모델 생성
# =====================================

def create_model(model_name, random_state=42):

    if model_name == "의사결정나무":
        return DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced"
        )

    elif model_name == "랜덤 포레스트":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced"
        )

    else:
        raise ValueError("올바르지 않은 모델입니다.")


# =====================================
# 6. 세션 상태 초기화
# =====================================

def initialize_session():

    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None

    if "selected_features" not in st.session_state:
        st.session_state.selected_features = []

    if "model_classes" not in st.session_state:
        st.session_state.model_classes = None

    if "model_metrics" not in st.session_state:
        st.session_state.model_metrics = None

    if "test_results" not in st.session_state:
        st.session_state.test_results = None

    if "model_name" not in st.session_state:
        st.session_state.model_name = None

    if "training_data" not in st.session_state:
        st.session_state.training_data = None


# =====================================
# 7. 데이터 불러오기 및 세션 초기화
# =====================================

def prepare_app():

    initialize_session()

    try:
        df = load_data()
        return df

    except Exception as e:
        st.error(str(e))

        st.info(
            "UCI 공식 데이터셋의 LP1 파일을 "
            "다운로드하고 데이터 형식을 확인해 주세요."
        )

        return None
