
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO

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

# 사용자가 요청한 센서별 열 이름
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

        # 원본 파일은 헤더 없이 공백으로 구분되어 있음
        raw = pd.read_csv(
            StringIO(response.text),
            sep=r"\s+",
            header=None,
            engine="python"
        )

        # 실제 열 개수 검사
        if raw.shape[1] != 91:
            raise ValueError(
                f"예상한 열 개수는 91개이지만 "
                f"실제 열 개수는 {raw.shape[1]}개입니다."
            )

        # 원본 순서는 시점별로 6개 센서가 반복됨
        original_columns = ["Class"]

        for i in range(1, 16):
            for sensor in SENSORS:
                original_columns.append(f"{sensor}{i}")

        raw.columns = original_columns

        # 센서별 순서로 열 재배열
        df = raw[ALL_COLUMNS].copy()

        # Class는 원본 값 그대로 보존
        # 나머지 센서값은 숫자형으로 변환
        for col in FEATURE_COLUMNS:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # Class 결측치가 있는 행 제거
        df = df.dropna(subset=["Class"]).copy()

        if df.empty:
            raise ValueError("사용할 수 있는 데이터가 없습니다.")

        # 센서 데이터에 결측치가 있는지 확인
        # 임의로 값을 채우지 않고 원본 상태를 유지
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
            "다운로드할 수 있는지 확인해 주세요."
        )
        return None
