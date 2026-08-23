import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Obesity Level Prediction",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL + PREPROCESSING OBJECTS
# ============================================================

@st.cache_resource
def load_objects():

    base_dir = Path(__file__).resolve().parent

    # Use the best model saved from your notebook
    model_path = base_dir / "best_obesity_model.pkl"

    # Fallback if you are still using the old filename
    if not model_path.exists():
        model_path = base_dir / "obesity_model.pkl"

    model = joblib.load(model_path)

    scaler = joblib.load(
        base_dir / "scaler.pkl"
    )

    label_encoder = joblib.load(
        base_dir / "label_encoder.pkl"
    )

    feature_columns = joblib.load(
        base_dir / "feature_columns.pkl"
    )

    return (
        model,
        scaler,
        label_encoder,
        feature_columns
    )


try:

    (
        model,
        scaler,
        label_encoder,
        feature_columns
    ) = load_objects()

except Exception as e:

    st.error(
        "Unable to load the trained model or preprocessing files."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("📊 Obesity Level Prediction System")

st.write(
    """
    This application predicts a person's obesity level
    based on physical characteristics, eating habits,
    and lifestyle information.
    """
)

st.divider()


# ============================================================
# USER INPUT
# ============================================================

st.subheader("👤 Personal Information")


# ============================================================
# BASIC INFORMATION
# ============================================================

gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)


age = st.number_input(
    "Age",
    min_value=10.0,
    max_value=100.0,
    value=25.0,
    step=1.0
)


height = st.number_input(
    "Height (metres)",
    min_value=1.20,
    max_value=2.20,
    value=1.70,
    step=0.01
)


weight = st.number_input(
    "Weight (kg)",
    min_value=30.0,
    max_value=250.0,
    value=70.0,
    step=1.0
)


# ============================================================
# BMI DISPLAY
# ============================================================

bmi = weight / (height ** 2)

st.metric(
    "BMI",
    f"{bmi:.2f}"
)


# ============================================================
# FAMILY HISTORY
# ============================================================

family_history = st.selectbox(
    "Family history of overweight?",
    [
        "yes",
        "no"
    ]
)


# ============================================================
# HIGH-CALORIE FOOD
# FAVC
# ============================================================

favc = st.selectbox(
    "Do you frequently consume high-calorie food?",
    [
        "yes",
        "no"
    ]
)


# ============================================================
# VEGETABLE CONSUMPTION
# FCVC
# ============================================================

fcvc = st.slider(
    "Vegetable consumption frequency",
    min_value=1.0,
    max_value=3.0,
    value=2.0,
    step=0.1,
    help=(
        "1 = Low consumption, "
        "2 = Moderate consumption, "
        "3 = High consumption"
    )
)


# ============================================================
# NUMBER OF MAIN MEALS
# NCP
# ============================================================

ncp = st.slider(
    "Number of main meals per day",
    min_value=1.0,
    max_value=4.0,
    value=3.0,
    step=0.1
)


# ============================================================
# FOOD BETWEEN MEALS
# CAEC
# ============================================================

caec = st.selectbox(
    "Food consumption between meals",
    [
        "no",
        "Sometimes",
        "Frequently",
        "Always"
    ]
)


# ============================================================
# SMOKING
# SMOKE
# ============================================================

smoke = st.selectbox(
    "Do you smoke?",
    [
        "yes",
        "no"
    ]
)


# ============================================================
# WATER CONSUMPTION
# CH2O
# ============================================================

ch2o = st.slider(
    "Daily water consumption",
    min_value=1.0,
    max_value=3.0,
    value=2.0,
    step=0.1,
    help=(
        "1 = Low, "
        "2 = Moderate, "
        "3 = High"
    )
)


# ============================================================
# CALORIE MONITORING
# SCC
# ============================================================

scc = st.selectbox(
    "Do you monitor your calorie intake?",
    [
        "yes",
        "no"
    ]
)


# ============================================================
# PHYSICAL ACTIVITY
# FAF
# ============================================================

faf = st.slider(
    "Physical activity frequency",
    min_value=0.0,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help=(
        "Higher values indicate "
        "more frequent physical activity."
    )
)


# ============================================================
# TECHNOLOGY USE
# TUE
# ============================================================

tue = st.slider(
    "Time using technology devices",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.1
)


# ============================================================
# ALCOHOL CONSUMPTION
# CALC
# ============================================================

calc = st.selectbox(
    "Alcohol consumption",
    [
        "no",
        "Sometimes",
        "Frequently",
        "Always"
    ]
)


# ============================================================
# TRANSPORTATION
# MTRANS
# ============================================================

mtrans = st.selectbox(
    "Main mode of transportation",
    [
        "Automobile",
        "Bike",
        "Motorbike",
        "Public_Transportation",
        "Walking"
    ]
)


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 Predict Obesity Level",
    use_container_width=True
):

    # ========================================================
    # CREATE RAW INPUT DATAFRAME
    # Must use the SAME column names as training dataset
    # ========================================================

    input_data = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Age": age,
                "Height": height,
                "Weight": weight,

                "family_history_with_overweight":
                    family_history,

                "FAVC": favc,
                "FCVC": fcvc,
                "NCP": ncp,
                "CAEC": caec,
                "SMOKE": smoke,
                "CH2O": ch2o,
                "SCC": scc,
                "FAF": faf,
                "TUE": tue,
                "CALC": calc,
                "MTRANS": mtrans
            }
        ]
    )


    # ========================================================
    # ONE-HOT ENCODING
    # Same method used during model training
    # ========================================================

    input_encoded = pd.get_dummies(
        input_data,
        drop_first=True
    )


    # ========================================================
    # MATCH TRAINING FEATURE COLUMNS
    # ========================================================

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # ========================================================
    # SCALE INPUT
    # ========================================================

    input_scaled = scaler.transform(
        input_encoded
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(
        input_scaled
    )


    # ========================================================
    # DECODE PREDICTION
    # ========================================================

    predicted_class = label_encoder.inverse_transform(
        prediction.astype(int)
    )[0]


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.success(
        f"Predicted Obesity Level: "
        f"**{predicted_class}**"
    )


    # ========================================================
    # PREDICTION CONFIDENCE
    # Only works for models supporting predict_proba()
    # ========================================================

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            input_scaled
        )[0]

        confidence = probabilities.max() * 100

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


        # ====================================================
        # SHOW ALL CLASS PROBABILITIES
        # ====================================================

        class_labels = label_encoder.inverse_transform(
            model.classes_.astype(int)
        )

        probability_df = pd.DataFrame(
            {
                "Obesity Level":
                    class_labels,

                "Probability (%)":
                    probabilities * 100
            }
        )

        probability_df = probability_df.sort_values(
            by="Probability (%)",
            ascending=False
        ).reset_index(drop=True)


        probability_df[
            "Probability (%)"
        ] = probability_df[
            "Probability (%)"
        ].round(2)


        st.subheader(
            "Prediction Probability"
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Machine Learning Obesity Classification Prototype"
)

st.write("Loaded Model:")
st.write(type(model).__name__)
