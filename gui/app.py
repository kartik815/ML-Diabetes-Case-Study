import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Diabetes ML Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.05rem;
            opacity: 0.75;
            margin-bottom: 1.5rem;
        }

        .result-box {
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            text-align: center;
            margin-top: 1rem;
        }

        .score {
            font-size: 3rem;
            font-weight: 700;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid rgba(128,128,128,0.25);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# FEATURE INFORMATION
# ============================================================

FEATURE_COLUMNS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]


# ============================================================
# MODEL NAMES
# ============================================================

MODEL_DISPLAY_NAMES = {
    "random_forest_random_search.pkl": "Random Forest Regressor",
    "random_forest.pkl": "Random Forest Regressor",
    "decision_tree.pkl": "Decision Tree Regressor",
    "gradient_boosting.pkl": "Gradient Boosting Regressor",
    "svr.pkl": "Support Vector Regressor",
    "knn.pkl": "K-Nearest Neighbors Regressor",
}


# ============================================================
# KNOWN RESULTS
# These are your current results.
# Add more models here as you finish them.
# ============================================================

MODEL_RESULTS = {
    "Random Forest Regressor": {
        "RMSE": 0.309547,
        "MAE": 0.196013,
        "MSE": 0.095820,
        "R²": 0.194054,
    },

    "Decision Tree Regressor": {
        "RMSE": 0.312087,
        "MAE": 0.195148,
        "MSE": 0.097398,
        "R²": 0.180775,
    },
}


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    """
    Load all .pkl models from the models directory.

    Streamlit caches the loaded objects so the models are not
    repeatedly loaded every time the user interacts with the UI.
    """

    models = {}

    if not MODEL_DIR.exists():
        return models

    for model_file in MODEL_DIR.glob("*.pkl"):
        try:
            model = joblib.load(model_file)
            display_name = MODEL_DISPLAY_NAMES.get(
                model_file.name,
                model_file.stem.replace("_", " ").title()
            )

            models[display_name] = model

        except Exception as error:
            st.warning(
                f"Could not load {model_file.name}: {error}"
            )

    return models


models = load_models()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🩺 Diabetes ML")

page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Model Comparison",
        "About",
    ],
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Diabetes ML Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Interactive dashboard for the CDC Diabetes Health Indicators
        machine learning project.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if page == "Prediction":

    st.header("Patient Prediction")

    if not models:
        st.error(
            "No trained models were found in the models/ directory."
        )
        st.info(
            "Place your .pkl model files inside the models/ folder."
        )
        st.stop()

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    selected_model_name = st.selectbox(
        "Select Model",
        list(models.keys()),
    )

    selected_model = models[selected_model_name]

    st.divider()

    # --------------------------------------------------------
    # HEALTH INFORMATION
    # --------------------------------------------------------

    st.subheader("Health Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        high_bp = st.selectbox(
            "High Blood Pressure",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

        high_chol = st.selectbox(
            "High Cholesterol",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

        chol_check = st.selectbox(
            "Cholesterol Check",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=100.0,
            value=27.0,
            step=0.1,
        )

    with col2:
        smoker = st.selectbox(
            "Smoker",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

        stroke = st.selectbox(
            "History of Stroke",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

        heart_disease = st.selectbox(
            "Heart Disease / Heart Attack",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

        physical_activity = st.selectbox(
            "Physical Activity",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    with col3:
        fruits = st.selectbox(
            "Consume Fruits Regularly",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

        veggies = st.selectbox(
            "Consume Vegetables Regularly",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

        heavy_alcohol = st.selectbox(
            "Heavy Alcohol Consumption",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

        healthcare = st.selectbox(
            "Has Healthcare Coverage",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    st.divider()

    # --------------------------------------------------------
    # HEALTHCARE / QUALITY OF LIFE
    # --------------------------------------------------------

    st.subheader("Healthcare & Quality of Life")

    col1, col2, col3 = st.columns(3)

    with col1:
        no_doc_cost = st.selectbox(
            "Could Not See Doctor Due to Cost",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

        gen_health = st.selectbox(
            "General Health",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Excellent",
                2: "Very Good",
                3: "Good",
                4: "Fair",
                5: "Poor",
            }[x],
        )

    with col2:
        ment_health = st.slider(
            "Days of Poor Mental Health",
            min_value=0,
            max_value=30,
            value=0,
        )

        phys_health = st.slider(
            "Days of Poor Physical Health",
            min_value=0,
            max_value=30,
            value=0,
        )

    with col3:
        diff_walk = st.selectbox(
            "Difficulty Walking / Climbing Stairs",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
        )

    st.divider()

    # --------------------------------------------------------
    # DEMOGRAPHICS
    # --------------------------------------------------------

    st.subheader("Demographic Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.selectbox(
            "Age Group",
            list(range(1, 14)),
            format_func=lambda x: f"Age category {x}",
        )

    with col2:
        education = st.selectbox(
            "Education Level",
            list(range(1, 7)),
            format_func=lambda x: f"Education category {x}",
        )

    with col3:
        income = st.selectbox(
            "Income Level",
            list(range(1, 9)),
            format_func=lambda x: f"Income category {x}",
        )

    st.divider()

    # --------------------------------------------------------
    # CREATE FEATURE VECTOR
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            high_bp,
            high_chol,
            chol_check,
            bmi,
            smoker,
            stroke,
            heart_disease,
            physical_activity,
            fruits,
            veggies,
            heavy_alcohol,
            healthcare,
            no_doc_cost,
            gen_health,
            ment_health,
            phys_health,
            diff_walk,
            sex,
            age,
            education,
            income,
        ]],
        columns=FEATURE_COLUMNS,
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predict_button = st.button(
        "🔮 Predict",
        type="primary",
        use_container_width=True,
    )

    if predict_button:

        try:

            prediction = float(
                selected_model.predict(input_data)[0]
            )

            st.subheader("Prediction Result")

            st.markdown(
                f"""
                <div class="result-box">
                    <div>Predicted Score</div>
                    <div class="score">{prediction:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            # ------------------------------------------------
            # SECONDARY CLASSIFICATION INTERPRETATION
            # ------------------------------------------------

            if prediction >= 0.5:

                st.warning(
                    "The regression score is at or above the "
                    "0.5 threshold."
                )

                interpretation = "Positive class"

            else:

                st.success(
                    "The regression score is below the "
                    "0.5 threshold."
                )

                interpretation = "Negative class"

            st.info(
                f"Secondary classification interpretation: "
                f"**{interpretation}**"
            )

            st.caption(
                "Note: This value is a regression score, not a "
                "calibrated probability. The 0.5 threshold is used "
                "only for secondary classification interpretation."
            )

            # ------------------------------------------------
            # MODEL PERFORMANCE
            # ------------------------------------------------

            if selected_model_name in MODEL_RESULTS:

                st.divider()
                st.subheader("Model Performance on Test Set")

                metrics = MODEL_RESULTS[selected_model_name]

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("R²", f"{metrics['R²']:.4f}")
                c2.metric("RMSE", f"{metrics['RMSE']:.4f}")
                c3.metric("MAE", f"{metrics['MAE']:.4f}")
                c4.metric("MSE", f"{metrics['MSE']:.4f}")

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# PAGE 2 — MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.header("Model Comparison")

    if not MODEL_RESULTS:

        st.info(
            "Model results will appear here as models are completed."
        )

    else:

        results_df = pd.DataFrame(MODEL_RESULTS).T

        results_df = results_df.sort_values(
            by="R²",
            ascending=False,
        )

        st.subheader("Regression Performance")

        st.dataframe(
            results_df.style.format(
                {
                    "R²": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "MAE": "{:.4f}",
                    "MSE": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

        st.divider()

        # ----------------------------------------------------
        # R² CHART
        # ----------------------------------------------------

        st.subheader("R² Comparison")

        r2_chart = results_df[["R²"]]

        st.bar_chart(r2_chart)

        # ----------------------------------------------------
        # RMSE CHART
        # ----------------------------------------------------

        st.subheader("RMSE Comparison")

        rmse_chart = results_df[["RMSE"]]

        st.bar_chart(rmse_chart)

        st.caption(
            "Higher R² indicates better explanatory performance, "
            "while lower RMSE indicates lower prediction error."
        )


# ============================================================
# PAGE 3 — ABOUT
# ============================================================

elif page == "About":

    st.header("About the Project")

    st.markdown(
        """
        ### CDC Diabetes Health Indicators

        This dashboard is the interactive component of a Machine
        Learning capstone project.

        The project investigates machine learning models using the
        CDC Diabetes Health Indicators dataset.

        ### Regression

        The target variable is treated as a numerical 0/1 target
        for regression experiments.

        The dashboard allows users to:

        - Enter health and demographic information
        - Select a trained regression model
        - Generate a prediction
        - View model performance
        - Compare completed regression models

        ### Current Models

        - Decision Tree Regressor
        - Random Forest Regressor

        Additional models can be added as they are completed.

        ### Technologies

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Joblib
        - Streamlit
        """
    )

    st.divider()

    st.subheader("Loaded Models")

    if models:

        for model_name in models:
            st.write(f"✅ {model_name}")

    else:

        st.write("No models loaded.")