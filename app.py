import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1. Carga y preparación de datos
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("Loan Eligibility Prediction.csv")

    # Columnas numéricas
    num_cols = ['Applicant_Income', 'Coapplicant_Income',
                'Loan_Amount', 'Loan_Amount_Term']

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Columnas categóricas
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Credit_History', 'Property_Area', 'Loan_Status']

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Codificación de categóricas con LabelEncoder
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders, num_cols, cat_cols


@st.cache_resource
def train_model(df):
    X = df.drop(['Loan_Status', 'Customer_ID'], axis=1)
    y = df['Loan_Status']

    feature_cols = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    return model, feature_cols, acc


def credit_score(cliente: pd.DataFrame, model, feature_cols):
    cliente = cliente[feature_cols]
    prob = model.predict_proba(cliente)[0][1]
    score = int(300 + prob * 600)
    return score, prob


# =========================
# 2. Interfaz Streamlit
# =========================

def main():
    st.set_page_config(page_title="Credit Score App", layout="centered")

    st.title("📊 Credit Score - Aprobación de Préstamos")
    st.write("App de ejemplo usando Random Forest y tu dataset de préstamos.")

    df, encoders, num_cols, cat_cols = load_data()
    model, feature_cols, acc = train_model(df)

    st.sidebar.header("Parámetros del solicitante")

    gender = st.sidebar.selectbox("Género", encoders['Gender'].classes_)
    married = st.sidebar.selectbox("¿Casado?", encoders['Married'].classes_)
    dependents = st.sidebar.selectbox("Dependientes", encoders['Dependents'].classes_)
    education = st.sidebar.selectbox("Educación", encoders['Education'].classes_)
    self_emp = st.sidebar.selectbox("¿Autónomo?", encoders['Self_Employed'].classes_)
    credit_hist = st.sidebar.selectbox("Historial de crédito", encoders['Credit_History'].classes_)
    property_area = st.sidebar.selectbox("Zona", encoders['Property_Area'].classes_)

    applicant_income = st.sidebar.number_input("Ingreso solicitante", 0, 200000, 5000)
    coapplicant_income = st.sidebar.number_input("Ingreso co-solicitante", 0, 200000, 1500)
    loan_amount = st.sidebar.number_input("Monto del préstamo", 1, 1000, 150)
    loan_term = st.sidebar.number_input("Plazo", 30, 480, 360)

    # Codificar valores
    data_dict = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Loan_Amount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": encoders['Credit_History'].transform([credit_hist])[0],
        "Gender": encoders['Gender'].transform([gender])[0],
        "Married": encoders['Married'].transform([married])[0],
        "Dependents": encoders['Dependents'].transform([dependents])[0],
        "Education": encoders['Education'].transform([education])[0],
        "Self_Employed": encoders['Self_Employed'].transform([self_emp])[0],
        "Property_Area": encoders['Property_Area'].transform([property_area])[0],
    }

    cliente = pd.DataFrame([data_dict])

    st.subheader("📄 Datos del solicitante")
    st.write(cliente)

    if st.button("Calcular Credit Score"):
        score, prob = credit_score(cliente, model, feature_cols)

        st.subheader("📈 Resultado del análisis")
        st.write(f"**Probabilidad de aprobación:** {prob:.2%}")
        st.write(f"**Credit Score:** `{score}` (300–900)")

        if score >= 720:
            st.success("Riesgo Bajo")
        elif score >= 620:
            st.info("Riesgo Medio")
        else:
            st.warning("Riesgo Alto")

        st.subheader("🔍 Importancia de características")
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        st.bar_chart(importances)


if __name__ == "__main__":
    main()
