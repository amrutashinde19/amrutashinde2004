import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Streamlit page setup
st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

# Accuracy history tracker
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🎓 Student Dropout/passout Prediction ")

# Sidebar menu
menu = st.sidebar.radio("Select Task", [
    "Train Model",
    "Predict Dropout",
    "View Accuracy History"
])

# Generate synthetic student data
def generate_data():
    X, y = make_classification(n_samples=300, n_features=6, n_informative=4,
                               n_redundant=0, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["Target"] = y
    return df

# ---------- 1. Train Model ----------
if menu == "Train Model":
    df = generate_data()
    st.success("✅ Synthetic data generated!")

    if st.checkbox("Show Sample Data"):
        st.dataframe(df.head())

    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=4,
        verbose=0
    )

    with st.spinner("Training model..."):
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    st.success(f"🎯 Model Accuracy: {acc:.2%}")

    # Save model and features to session
    st.session_state["model"] = model
    st.session_state["features"] = list(X.columns)
    st.session_state.history.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "accuracy": acc
    })

    # Display classification report
    st.subheader("📋 Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    # Show confusion matrix
    st.subheader("🔍 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)

    # Show feature importance
    st.subheader("📊 Feature Importance")
    feat_imp = pd.Series(model.get_feature_importance(), index=X.columns)
    fig2, ax2 = plt.subplots()
    feat_imp.sort_values().plot(kind='barh', ax=ax2)
    st.pyplot(fig2)

# ---------- 2. Predict Dropout ----------
elif menu == "Predict Dropout":
    if "model" not in st.session_state:
        st.warning("⚠️ Please train the model first.")
    else:
        st.subheader("Enter Student Information")
        input_data = {}
        for col in st.session_state["features"]:
            input_data[col] = st.number_input(col, step=1.0)

        if st.button("Predict"):
            df_input = pd.DataFrame([input_data])
            prediction = st.session_state["model"].predict(df_input)[0]

            st.subheader("Prediction Result")
            if str(prediction) == "1":
                st.error(f"⚠️ Prediction: {prediction} (Likely to Dropout)")
            else:
                st.success(f"✅ Prediction: {prediction} (Likely to Continue)")

# ---------- 3. View Accuracy History ----------
elif menu == "View Accuracy History":
    if st.session_state.history:
        st.subheader("📈 Model Accuracy Over Time")
        hist_df = pd.DataFrame(st.session_state.history)
        st.line_chart(hist_df.set_index("date"))
        st.dataframe(hist_df)
    else:
        st.info("ℹ️ No model training history found.")
