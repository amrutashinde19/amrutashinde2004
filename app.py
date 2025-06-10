import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Page config
st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

# Accuracy history tracker
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🎓 Student Dropout/Passout Prediction")

# Sidebar menu
menu = st.sidebar.radio("Select Task", [
    "Train Model",
    "Predict Dropout",
    "View Accuracy History"
])

# ---------- 1. Train Model ----------
if menu == "Train Model":
    try:
        df = pd.read_csv("/mnt/data/data.csv")
        st.success("✅ Dataset loaded successfully!")

        if st.checkbox("Show Sample Data"):
            st.dataframe(df.head())

        if "Target" not in df.columns:
            st.error("❌ 'Target' column not found in the dataset.")
        else:
            X = df.drop("Target", axis=1)
            y = df["Target"]

            # Identify categorical features
            cat_features = X.select_dtypes(include=['object']).columns.tolist()

            # Ensure consistent dtype for categorical features
            for col in cat_features:
                X[col] = X[col].astype(str)

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
                model.fit(X_train, y_train, cat_features=cat_features)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            st.success(f"🎯 Model Accuracy: {acc:.2%}")

            # Save model and feature info in session
            st.session_state["model"] = model
            st.session_state["features"] = X.columns.tolist()
            st.session_state["cat_features"] = cat_features
            st.session_state.history.append({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "accuracy": acc
            })

            # Show reports
            st.subheader("📋 Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

            st.subheader("🔍 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            st.pyplot(fig)

            st.subheader("📊 Feature Importance")
            feat_imp = pd.Series(model.get_feature_importance(), index=X.columns)
            fig2, ax2 = plt.subplots()
            feat_imp.sort_values().plot(kind='barh', ax=ax2)
            st.pyplot(fig2)

    except FileNotFoundError:
        st.error("❌ Dataset file not found. Please upload 'data.csv'.")

# ---------- 2. Predict Dropout ----------
elif menu == "Predict Dropout":
    if "model" not in st.session_state:
        st.warning("⚠️ Please train the model first.")
    else:
        st.subheader("📝 Enter Student Information")

        input_data = {}
        for col in st.session_state["features"]:
            if col in st.session_state["cat_features"]:
                input_data[col] = st.text_input(col, placeholder="Enter category value")
            else:
                input_data[col] = st.number_input(col, step=1.0)

        if st.button("Predict"):
            df_input = pd.DataFrame([input_data])

            # Ensure category dtype for CatBoost
            for cat_col in st.session_state["cat_features"]:
                df_input[cat_col] = df_input[cat_col].astype(str)

            prediction = st.session_state["model"].predict(df_input)[0]

            st.subheader("🔮 Prediction Result")
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
