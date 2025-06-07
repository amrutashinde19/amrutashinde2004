import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import io
from datetime import datetime

st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

# Store accuracy history
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🎓 Student Dropout Prediction using CatBoost")

menu = st.sidebar.radio("Select Task", [
    "Upload and Train Model",
    "Predict Dropout",
    "View Accuracy History"
])

# ----------------- 1. Upload and Train Model -----------------
if menu == "Upload and Train Model":
    data_file = st.file_uploader("Upload student data CSV file", type=["csv"])

    if data_file:
        try:
            df = pd.read_csv(data_file, delimiter=';')
            st.success("✅ Data loaded successfully!")

            if st.checkbox("Show Raw Data"):
                st.dataframe(df.head())

            if "Target" not in df.columns:
                st.error("❌ The dataset must include a 'Target' column.")
            else:
                X = df.drop("Target", axis=1)
                y = df["Target"]
                cat_cols = X.select_dtypes(include='object').columns.tolist()

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                model = CatBoostClassifier(
                    iterations=300,
                    learning_rate=0.1,
                    depth=6,
                    cat_features=cat_cols,
                    verbose=0
                )

                with st.spinner("Training CatBoost model..."):
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                cm = confusion_matrix(y_test, y_pred)

                st.success(f"🎯 Model Accuracy: {acc:.2%}")
                st.session_state.history.append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "accuracy": acc
                })

                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(report).transpose())

                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                st.pyplot(fig)

                st.subheader("Feature Importance")
                feat_imp = pd.Series(model.get_feature_importance(), index=X.columns)
                fig2, ax2 = plt.subplots()
                feat_imp.sort_values().plot(kind='barh', ax=ax2)
                st.pyplot(fig2)

                if st.button("Save Trained Model"):
                    with open("student_dropout_model.pkl", "wb") as f:
                        pickle.dump((model, cat_cols, list(X.columns)), f)
                    st.success("💾 Model saved as 'student_dropout_model.pkl'")

        except Exception as e:
            st.error(f"❌ Error during training: {e}")

# ----------------- 2. Predict Dropout -----------------
elif menu == "Predict Dropout":
    model_file = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])

    if model_file:
        try:
            loaded = pickle.load(io.BytesIO(model_file.read()))
            model, cat_cols, feature_cols = loaded

            st.success("✅ Model loaded successfully.")
            st.subheader("Enter Student Information")

            input_data = {}
            for col in feature_cols:
                if col in cat_cols:
                    input_data[col] = st.selectbox(col, [str(i) for i in range(8)])
                else:
                    input_data[col] = st.number_input(col, step=1.0)

            if st.button("Predict"):
                df_input = pd.DataFrame([input_data])
                df_input = df_input[feature_cols]
                prediction = model.predict(df_input)[0]

                st.subheader("Prediction Result")
                if str(prediction) == "1":
                    st.error(f"⚠️ Prediction: {prediction} (Likely to Dropout)")
                else:
                    st.success(f"✅ Prediction: {prediction} (Likely to Continue)")
        except Exception as e:
            st.error(f"❌ Error loading model or making prediction: {e}")

# ----------------- 3. View Accuracy History -----------------
elif menu == "View Accuracy History":
    if st.session_state.history:
        st.subheader("📈 Model Accuracy Over Time")
        hist_df = pd.DataFrame(st.session_state.history)
        st.line_chart(hist_df.set_index("date"))
        st.dataframe(hist_df)
    else:
        st.info("ℹ️ No model training history found.")
