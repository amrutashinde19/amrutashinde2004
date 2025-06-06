import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

# Initialize session state for metrics
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Student Dropout Predictor", layout="wide")
st.title("🎓 Student Dropout Prediction using CatBoost")

# Sidebar options
option = st.sidebar.radio("Choose Task", [
    "📂 Upload & Train Model",
    "🔮 Predict Using Trained Model",
    "📊 View Accuracy History"
])

# ============ 📂 TRAIN & SAVE MODEL ============
if option == "📂 Upload & Train Model":
    uploaded_file = st.file_uploader("Upload your student dataset (.csv)", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, delimiter=';')
            st.success("✅ File uploaded successfully!")

            if st.checkbox("Show Data"):
                st.dataframe(df.head())

            if "Target" not in df.columns:
                st.error("❌ 'Target' column not found.")
            else:
                X = df.drop("Target", axis=1)
                y = df["Target"]
                cat_features = X.select_dtypes(include='object').columns.tolist()

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                model = CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.1,
                    depth=6,
                    cat_features=cat_features,
                    verbose=0
                )

                with st.spinner("Training model..."):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                cm = confusion_matrix(y_test, y_pred)

                st.subheader("✅ Accuracy:")
                st.success(f"{acc:.2%}")
                st.session_state.history.append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "accuracy": acc
                })

                st.subheader("📊 Classification Report:")
                st.dataframe(pd.DataFrame(report).transpose())

                st.subheader("📌 Confusion Matrix:")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=model.classes_, yticklabels=model.classes_)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

                st.subheader("🔍 Feature Importance:")
                feature_imp = pd.Series(model.get_feature_importance(), index=X.columns)
                fig2, ax2 = plt.subplots()
                feature_imp.sort_values().plot(kind='barh', ax=ax2)
                st.pyplot(fig2)

                if st.button("💾 Save Trained Model"):
                    with open("student_dropout_model.pkl", "wb") as f:
                        pickle.dump((model, cat_features, X.columns.tolist()), f)
                    st.success("✅ Model saved as 'student_dropout_model.pkl'")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ============ 🔮 PREDICT NEW STUDENT ============
elif option == "🔮 Predict Using Trained Model":
    st.info("Upload a saved model to make predictions.")
    model_file = st.file_uploader("Upload saved model (.pkl)", type=["pkl"])
    
    if model_file:
        try:
            model, cat_features, feature_cols = pickle.load(model_file)
            st.success("✅ Model loaded.")

            st.subheader("📥 Enter Student Information:")

            new_data = {}
            for col in feature_cols:
                if col in cat_features:
                    new_data[col] = st.selectbox(f"{col}", ["yes", "no", "maybe", "other"])
                else:
                    new_data[col] = st.number_input(f"{col}", format="%f")

            if st.button("🚀 Predict"):
                input_df = pd.DataFrame([new_data])
                prediction = model.predict(input_df)[0]
                st.markdown("### 🎯 Predicted Result:")
                if prediction.lower() in ["dropout", "1", "yes"]:
                    st.error(f"⚠️ {prediction}")
                else:
                    st.success(f"✅ {prediction}")

        except Exception as e:
            st.error(f"❌ Error loading model: {e}")

# ============ 📉 ACCURACY HISTORY ============
elif option == "📊 View Accuracy History":
    if st.session_state.history:
        st.subheader("📈 Model Accuracy Over Time")
        hist_df = pd.DataFrame(st.session_state.history)
        st.line_chart(hist_df.set_index("date"))
        st.dataframe(hist_df)
    else:
        st.info("No training history available yet.")
