import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Student Dropout Predictor", layout="centered")

st.title("🎓 Student Dropout Prediction using CatBoost")

# File uploader
uploaded_file = st.file_uploader("C:\\Users\\Shree\\Desktop\\ML CAS STUDENT DROPOUT PASSOUT\\data.csv")


if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, delimiter=';')
        st.success("✅ File uploaded successfully!")
        
        # Display preview
        if st.checkbox("Show Data"):
            st.dataframe(df.head())
        
        if "Target" not in df.columns:
            st.error("❌ 'Target' column not found in uploaded file.")
        else:
            X = df.drop("Target", axis=1)
            y = df["Target"]
            
            # Detect categorical columns
            cat_features = X.select_dtypes(include='object').columns.tolist()
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Model training
            model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.1,
                depth=6,
                cat_features=cat_features,
                verbose=0  # hide training logs
            )
            
            with st.spinner("Training model..."):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Output results
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            st.subheader("✅ Accuracy:")
            st.success(f"{acc:.2%}")

            st.subheader("📊 Classification Report:")
            st.dataframe(pd.DataFrame(report).transpose())

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a dataset to begin.")
