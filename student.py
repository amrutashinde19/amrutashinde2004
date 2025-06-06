import pandas as pd
from catboost import catBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load data
df = pd.read_csv("C:\\Users\\Shree\\Desktop\\ML CAS STUDENT DROUPOUT PASSOUT\\data.csv", delimiter=';')

# Step 2: Split X and y
X = df.drop("Target", axis=1)
y = df["Target"]

# Step 3: Categorical columns
cat_features = X.select_dtypes(include='object').columns.tolist()

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Define the model
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_features,
    verbose=100
)

# Step 6: Train the model
model.fit(X_train, y_train)
# Step 7: Predictions and accuracy
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))
