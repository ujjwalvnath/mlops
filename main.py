import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib

# Load the dataset
data = pd.read_csv("data/iris.csv")

# Split into train and test sets
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)

X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train['species']
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test['species']

# Train a Decision Tree model
model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, y_train)

# Make predictions and evaluate
predictions = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print(f"Decision Tree Accuracy: {accuracy:.3f}")

# Save the trained model
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model.joblib")
print("Model saved to artifacts/model.joblib")
