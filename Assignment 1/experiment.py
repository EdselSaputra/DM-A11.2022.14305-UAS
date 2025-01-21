import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = '/mnt/data/ufcfights_sept30 (1).csv'
data = pd.read_csv(file_path)

# Preprocessing
data = data.copy()

# Encode categorical columns
label_encoders = {}
for col in ['event', 'fighter_1', 'fighter_2', 'method']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Convert 'result' to binary outcome: win = 1, loss = 0
data['result'] = data['result'].apply(lambda x: 1 if x == 'win' else 0)

# Feature selection
X = data[['event', 'fighter_1', 'fighter_2', 'method', 'round']]
y = data['result']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Optional: Save the label encoders for later use
import pickle
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Save the model
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
