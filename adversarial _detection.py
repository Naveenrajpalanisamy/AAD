# Phase 1: Data Collection and Preprocessing 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import FastGradientMethod
from flask import Flask, request, jsonify

# Load the dataset
train_data = pd.read_csv("NSL_KDD_Train.csv")
test_data = pd.read_csv("NSL_KDD_Test.csv")

# Handle missing values
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Encode categorical features
categorical_columns = ["protocol_type", "service", "flag"]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

# Normalize numerical features
numerical_columns = train_data.select_dtypes(include=["int64", "float64"]).columns
scaler = MinMaxScaler()
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

# Encode target labels
# Encode target labels
label_encoder = LabelEncoder()
label_encoder.fit(train_data["label"])  # Fit only on training labels

# Transform y_train
y_train = label_encoder.transform(train_data["label"])

# Handle unseen labels in y_test
y_test = test_data["label"].apply(lambda x: x if x in label_encoder.classes_ else "unknown")
label_encoder.classes_ = np.append(label_encoder.classes_, "unknown")  # Add "unknown" class
y_test = label_encoder.transform(y_test)


# Feature selection (example: select top 20 features)
selected_features = numerical_columns[:20]
X_train = train_data[selected_features]
X_test = test_data[selected_features]

print("Preprocessing complete!")

# Phase 2: Model Selection and Training
print("Training Logistic Regression Model...")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
print("Logistic Regression Model Trained!")

# Evaluate the model
y_pred = log_reg.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Phase 3: Adversarial Attack Simulation
print("Generating Adversarial Examples...")

# Create ART classifier for Logistic Regression
art_classifier = SklearnClassifier(model=log_reg)

# Generate adversarial examples using FGSM
attack = FastGradientMethod(estimator=art_classifier, eps=0.1)
X_test_adv = attack.generate(X_test.to_numpy())

# Evaluate on adversarial examples
y_pred_adv = log_reg.predict(X_test_adv)
print(f"Accuracy on Adversarial Examples: {accuracy_score(y_test, y_pred_adv):.4f}")
print("\nClassification Report on Adversarial Examples:")
print(classification_report(y_test, y_pred_adv))

# Phase 4: Flask API for Real-Time Prediction
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request
    data = request.json["data"]
    
    # Convert input to NumPy array and preprocess it
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)  # Normalize numerical features
    
    # Make a prediction
    prediction = log_reg.predict(data)
     # Convert the prediction back to the original label
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    return jsonify({"prediction": predicted_label})

@app.route("/alert", methods=["POST"])
def alert():
    # Get the input data from the request
    data = request.json["data"]
    
    # Convert input to NumPy array and preprocess it
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)  # Normalize numerical features
    
    # Make a prediction
    prediction = log_reg.predict(data)
    
  # Convert the prediction back to the original label
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    # Check if the prediction is an attack
    if predicted_label != "normal":
        return jsonify({"alert": "Adversarial Attack Detected!", "prediction": predicted_label})
    else:
        return jsonify({"alert": "No Attack Detected.", "prediction": predicted_label})
if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000)  