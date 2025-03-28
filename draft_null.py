# Step 1: Data Collection and Preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainer
from flask import Flask, request, jsonify

# Load the dataset (replace with your actual data loading code)
train_data = pd.read_csv("NSL_KDD_Train.csv")
test_data = pd.read_csv("NSL_KDD_Test.csv")

# Handle missing values (if any)
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

# Feature selection (example: select top 20 features based on importance)
# Replace this with your actual feature selection logic
selected_features = numerical_columns[:20]  # Example: select first 20 numerical features
X_train = train_data[selected_features]
y_train = train_data["label"]
X_test = test_data[selected_features]
y_test = test_data["label"]

print("Preprocessing complete!")

# Step 2: Model Selection and Training (SVM)
# Train an SVM model
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)

# Evaluate the baseline model
y_pred = svm.predict(X_test)
print(f"Baseline SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nBaseline SVM Classification Report:")
print(classification_report(y_test, y_pred))

# Step 3: Adversarial Attack Simulation (ART)
# Create ART classifier
art_classifier = SklearnClassifier(model=svm)

from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import FastGradientMethod

# Train a model (e.g., SVM or Random Forest)
from sklearn.svm import SVC
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)

# Create ART classifier
art_classifier = SklearnClassifier(model=svm)

# Generate adversarial examples using Fast Gradient Method (FGM)
attack = FastGradientMethod(estimator=art_classifier, eps=0.1)
X_test_adv = attack.generate(X_test)

# Evaluate the model on adversarial examples
y_pred_adv = svm.predict(X_test_adv)
print(f"Accuracy on Adversarial Examples: {accuracy_score(y_test, y_pred_adv):.4f}")

# Generate adversarial examples using Fast Gradient Method (FGM)
attack = FastGradientMethod(estimator=art_classifier, eps=0.1)
X_test_adv = attack.generate(X_test)

import numpy as np

# Combine original and adversarial examples
X_train_augmented = np.vstack((X_train, X_test_adv))
y_train_augmented = np.hstack((y_train, y_test))

# Shuffle the augmented dataset
from sklearn.utils import shuffle
X_train_augmented, y_train_augmented = shuffle(X_train_augmented, y_train_augmented, random_state=42)

# Evaluate on adversarial examples
y_pred_adv = svm.predict(X_test_adv)
print(f"Accuracy on Adversarial Examples: {accuracy_score(y_test, y_pred_adv):.4f}")
print("\nClassification Report on Adversarial Examples:")
print(classification_report(y_test, y_pred_adv))

# Step 4: Robust Model Training
# Create an adversarial trainer
trainer = AdversarialTrainer(art_classifier, attacks=attack)

# Train the robust model on adversarial examples
trainer.fit(X_train, y_train)

# Train a robust model on the augmented dataset
robust_svm = SVC(probability=True, random_state=42)
robust_svm.fit(X_train_augmented, y_train_augmented)

# Evaluate the robust model on adversarial examples
y_pred_robust = robust_svm.predict(X_test_adv)
print(f"Accuracy of Robust Model on Adversarial Examples: {accuracy_score(y_test, y_pred_robust):.4f}")

# Evaluate the robust model
y_pred_robust = trainer.predict(X_test_adv)
print(f"Accuracy of Robust Model on Adversarial Examples: {accuracy_score(y_test, y_pred_robust):.4f}")
print("\nClassification Report of Robust Model:")
print(classification_report(y_test, y_pred_robust))

# Step 5: Integrate with Network Traffic Monitoring System
app = Flask(__name__)

# Load the robust model
model = trainer

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request
    data = request.json["data"]
    
    # Convert the input data to a NumPy array
    data = np.array(data).reshape(1, -1)
    
    # Preprocess the input data (normalize and encode)
    data = scaler.transform(data)  # Normalize numerical features
    for col, le in label_encoders.items():  # Encode categorical features
        if col in selected_features:
            data[col] = le.transform(data[col])
    
    # Make a prediction
    prediction = model.predict(data)
    
    # Return the prediction as a JSON response
    return jsonify({"prediction": int(prediction[0])})

@app.route("/alert", methods=["POST"])
def alert():
    # Get the input data from the request
    data = request.json["data"]
    
    # Convert the input data to a NumPy array
    data = np.array(data).reshape(1, -1)
    
    # Preprocess the input data (normalize and encode)
    data = scaler.transform(data)  # Normalize numerical features
    for col, le in label_encoders.items():  # Encode categorical features
        if col in selected_features:
            data[col] = le.transform(data[col])
    
    # Make a prediction
    prediction = model.predict(data)
    
    # Check if the prediction is an attack
    if prediction[0] != "normal":
        return jsonify({"alert": "Adversarial Attack Detected!", "prediction": int(prediction[0])})
    else:
        return jsonify({"alert": "No Attack Detected.", "prediction": int(prediction[0])})

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000)