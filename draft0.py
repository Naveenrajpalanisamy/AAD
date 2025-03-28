import datetime
import numpy as np
import pandas as pd
import joblib
from scapy.all import sniff, wrpcap, rdpcap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Function to handle and display captured packets
def packet_handler(packet):
    if packet.haslayer("IP"):
        print(packet.summary())  # Print basic packet details

# Ask user for port filtering option
port = input("Enter a port number to filter (or press Enter to capture all traffic): ")
filter_str = f"port {port}" if port.isdigit() else ""

# Generate a timestamped filename
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
pcap_filename = f"manual_capture_{timestamp}.pcap"

print(f"\nCapturing packets on port {port if port else 'ALL'}... Press CTRL+C to stop.")
try:
    packets = sniff(filter=filter_str, prn=packet_handler, store=True)
    wrpcap(pcap_filename, packets)
    print(f"\n Packets saved to {pcap_filename}")
except KeyboardInterrupt:
    print("\n Capture stopped manually.")
    if packets:
        wrpcap(pcap_filename, packets)
        print(f" Packets saved to {pcap_filename}")
    else:
        print("No packets were captured.")

# Step 2: Convert PCAP to CSV
print("\n Extracting features from captured packets...")
packets = rdpcap(pcap_filename)
data = []
for pkt in packets:
    if pkt.haslayer("IP"):
        data.append({
            "src_ip": pkt["IP"].src,
            "dst_ip": pkt["IP"].dst,
            "protocol": pkt["IP"].proto,
            "packet_length": len(pkt),
            "timestamp": pkt.time
        })
df = pd.DataFrame(data)
csv_filename = f"network_data_{timestamp}.csv"
df.to_csv(csv_filename, index=False)
print(f" Captured data saved to {csv_filename}")

# Step 3: Train Machine Learning Model
print("\n Training Attack Detection Model...")

# Load the captured data
df = pd.read_csv(csv_filename)
df["label"] = "normal"  # Placeholder, in real cases you'd label attacks

# Load the NSL-KDD dataset
nsl_kdd = pd.read_csv("NSL_KDD_Train.csv")  # Ensure it's preprocessed correctly

# Fix column names if necessary
print("\n Checking NSL-KDD dataset columns:", nsl_kdd.columns)

# Ensure 'protocol' column exists
if "protocol" not in nsl_kdd.columns:
    if "protocol_type" in nsl_kdd.columns:  # NSL-KDD may use 'protocol_type'
        nsl_kdd.rename(columns={"protocol_type": "protocol"}, inplace=True)
    else:
        raise KeyError(" 'protocol' column not found in NSL-KDD dataset!")

# Identify categorical columns
categorical_columns = ["protocol", "service", "flag"] 

# Encode categorical columns
label_encoders = {}
for col in categorical_columns:
    if col in nsl_kdd.columns:
        label_encoders[col] = LabelEncoder()
        nsl_kdd[col] = label_encoders[col].fit_transform(nsl_kdd[col])
    else:
        print(f" Warning: Column '{col}' not found in NSL-KDD dataset.")

# Ensure all columns are numeric before training
X = nsl_kdd.drop(["label", "difficulty"], axis=1, errors="ignore")
y = nsl_kdd["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test) #accuracy
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "attack_detection_model.pkl")
print(" Model saved as attack_detection_model.pkl")

# Step 4: Detect Adversarial Attacks in Live Traffic
print("\n Running Real-Time Attack Detection...")
clf = joblib.load("attack_detection_model.pkl")

# Predefined protocol mapping based on NSL-KDD dataset
protocol_mapping = {"icmp": 1, "tcp": 6, "udp": 17}

# Safely encode unseen categorical values
def encode_value(column, value):
    return protocol_mapping.get(value, protocol_mapping["tcp"])  # Default to TCP

# Real-time attack detection
def detect_attack(packet):
    if packet.haslayer("IP"):
        feature_vector = {
            "protocol": encode_value("protocol", packet["IP"].proto),
            "service": 0,  # Placeholder (you need to extract real service info)
            "flag": 0,  # Placeholder (you need to extract real flag info)
            "src_bytes": len(packet),
            "dst_bytes": 0,  # Cannot infer directly
            "count": 0,
            "srv_count": 0,
            "serror_rate": 0.0,
            "srv_serror_rate": 0.0,
            "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0,
            "same_srv_rate": 0.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 0,
            "dst_host_srv_count": 0,
            "dst_host_same_srv_rate": 0.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0
        }

        # Convert to DataFrame for model prediction
        df = pd.DataFrame([feature_vector])

        # Ensure all missing columns from X_train exist in df
        missing_cols = set(X_train.columns) - set(df.columns)
        for col in missing_cols:
            
            df[col] = 0  # Assign a default value (0 for numeric features)

        # Ensure column order matches model input
        df = df[X_train.columns]

        # Make prediction
        prediction = clf.predict(df)[0]

        if prediction != "normal":  # No need to use prediction[0], it's already a single value
            print(f" Adversarial Attack Detected: {packet['IP'].src} -> {packet['IP'].dst}")

print("\n Running Real-Time Attack Detection...")
sniff(prn=detect_attack, store=0)
