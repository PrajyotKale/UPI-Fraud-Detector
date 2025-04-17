# UPI-Fraud-Detector

## 🚀 Overview
This Streamlit application utilizes a **hybrid machine learning model** (XGBoost + PyTorch Transformer) to detect fraudulent transactions. Users can analyze **single transactions** or **batch transactions** via CSV upload.

## 📌 Features
- 🌟 **Single Transaction Prediction**: Input transaction details via UI form.
- 📊 **Batch Prediction**: Upload CSV files for bulk fraud analysis.
- 💡 **Hybrid Model**: Combines XGBoost, Transformer, and a Meta model.
- 🎨 **Light Theme & Intuitive UI**: Designed with sliders, dropdowns, and progress bars for user-friendly experience.

## ⚙️ Installation

### **1️⃣ Clone Repository**
```bash
git clone https://github.com/PrajyotKale/UPI-Fraud-Detector.git
cd UPI-Fraud-Detector
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the App
```bash
streamlit run app.py
```
## 🔮 How to Use
### Single Transaction Analysis
- Launch the app.

- Navigate to "Single Prediction" via the sidebar.

- Enter transaction details using the form.

- Click "Analyze Transaction" to get fraud prediction results.

### Batch Transaction Analysis
- Click "Batch Prediction" in the sidebar.

- Upload a CSV file with transaction data.

- Click "Run Batch Analysis" to get fraud insights.

## 🏗️ Model Files & Loading
The app loads models from the /models directory:

- meta_model.json: Meta model (XGBoost)

- xgb_model.json: XGBoost base model

- transformer_model.pt: PyTorch Transformer model

- scaler.pkl: Preprocessing scaler

- label_encoders.pkl: Label encoders for categorical data

## 🌍 Deploying Online
### Streamlit Community Cloud (Free)
- 1) Push the project to GitHub.

- 2) Visit Streamlit Community Cloud.

- 3) Deploy your app!

### Alternative Hosting Platforms
- Render: Render.com

- Hugging Face Spaces: Hugging Face Spaces

- AWS / Azure: Deploy via serverless or containerized solutions.

## 🛠️ IoT Integration
- To integrate with IoT devices:

- Use MQTT to receive real-time transaction data.

- Deploy on Raspberry Pi / Edge AI devices.

- Connect to AWS IoT, Azure IoT Hub, or Google IoT Core.

## 🎯 Future Improvements
### ✅ Enhancing Meta model for better fraud detection.

### ✅ Adding Geolocation-based fraud risk analysis.

### ✅ Improving IoT data pipeline for real-time monitoring.
