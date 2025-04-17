import streamlit as st
from datetime import datetime
from fraud_detection import predict_fraud_hybrid  # Import fraud detection function

# Set light theme
st.set_page_config(page_title="Fraud Prediction", page_icon="⚡", layout="centered")

st.title("🚀 Fraud Prediction Form")

with st.form("prediction_form"):
    st.write("🔍 Enter transaction details:")

    event_timestamp = st.text_input("Event Timestamp (YYYY-MM-DDTHH:MM:SSZ)", "2022-12-15T21:45:30Z")
    entity_id = st.text_input("Entity ID", "123-45-6789")
    card_bin = st.text_input("Card BIN", "524301")
    billing_latitude = st.number_input("Billing Latitude", value=40.7128)
    billing_longitude = st.number_input("Billing Longitude", value=-74.0060)
    ip_address = st.text_input("IP Address", "203.0.113.1")
    product_category = st.text_input("Product Category", "electronics")
    
    # Order price slider
    order_price = st.slider("Order Price ($)", min_value=0.0, max_value=5000.0, value=899.99, step=0.01)
    
    merchant = st.text_input("Merchant", "unknown_Electronics Store")
    user_agent = st.text_input("User Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

    submit_button = st.form_submit_button("🚀 Predict Fraud")

if submit_button:
    sample_transaction = {
        'EVENT_TIMESTAMP': event_timestamp,
        'ENTITY_ID': entity_id,
        'card_bin': card_bin,
        'billing_latitude': billing_latitude,
        'billing_longitude': billing_longitude,
        'ip_address': ip_address,
        'product_category': product_category,
        'order_price': order_price,
        'merchant': merchant,
        'user_agent': user_agent
    }

    prediction = predict_fraud_hybrid(sample_transaction)

    st.write("### 🔮 Prediction Result")
    st.write(f"💰 Amount: ${sample_transaction['order_price']}, ⏰ Hour: {datetime.strptime(sample_transaction['EVENT_TIMESTAMP'], '%Y-%m-%dT%H:%M:%SZ').hour}")
    st.write(f"🚨 Prediction: **{prediction['prediction']}** (Confidence: {prediction['confidence']})")
    st.write(f"⚠️ Risk Level: **{prediction['risk_level']}**")
    st.write(f"📊 Model Scores - XGBoost: {prediction['xgb_score']}, Transformer: {prediction['transformer_score']}")
