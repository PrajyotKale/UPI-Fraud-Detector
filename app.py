import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
from datetime import datetime
from xgboost import XGBClassifier
import torch.nn as nn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the TransformerFraudModel class to match the saved model
class TransformerFraudModel(nn.Module):
    def __init__(self, input_dim, num_heads=2, dropout_rate=0.2, hidden_dim=32):
        super(TransformerFraudModel, self).__init__()
        
        # Initial dense layer
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        
        # Transformer block
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Initial processing
        x = self.activation(self.dense(x))
        x = x.unsqueeze(1)  # Add sequence dimension [batch, 1, hidden_dim]
        
        # Self-attention
        attn_output, _ = self.multihead_attention(x, x, x)
        x = self.layer_norm(x + attn_output)
        
        # Average pooling (since we have only one sequence element)
        x = x.squeeze(1)
        
        # Output layer
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

def load_hybrid_models(model_dir="./models"):
    """
    Load the hybrid fraud detection models and preprocessing objects
    
    Parameters:
    model_dir (str): Directory containing the saved models and preprocessing objects
    
    Returns:
    dict: Dictionary containing all loaded models and preprocessing objects
    """
    try:
        # Load XGBoost model
        xgb_model = XGBClassifier()
        xgb_model.load_model(f"{model_dir}/xgb_model.json")
        
        # Load preprocessing objects
        scaler = joblib.load(f"{model_dir}/scaler.pkl")
        label_encoders = joblib.load(f"{model_dir}/label_encoders.pkl")
        
        # Get the input dimensions for the transformer model
        # This assumes we saved the numerical_features alongside the models
        numerical_features_path = f"{model_dir}/numerical_features.pkl"
        if os.path.exists(numerical_features_path):
            numerical_features = joblib.load(numerical_features_path)
        else:
            # Default list if file doesn't exist
            numerical_features = ['order_price', 'transaction_hour', 'billing_latitude', 'billing_longitude', 'ip_first_octet']
        
        # Determine input dimension for the transformer
        input_dim = len(numerical_features)
        
        # Load PyTorch Transformer model
        transformer_model = TransformerFraudModel(input_dim=input_dim).to(device)
        transformer_model.load_state_dict(torch.load(
            f"{model_dir}/transformer_model.pt", 
            map_location=device
        ))
        transformer_model.eval()  # Set to evaluation mode
        
        # Load Meta model
        meta_model = XGBClassifier()
        meta_model.load_model(f"{model_dir}/meta_model.json")
        
        # Load feature list if available
        features_path = f"{model_dir}/features.pkl"
        if os.path.exists(features_path):
            features = joblib.load(features_path)
        else:
            # Get feature names directly from XGBoost model
            if hasattr(xgb_model, 'feature_names_'):
                features = xgb_model.feature_names_
            else:
                # Default comprehensive feature list based on error message
                features = [
                    'transaction_hour', 'merchant_type', 'is_night', 'is_business_hour', 
                    'amount_category', 'is_high_amount', 'ip_first_octet', 'browser_type', 
                    'order_price', 'ENTITY_TYPE', 'card_bin', 'customer_name', 'billing_city', 
                    'billing_state', 'billing_country', 'customer_job', 'product_category', 
                    'payment_currency', 'merchant_name', 'billing_latitude', 'billing_longitude'
                ]
            
        print(f"Loaded features: {features}")
        return {
            'xgb_model': xgb_model,
            'transformer_model': transformer_model,
            'meta_model': meta_model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'features': features,
            'numerical_features': numerical_features
        }
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

def predict_fraud_hybrid(transaction_data, hybrid_models):
    """
    Predict if a transaction is fraudulent using the hybrid model
    
    Parameters:
    transaction_data (dict): Transaction details with fields matching the training data
    hybrid_models (dict): Dictionary containing trained models and preprocessing objects
    
    Returns:
    dict: Prediction results with classification and confidence score
    """
    # Process input data
    processed_data = {}
    
    # Extract transaction hour from timestamp
    if 'EVENT_TIMESTAMP' in transaction_data:
        timestamp = transaction_data['EVENT_TIMESTAMP']
        dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
        processed_data['transaction_hour'] = dt.hour
    else:
        processed_data['transaction_hour'] = 12  # Default if not provided
    
    # Extract merchant type and name
    if 'merchant' in transaction_data:
        merchant = transaction_data['merchant']
        processed_data['merchant_type'] = 1 if merchant.lower().startswith('fraud') else 0
        processed_data['merchant_name'] = merchant.split('_', 1)[1] if '_' in merchant else merchant
    else:
        processed_data['merchant_type'] = 0
        processed_data['merchant_name'] = 'Unknown'
    
    # Process entity type
    processed_data['ENTITY_TYPE'] = transaction_data.get('ENTITY_TYPE', 'individual')
    
    # Handle categorical features
    for col, encoder in hybrid_models['label_encoders'].items():
        if col in transaction_data:
            try:
                processed_data[col] = encoder.transform([transaction_data[col]])[0]
            except (ValueError, KeyError):
                # Handle unseen values
                processed_data[col] = 0  # Default value for unknown categories
        else:
            processed_data[col] = 0  # Default value if not provided
    
    # Handle numerical and other features
    processed_data['order_price'] = float(transaction_data.get('order_price', 0))
    
    # Extract geolocation if available
    if 'billing_latitude' in transaction_data and 'billing_longitude' in transaction_data:
        processed_data['billing_latitude'] = float(transaction_data.get('billing_latitude', 0))
        processed_data['billing_longitude'] = float(transaction_data.get('billing_longitude', 0))
    else:
        processed_data['billing_latitude'] = 0.0
        processed_data['billing_longitude'] = 0.0
    
    # Extract IP information if available
    if 'ip_address' in transaction_data:
        ip = transaction_data['ip_address']
        processed_data['ip_first_octet'] = int(ip.split('.')[0]) if isinstance(ip, str) and '.' in ip else 0
    else:
        processed_data['ip_first_octet'] = 0
    
    # Extract browser type if available
    if 'user_agent' in transaction_data:
        user_agent = transaction_data['user_agent'].lower() if isinstance(transaction_data.get('user_agent'), str) else ''
        if 'chrome' in user_agent:
            processed_data['browser_type'] = 0
        elif 'firefox' in user_agent:
            processed_data['browser_type'] = 1
        elif 'safari' in user_agent:
            processed_data['browser_type'] = 2
        elif 'opera' in user_agent:
            processed_data['browser_type'] = 3
        elif 'edge' in user_agent:
            processed_data['browser_type'] = 4
        elif 'msie' in user_agent or 'trident' in user_agent:
            processed_data['browser_type'] = 5
        else:
            processed_data['browser_type'] = 6
    else:
        processed_data['browser_type'] = 0
    
    # Create engineered features
    processed_data['is_night'] = 1 if processed_data['transaction_hour'] >= 20 or processed_data['transaction_hour'] <= 5 else 0
    processed_data['is_business_hour'] = 1 if 9 <= processed_data['transaction_hour'] <= 18 else 0
    processed_data['amount_category'] = min(4, int(processed_data['order_price'] // 200))
    processed_data['is_high_amount'] = 1 if processed_data['order_price'] >= 800 else 0
    
    # Default values for other expected features that might be missing
    default_features = {
        'card_bin': 0,
        'customer_name': 0,
        'billing_city': 0,
        'billing_state': 0, 
        'billing_country': 0,
        'customer_job': 0,
        'product_category': 0,
        'payment_currency': 0
    }
    
    for feature, default_value in default_features.items():
        if feature not in processed_data:
            processed_data[feature] = default_value
    
    # Create input for XGBoost - ensure all required features are present
    features = hybrid_models['features']
    xgb_features = {}
    for feature in features:
        if feature in processed_data:
            xgb_features[feature] = processed_data[feature]
        else:
            xgb_features[feature] = 0  # Default value for missing features
    
    # Create DataFrame with expected column order
    xgb_input = pd.DataFrame([xgb_features])
    
    # Create input for Transformer - ensure all required numerical features are present
    numerical_features = hybrid_models['numerical_features']
    numerical_input = np.array([[processed_data.get(feature, 0) for feature in numerical_features]])
    numerical_input_scaled = hybrid_models['scaler'].transform(numerical_input)
    
    # Get predictions from both models
    xgb_pred = hybrid_models['xgb_model'].predict_proba(xgb_input)[:, 1][0]
    
    # PyTorch transformer prediction
    with torch.no_grad():
        transformer_input = torch.tensor(numerical_input_scaled, dtype=torch.float32).to(device)
        transformer_pred = hybrid_models['transformer_model'](transformer_input).cpu().numpy().flatten()[0]
    
    # Create meta-features
    meta_features = np.array([[
        xgb_pred,
        transformer_pred,
        processed_data['order_price'],
        processed_data['transaction_hour'],
        processed_data['is_night'],
        processed_data['is_high_amount']
    ]])
    
    # Final prediction
    final_pred_proba = hybrid_models['meta_model'].predict_proba(meta_features)[:, 1][0]
    final_pred = 1 if final_pred_proba >= 0.5 else 0
    
    # Prepare result
    result = {
        'prediction': "Fraudulent" if final_pred == 1 else "Legitimate",
        'confidence': final_pred_proba * 100,
        'confidence_str': f"{final_pred_proba * 100:.2f}%",
        'xgb_score': xgb_pred * 100,
        'xgb_score_str': f"{xgb_pred * 100:.2f}%",
        'transformer_score': transformer_pred * 100,
        'transformer_score_str': f"{transformer_pred * 100:.2f}%",
        'risk_level': "High" if final_pred_proba > 0.8 else "Medium" if final_pred_proba > 0.5 else "Low"
    }
    
    return result

# Streamlit app
def main():
    st.set_page_config(
        page_title="Fraud Detection System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Transaction Fraud Detection System")
    st.write("This application uses a hybrid machine learning model combining XGBoost and a PyTorch Transformer to detect fraudulent transactions.")
    
    # Check if models have been loaded
    if 'hybrid_models' not in st.session_state:
        try:
            with st.spinner("Loading models..."):
                st.session_state.hybrid_models = load_hybrid_models()
            st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Please make sure the model files exist in the correct directory.")
            return
    
    # Create sidebar for navigation
    page = st.sidebar.radio("Navigation", ["Single Prediction", "Batch Prediction"])
    
    if page == "Single Prediction":
        st.header("Evaluate Single Transaction")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction details form
            st.subheader("Transaction Details")
            timestamp = st.text_input("Timestamp (YYYY-MM-DDTHH:MM:SSZ)", "2023-04-17T14:30:00Z")
            entity_id = st.text_input("Entity ID", "123-45-6789")
            entity_type = st.selectbox("Entity Type", ["individual", "business", "other"])
            card_bin = st.text_input("Card BIN", "524301")
            merchant = st.text_input("Merchant", "Store_Electronics Store")
            
            # Additional details for XGBoost model
            customer_name = st.text_input("Customer Name", "John Doe")
            billing_city = st.text_input("Billing City", "New York")
            billing_state = st.text_input("Billing State", "NY")
            billing_country = st.text_input("Billing Country", "US")
            customer_job = st.text_input("Customer Job", "Engineer")
            
            product_category = st.selectbox(
                "Product Category",
                ["electronics", "clothing", "jewelry", "travel", "digital_goods", "home_goods", "other"]
            )
            
            payment_currency = st.selectbox("Payment Currency", ["USD", "EUR", "GBP", "CAD", "JPY", "Other"])
            order_price = st.number_input("Order Price ($)", min_value=0.0, max_value=10000.0, value=499.99)
            
        with col2:
            # More transaction details
            st.subheader("Additional Details")
            lat = st.number_input("Billing Latitude", min_value=-90.0, max_value=90.0, value=40.7128)
            lon = st.number_input("Billing Longitude", min_value=-180.0, max_value=180.0, value=-74.0060)
            ip_address = st.text_input("IP Address", "192.168.1.1")
            
            user_agents = {
                "Chrome": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
                "Firefox": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0",
                "Safari": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
                "Edge": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36 Edg/96.0.1054.62",
                "Opera": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36 OPR/82.0.4227.43"
            }
            
            browser = st.selectbox("Browser", list(user_agents.keys()))
            user_agent = user_agents[browser]
        
        # Submit button
        if st.button("Analyze Transaction"):
            transaction_data = {
                'EVENT_TIMESTAMP': timestamp,
                'ENTITY_ID': entity_id,
                'ENTITY_TYPE': entity_type,
                'card_bin': card_bin,
                'customer_name': customer_name,
                'billing_city': billing_city,
                'billing_state': billing_state,
                'billing_country': billing_country,
                'customer_job': customer_job,
                'billing_latitude': lat,
                'billing_longitude': lon,
                'ip_address': ip_address,
                'product_category': product_category,
                'payment_currency': payment_currency,
                'order_price': order_price,
                'merchant': merchant,
                'user_agent': user_agent
            }
            
            # Make prediction
            with st.spinner("Analyzing transaction..."):
                result = predict_fraud_hybrid(transaction_data, st.session_state.hybrid_models)
            
            # Display result
            st.subheader("Analysis Result")
            
            # Create risk indicator
            risk_color = "red" if result['prediction'] == "Fraudulent" else "green"
            
            # Display in three columns
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.markdown(f"### Prediction: <span style='color:{risk_color};'>{result['prediction']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {result['confidence_str']}")
                st.markdown(f"**Risk Level:** {result['risk_level']}")
                
            with res_col2:
                st.markdown("### Model Scores")
                st.markdown(f"**XGBoost:** {result['xgb_score_str']}")
                st.markdown(f"**Transformer:** {result['transformer_score_str']}")
                
            with res_col3:
                st.markdown("### Transaction Summary")
                st.markdown(f"**Amount:** ${order_price:.2f}")
                st.markdown(f"**Time:** {datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ').strftime('%H:%M')}")
                st.markdown(f"**Product:** {product_category}")
            
            # Progress bars for scores
            st.subheader("Confidence Scores")
            st.progress(result['confidence'] / 100)
            st.caption(f"Overall Confidence: {result['confidence_str']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.progress(result['xgb_score'] / 100)
                st.caption(f"XGBoost Score: {result['xgb_score_str']}")
            with col2:
                st.progress(result['transformer_score'] / 100)
                st.caption(f"Transformer Score: {result['transformer_score_str']}")
                
    else:  # Batch Prediction
        st.header("Batch Transaction Analysis")
        
        st.write("Upload a CSV file with transaction data for batch prediction.")
        st.write("The CSV should contain columns matching the expected transaction fields.")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            # Load the data
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} transactions")
                
                # Show sample data
                st.subheader("Sample Data")
                st.dataframe(df.head())
                
                # Batch prediction
                if st.button("Run Batch Analysis"):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, row in df.iterrows():
                        # Convert row to dict
                        transaction = row.to_dict()
                        
                        # Make prediction
                        result = predict_fraud_hybrid(transaction, st.session_state.hybrid_models)
                        
                        # Add result with transaction ID
                        results.append({
                            'Transaction ID': transaction.get('EVENT_ID', f"TX{i}"),
                            'Amount': transaction.get('order_price', 0),
                            'Prediction': result['prediction'],
                            'Confidence': result['confidence'],
                            'Risk Level': result['risk_level']
                        })
                        
                        # Update progress
                        progress_bar.progress((i + 1) // len(df))
                    
                    # Convert results to DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.subheader("Batch Analysis Results")
                    st.dataframe(results_df)
                    
                    # Calculate fraud stats
                    fraud_count = sum(1 for r in results if r['Prediction'] == 'Fraudulent')
                    fraud_percent = (fraud_count // len(results)) * 100
                    
                    # Display stats
                    st.subheader("Summary Statistics")
                    st.write(f"Total Transactions: {len(results)}")
                    st.write(f"Fraudulent Transactions: {fraud_count} ({fraud_percent:.2f}%)")
                    st.write(f"Legitimate Transactions: {len(results) - fraud_count} ({100 - fraud_percent:.2f}%)")
                    
                    # Download option
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv",
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
