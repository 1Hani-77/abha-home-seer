
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

# Set page configuration
st.set_page_config(
    page_title="ABHA HomeSeer - Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Define global variables for the model and data
model = None
df = None

# Function to load data
@st.cache_data
def load_data():
    # Load data from the provided GitHub URL
    url = "https://raw.githubusercontent.com/SLW-20/ProjectMIS/refs/heads/master/abha%20real%20estate.csv"
    data = pd.read_csv(url)
    return data

# Function to preprocess data and train model
def train_model(df):
    # Assume columns from the dataset - adjust these based on actual columns
    X = df.drop('price', axis=1)  # Assuming 'price' is the target column
    y = df['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create and train the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, X.columns.tolist(), mae, r2, y_test, y_pred

# Function to make predictions
def predict_price(model, features, input_data):
    # Convert input_data to DataFrame
    input_df = pd.DataFrame([input_data], columns=features)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Generate similar properties (simplified)
    similar_properties = []
    for _ in range(3):  # Reduced from 5 to 3
        variation = 0.9 + (np.random.random() * 0.2)
        similar_prop = {
            'price': round(prediction * variation),
            'area': input_data.get('area', 200),
            'bedrooms': input_data.get('bedrooms', 3),
            'location': input_data.get('location', 'Abha City Center')
        }
        similar_properties.append(similar_prop)
    
    result = {
        'predictedPrice': round(prediction),
        'confidence': round(90),  # Simplified confidence
        'similarProperties': similar_properties
    }
    
    return result

# API with Flask
api = Flask(__name__)
CORS(api)

@api.route('/predict', methods=['POST'])
def api_predict():
    data = request.json
    result = predict_price(model, features, data)
    return jsonify(result)

# Function to run the Flask API
def run_api():
    api.run(host='0.0.0.0', port=5000)

# Main Streamlit UI
def main():
    global model, df, features
    
    st.title("🏠 ABHA HomeSeer - Real Estate Price Predictor")
    st.markdown("This app predicts real estate prices in ABHA based on property features.")
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_data()
    
    # Show data exploration tabs
    tab1, tab2 = st.tabs(["Predict", "API Info"])
    
    with tab1:
        st.header("Predict Property Price")
        
        # Train model if not already trained
        if model is None:
            with st.spinner('Training model...'):
                model, features, mae, r2, y_test, y_pred = train_model(df)
                st.success('Model trained successfully!')
                
        # Input form for prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Property Details")
            
            area = st.slider("Area (sqm)", min_value=50, max_value=800, value=200, step=10)
            bedrooms = st.slider("Bedrooms", min_value=1, max_value=8, value=3, step=1)
            bathrooms = st.slider("Bathrooms", min_value=1, max_value=8, value=2, step=1)
            
            locations = ["Abha City Center", "Al Sad", "Al Numas", "Al Aziziyah", 
                         "Al Marooj", "Al Mansak", "Al Qabel", "Al Warood"]
            location = st.selectbox("Location", options=locations)
            
            property_types = ["Apartment", "Villa", "Duplex", "Townhouse", "Studio"]
            property_type = st.selectbox("Property Type", options=property_types)
            
            year_built = st.selectbox("Year Built", 
                                     options=list(range(2030, 1990, -1)), 
                                     index=10)
        
        with col2:
            input_data = {
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'location': location,
                'propertyType': property_type,
                'yearBuilt': year_built
            }
            
            st.subheader("Get Prediction")
            if st.button("Predict Price"):
                with st.spinner('Calculating property value...'):
                    result = predict_price(model, features, input_data)
                    
                    st.success(f"Estimated Property Value: {result['predictedPrice']:,} SAR")
                    st.info(f"Confidence: {result['confidence']}%")
                    
                    st.subheader("Comparable Properties")
                    similar_df = pd.DataFrame(result['similarProperties'])
                    st.dataframe(similar_df, hide_index=True)
    
    with tab2:
        st.header("API Information")
        st.markdown("""
        ### REST API Endpoints
        
        This app provides a REST API that can be used to get predictions programmatically:
        
        #### Predict Endpoint
        
        ```
        POST /predict
        ```
        
        **Request Body:**
        ```json
        {
            "area": 200,
            "bedrooms": 3,
            "bathrooms": 2,
            "location": "Abha City Center",
            "propertyType": "Apartment",
            "yearBuilt": 2010
        }
        ```
        
        The API server runs on port 5000.
        """)
        
        # API status
        st.subheader("API Status")
        if st.button("Start API Server"):
            api_thread = threading.Thread(target=run_api, daemon=True)
            api_thread.start()
            st.success("API server started on http://localhost:5000")

if __name__ == "__main__":
    main()
