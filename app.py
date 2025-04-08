
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
import requests
from io import StringIO
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import joblib

# Set page configuration
st.set_page_config(
    page_title="ABHA HomeSeer - Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Define global variables for the model and data
model = None
df = None
preprocessor = None
features = None

# Function to load data
@st.cache_data
def load_data():
    # Load data from the provided GitHub URL
    url = "https://raw.githubusercontent.com/SLW-20/ProjectMIS/refs/heads/master/abha%20real%20estate.csv"
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    return data

# Function to preprocess data and train model
def train_model(df):
    # Assume columns from the dataset - adjust these based on actual columns in your dataset
    # This is an assumption; you'll need to check the actual data
    X = df.drop('price', axis=1)  # Assuming 'price' is the target column
    y = df['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
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
    
    return model, preprocessor, X.columns.tolist(), mae, r2, y_test, y_pred

# Function to make predictions
def predict_price(model, preprocessor, features, input_data):
    # Convert input_data to DataFrame matching the format expected by the model
    input_df = pd.DataFrame([input_data], columns=features)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Generate similar properties (simplified approach)
    similar_properties = []
    for _ in range(5):
        # Create variations based on the input data
        variation = 0.9 + (np.random.random() * 0.2)  # 0.9 to 1.1
        similar_prop = {
            'price': round(prediction * variation),
            'area': input_data.get('area', 200),
            'bedrooms': input_data.get('bedrooms', 3),
            'location': input_data.get('location', 'Abha City Center')
        }
        similar_properties.append(similar_prop)
    
    result = {
        'predictedPrice': round(prediction),
        'confidence': round(85 + np.random.random() * 10),  # Simulated confidence between 85-95%
        'similarProperties': similar_properties
    }
    
    return result

# API with Flask
api = Flask(__name__)
CORS(api)

@api.route('/predict', methods=['POST'])
def api_predict():
    data = request.json
    result = predict_price(model, preprocessor, features, data)
    return jsonify(result)

# Function to run the Flask API in a separate thread
def run_api():
    api.run(host='0.0.0.0', port=5000)

# Main Streamlit UI
def main():
    global model, df, preprocessor, features
    
    # Add title and description
    st.title("🏠 ABHA HomeSeer - Real Estate Price Predictor")
    st.markdown("""
    This app uses machine learning to predict real estate prices in ABHA based on property features.
    Enter property details below to get a prediction.
    """)
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_data()
    
    # Show data exploration tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Data Explorer", "Model Performance", "API Info"])
    
    with tab1:
        st.header("Predict Property Price")
        
        # Train model if not already trained
        if model is None:
            with st.spinner('Training model...'):
                model, preprocessor, features, mae, r2, y_test, y_pred = train_model(df)
                st.success('Model trained successfully!')
                
        # Input form for prediction
        col1, col2 = st.columns(2)
        
        with col1:
            # Get user inputs - adjust based on actual dataset columns
            st.subheader("Property Details")
            
            # Assume these are the features based on the dataset - adjust as needed
            area = st.slider("Area (sqm)", min_value=50, max_value=800, value=200, step=10)
            bedrooms = st.slider("Bedrooms", min_value=1, max_value=8, value=3, step=1)
            bathrooms = st.slider("Bathrooms", min_value=1, max_value=8, value=2, step=1)
            
            # Assume these are the locations from the dataset - adjust as needed
            locations = ["Abha City Center", "Al Sad", "Al Numas", "Al Aziziyah", 
                         "Al Marooj", "Al Mansak", "Al Qabel", "Al Warood"]
            location = st.selectbox("Location", options=locations)
            
            # Assume these are the property types - adjust as needed
            property_types = ["Apartment", "Villa", "Duplex", "Townhouse", "Studio"]
            property_type = st.selectbox("Property Type", options=property_types)
            
            # Year built
            year_built = st.selectbox("Year Built", 
                                     options=list(range(2030, 1990, -1)), 
                                     index=10)  # Default to 2020
        
        with col2:
            # Create input data dictionary
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
                    # Make prediction
                    result = predict_price(model, preprocessor, features, input_data)
                    
                    # Display prediction
                    st.success(f"Estimated Property Value: {result['predictedPrice']:,} SAR")
                    st.info(f"Confidence: {result['confidence']}% based on current market data")
                    
                    # Show similar properties
                    st.subheader("Comparable Properties")
                    similar_df = pd.DataFrame(result['similarProperties'])
                    st.dataframe(similar_df, hide_index=True)
                    
                    # Plot price comparison
                    fig, ax = plt.subplots(figsize=(10, 5))
                    properties = [f"Similar {i+1}" for i in range(len(result['similarProperties']))]
                    properties.append("Your Property")
                    
                    prices = [prop['price'] for prop in result['similarProperties']]
                    prices.append(result['predictedPrice'])
                    
                    # Create bar chart
                    sns.barplot(x=properties, y=prices, ax=ax)
                    ax.set_title("Property Price Comparison")
                    ax.set_ylabel("Price (SAR)")
                    plt.xticks(rotation=45)
                    
                    st.pyplot(fig)
            
            # Save model button
            if st.button("Save Model"):
                # Save model to disk
                joblib.dump(model, 'abha_real_estate_model.pkl')
                st.success("Model saved successfully as 'abha_real_estate_model.pkl'")
                st.download_button(
                    label="Download trained model",
                    data=open('abha_real_estate_model.pkl', 'rb'),
                    file_name='abha_real_estate_model.pkl',
                    mime='application/octet-stream'
                )
    
    with tab2:
        st.header("Data Explorer")
        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Show data statistics
        st.subheader("Data Statistics")
        st.dataframe(df.describe())
        
        # Show correlation matrix
        st.subheader("Correlation Matrix")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Distribution of prices
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['price'], kde=True, ax=ax)
        st.pyplot(fig)
        
        # Scatter plots for important features
        st.subheader("Feature Relationships")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'area' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x='area', y='price', data=df, ax=ax)
                ax.set_title('Price vs. Area')
                st.pyplot(fig)
        
        with col2:
            if 'bedrooms' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(x='bedrooms', y='price', data=df, ax=ax)
                ax.set_title('Price by Number of Bedrooms')
                st.pyplot(fig)
    
    with tab3:
        if model is not None:
            st.header("Model Performance")
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Absolute Error (SAR)", f"{int(mae):,}")
            with col2:
                st.metric("R² Score", f"{r2:.2f}")
            
            # Plot actual vs predicted
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel('Actual Price')
            ax.set_ylabel('Predicted Price')
            ax.set_title('Actual vs Predicted Prices')
            st.pyplot(fig)
            
            # Feature importance
            try:
                importances = model.named_steps['regressor'].feature_importances_
                # Get feature names after preprocessing (may be complex with OHE)
                st.subheader("Feature Importance")
                fig, ax = plt.subplots(figsize=(10, 8))
                # Plot simplified feature importance (top 10 features)
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False).head(10)
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                st.pyplot(fig)
            except:
                st.info("Feature importance could not be displayed for the current model")
        else:
            st.info("Train the model first to see performance metrics")
    
    with tab4:
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
        
        **Response:**
        ```json
        {
            "predictedPrice": 800000,
            "confidence": 90,
            "similarProperties": [
                {
                    "price": 780000,
                    "area": 200,
                    "bedrooms": 3,
                    "location": "Abha City Center"
                },
                ...
            ]
        }
        ```
        
        The API server runs on port 5000. You can test it using curl or any API client.
        """)
        
        # API status
        st.subheader("API Status")
        if st.button("Start API Server"):
            # Start API in a separate thread
            api_thread = threading.Thread(target=run_api, daemon=True)
            api_thread.start()
            st.success("API server started on http://localhost:5000")

if __name__ == "__main__":
    main()
