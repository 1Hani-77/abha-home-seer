import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Abha Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("Abha Real Estate Price Predictor")
st.markdown("### Predict property prices in Abha, Saudi Arabia using machine learning")

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Data Exploration", "Price Prediction", "About"])

# Load sample data (in a real app, you would load actual Abha real estate data)
@st.cache_data
def load_data():
    # This is placeholder data - in a real application you would:
    # 1. Collect real data from Abha real estate market
    # 2. Clean and preprocess it appropriately
    
    # Creating synthetic data for demonstration
    np.random.seed(42)
    n_samples = 500
    
    # Features that might affect real estate prices in Abha
    data = {
        'area_sqm': np.random.normal(250, 100, n_samples),  # Property area in square meters
        'bedrooms': np.random.randint(1, 6, n_samples),  # Number of bedrooms
        'bathrooms': np.random.randint(1, 5, n_samples),  # Number of bathrooms
        'age_years': np.random.randint(0, 30, n_samples),  # Age of property in years
        'distance_to_city_center_km': np.random.uniform(0, 15, n_samples),  # Distance to city center
        'has_garden': np.random.randint(0, 2, n_samples),  # Whether property has a garden
        'has_parking': np.random.randint(0, 2, n_samples),  # Whether property has parking
        'neighborhood_rating': np.random.uniform(1, 10, n_samples),  # Neighborhood rating
        'elevation_m': np.random.uniform(2000, 2500, n_samples),  # Elevation (Abha is at high altitude)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate price based on features (simplified model for demonstration)
    price = (
        100000 +  # Base price
        (df['area_sqm'] * 1000) +  # Area has significant impact
        (df['bedrooms'] * 50000) +  # Each bedroom adds value
        (df['bathrooms'] * 30000) +  # Each bathroom adds value
        (df['has_garden'] * 70000) +  # Garden adds premium
        (df['has_parking'] * 40000) +  # Parking adds value
        (df['neighborhood_rating'] * 20000) -  # Better neighborhoods cost more
        (df['age_years'] * 5000) -  # Older properties are cheaper
        (df['distance_to_city_center_km'] * 10000) +  # Properties closer to center are more expensive
        (np.random.normal(0, 50000, n_samples))  # Random noise
    )
    
    df['price_sar'] = price  # Price in Saudi Riyals
    
    # Ensure all prices are positive
    df['price_sar'] = df['price_sar'].clip(lower=100000)
    
    return df

data = load_data()

# HOME PAGE
if page == "Home":
    st.header("Welcome to the Abha Real Estate Price Predictor")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        This application helps predict real estate prices in Abha, Saudi Arabia using machine learning algorithms.
        
        ### Features:
        - Data exploration and visualization
        - Property price prediction based on key features
        - Interactive map of Abha properties (coming soon)
        
        ### How to use:
        1. Explore the data to understand market trends
        2. Navigate to the Price Prediction page
        3. Enter property details to get a price estimate
        
        **Note:** This is a demonstration using synthetic data. For accurate predictions, the model would need to be trained on actual Abha real estate data.
        """)
    
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Abha_City.jpg/1280px-Abha_City.jpg", caption="Abha City, Saudi Arabia")
    
    # Display a sample of the data
    st.subheader("Sample Property Data")
    st.dataframe(data.sample(5))
    
    # Quick statistics
    st.subheader("Market Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Price", f"{data['price_sar'].mean():,.0f} SAR")
    with col2:
        st.metric("Price Range", f"{data['price_sar'].min():,.0f} - {data['price_sar'].max():,.0f} SAR")
    with col3:
        st.metric("Total Properties", f"{len(data):,}")

# DATA EXPLORATION PAGE
elif page == "Data Exploration":
    st.header("Data Exploration and Visualization")
    
    # Data overview
    if st.checkbox("Show Dataset"):
        st.dataframe(data)
    
    # Summary statistics
    if st.checkbox("Show Summary Statistics"):
        st.write(data.describe())
    
    # Price distribution
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['price_sar'], bins=30, kde=True, ax=ax)
    plt.title("Distribution of Property Prices in Abha")
    plt.xlabel("Price (SAR)")
    plt.ylabel("Count")
    st.pyplot(fig)
    
    # Feature correlations
    st.subheader("Feature Correlation with Price")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation = data.corr()['price_sar'].sort_values(ascending=False)
    sns.barplot(x=correlation.values[1:], y=correlation.index[1:], ax=ax)
    plt.title("Correlation of Features with Price")
    plt.xlabel("Correlation Coefficient")
    st.pyplot(fig)
    
    # Feature exploration
    st.subheader("Explore Individual Features")
    feature = st.selectbox("Select Feature to Explore", data.columns.drop('price_sar'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of selected feature
        fig, ax = plt.subplots(figsize=(8, 5))
        if data[feature].dtype == 'int64' and data[feature].nunique() < 10:
            sns.countplot(data=data, x=feature, ax=ax)
        else:
            sns.histplot(data=data, x=feature, bins=20, kde=True, ax=ax)
        plt.title(f"Distribution of {feature}")
        st.pyplot(fig)
    
    with col2:
        # Relationship with price
        fig, ax = plt.subplots(figsize=(8, 5))
        if data[feature].nunique() < 10:
            sns.boxplot(data=data, x=feature, y='price_sar', ax=ax)
        else:
            sns.scatterplot(data=data, x=feature, y='price_sar', alpha=0.6, ax=ax)
        plt.title(f"Relationship between {feature} and Price")
        plt.ylabel("Price (SAR)")
        st.pyplot(fig)

# PRICE PREDICTION PAGE
elif page == "Price Prediction":
    st.header("Real Estate Price Prediction")
    
    # Sidebar for model selection
    model_type = st.sidebar.radio(
        "Select Model", 
        ["Linear Regression", "Random Forest Regressor"]
    )
    
    # Prepare data for modeling
    X = data.drop('price_sar', axis=1)
    y = data['price_sar']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model based on selection
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        model_name = "Linear Regression"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)  # RF doesn't require scaling
        model_name = "Random Forest"
    
    # Model evaluation
    if model_type == "Linear Regression":
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Display model performance
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", model_name)
    with col2:
        st.metric("RMSE", f"{rmse:,.0f} SAR")
    with col3:
        st.metric("R² Score", f"{r2:.3f}")
    
    # Feature importance (for Random Forest)
    if model_type == "Random Forest":
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
        plt.title("Feature Importance for Property Price")
        st.pyplot(fig)
    
    # Interactive prediction
    st.subheader("Predict Property Price")
    st.markdown("Enter property details to get a price estimate:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("Area (square meters)", min_value=50, max_value=1000, value=250)
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
        age = st.number_input("Property Age (years)", min_value=0, max_value=50, value=5)
        distance = st.number_input("Distance to City Center (km)", min_value=0.0, max_value=30.0, value=5.0)
    
    with col2:
        garden = st.checkbox("Has Garden", value=True)
        parking = st.checkbox("Has Parking", value=True)
        neighborhood = st.slider("Neighborhood Rating (1-10)", min_value=1.0, max_value=10.0, value=7.0, step=0.5)
        elevation = st.slider("Elevation (meters)", min_value=2000.0, max_value=2500.0, value=2200.0, step=10.0)
    
    # Create input for prediction
    input_data = pd.DataFrame({
        'area_sqm': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'age_years': [age],
        'distance_to_city_center_km': [distance],
        'has_garden': [int(garden)],
        'has_parking': [int(parking)],
        'neighborhood_rating': [neighborhood],
        'elevation_m': [elevation]
    })
    
    # Make prediction
    if st.button("Predict Price"):
        if model_type == "Linear Regression":
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
        else:
            prediction = model.predict(input_data)[0]
        
        st.success(f"Estimated Property Price: **{prediction:,.0f} SAR**")
        
        # Show a breakdown of the prediction (simplified)
        st.subheader("Price Component Breakdown")
        components = pd.DataFrame({
            'Feature': ['Base value'] + input_data.columns.tolist(),
            'Contribution': [100000] + [0] * len(input_data.columns)  # Placeholder values
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Contribution', y='Feature', data=components, ax=ax)
        plt.title("Component Contribution to Price (Illustrative)")
        st.pyplot(fig)
        
        # Comparative properties
        st.subheader("Similar Properties")
        # Find properties with similar features
        similar_props = data.copy()
        for col in input_data.columns:
            if col in ['area_sqm', 'distance_to_city_center_km', 'neighborhood_rating', 'elevation_m']:
                similar_props = similar_props[
                    (similar_props[col] >= input_data[col].values[0] * 0.8) & 
                    (similar_props[col] <= input_data[col].values[0] * 1.2)
                ]
            elif col in ['bedrooms', 'bathrooms', 'has_garden', 'has_parking']:
                similar_props = similar_props[similar_props[col] == input_data[col].values[0]]
        
        st.dataframe(similar_props.head(5).reset_index(drop=True))

# ABOUT PAGE
elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ## Abha Real Estate Price Predictor
    
    This application is a demonstration of how machine learning can be applied to predict real estate prices in Abha, Saudi Arabia.
    
    ### Project Goals:
    - Develop an accurate predictive model for Abha's unique real estate market
    - Provide useful insights into price determinants
    - Create an intuitive tool for buyers, sellers, and real estate professionals
    
    ### Data Sources:
    For a production version, data would be sourced from:
    - Local real estate listings and transactions
    - Government property records
    - Economic indicators for the Asir region
    - Geographic and environmental data specific to Abha
    
    ### Methodology:
    1. Data collection and cleaning
    2. Feature engineering relevant to Abha's market
    3. Model development and validation
    4. Continuous improvement with new data
    
    ### Next Steps:
    - Collect actual Abha real estate data
    - Refine machine learning models
    - Add geospatial visualization
    - Develop market trend analysis
    
    ### Contact:
    For questions or suggestions about this project, please contact [Your Contact Information].
    """)
    
    # Add a map of Abha (placeholder)
    st.subheader("Abha Location")
    st.map(pd.DataFrame({
        'lat': [18.2164],
        'lon': [42.5053]
    }))

# Footer
st.markdown("""
---
Developed with ❤️ using Streamlit | © 2023 Abha Real Estate Price Predictor
""")
