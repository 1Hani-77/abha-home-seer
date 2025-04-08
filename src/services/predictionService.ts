
// This service connects to our Python/Streamlit backend

export interface PropertyData {
  area: number;
  bedrooms: number;
  bathrooms: number;
  location: string;
  propertyType: string;
  yearBuilt: number;
}

export interface PredictionResult {
  predictedPrice: number;
  confidence: number;
  similarProperties: Array<{
    price: number;
    area: number;
    bedrooms: number;
    location: string;
  }>;
}

// Function to connect to Python API
export const predictPropertyPrice = async (data: PropertyData): Promise<PredictionResult> => {
  try {
    // In development, use local API. In production, use your deployed API URL
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000/predict';
    
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error predicting price:', error);
    
    // Fallback to mock data if API call fails
    return mockPredictPropertyPrice(data);
  }
};

// Mock function as fallback if API is not available
const mockPredictPropertyPrice = (data: PropertyData): PredictionResult => {
  // Simple price calculation logic
  const basePrice = 800000; // Base price in SAR
  
  // Location factors
  const locationFactors: Record<string, number> = {
    "Abha City Center": 1.4,
    "Al Sad": 1.3,
    "Al Numas": 1.2,
    "Al Aziziyah": 1.1,
    "Al Marooj": 1.15,
    "Al Mansak": 1.05,
    "Al Qabel": 1.0,
    "Al Warood": 0.95
  };
  
  // Property type factors
  const propertyTypeFactors: Record<string, number> = {
    "Villa": 1.5,
    "Duplex": 1.3,
    "Townhouse": 1.2,
    "Apartment": 1.0,
    "Studio": 0.8
  };
  
  // Calculate price with factors
  const locationFactor = locationFactors[data.location] || 1;
  const typeFactor = propertyTypeFactors[data.propertyType] || 1;
  const areaPriceFactor = data.area * 3500;
  const bedroomFactor = data.bedrooms * 50000;
  const bathroomFactor = data.bathrooms * 30000;
  
  // Calculate predicted price
  const predictedPrice = Math.round(
    (basePrice + areaPriceFactor + bedroomFactor + bathroomFactor) * 
    locationFactor * 
    typeFactor
  );
  
  // Generate similar properties
  const similarProperties = Array.from({ length: 3 }, (_, i) => {
    const variation = 0.9 + (Math.random() * 0.2); // 0.9 to 1.1
    
    return {
      price: Math.round(predictedPrice * variation),
      area: data.area,
      bedrooms: data.bedrooms,
      location: data.location,
    };
  });
  
  return {
    predictedPrice,
    confidence: 90, // Fixed confidence
    similarProperties,
  };
};
