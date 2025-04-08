
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const MachineLearningInfo = () => {
  return (
    <Card className="w-full">
      <CardHeader className="bg-estate-secondary text-white rounded-t-lg">
        <CardTitle className="text-xl">Our Machine Learning Model</CardTitle>
      </CardHeader>
      <CardContent className="pt-6">
        <Tabs defaultValue="how-it-works">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="how-it-works">How It Works</TabsTrigger>
            <TabsTrigger value="features">Features Used</TabsTrigger>
            <TabsTrigger value="accuracy">Model Accuracy</TabsTrigger>
          </TabsList>
          <TabsContent value="how-it-works" className="mt-6 p-4 bg-estate-light rounded-lg">
            <h3 className="text-estate-primary font-semibold text-lg mb-3">Machine Learning Pipeline</h3>
            <p className="mb-4">
              Our property price prediction model uses advanced machine learning techniques to analyze historical 
              real estate data from ABHA and predict current market values with high accuracy.
            </p>
            <ol className="space-y-3 list-decimal list-inside">
              <li>
                <span className="font-medium">Data Collection:</span> We gather historical property data from ABHA, including sale prices, property features, and location data.
              </li>
              <li>
                <span className="font-medium">Data Preprocessing:</span> Raw data is cleaned, normalized, and prepared for training.
              </li>
              <li>
                <span className="font-medium">Feature Engineering:</span> We identify and create the most relevant features for accurate prediction.
              </li>
              <li>
                <span className="font-medium">Model Training:</span> Several ML algorithms are trained and evaluated (Random Forest, XGBoost, etc.).
              </li>
              <li>
                <span className="font-medium">Hyperparameter Tuning:</span> Model parameters are optimized for best performance.
              </li>
              <li>
                <span className="font-medium">Evaluation:</span> Models are tested against real market data to ensure accuracy.
              </li>
            </ol>
          </TabsContent>
          <TabsContent value="features" className="mt-6 p-4 bg-estate-light rounded-lg">
            <h3 className="text-estate-primary font-semibold text-lg mb-3">Key Prediction Features</h3>
            <p className="mb-4">
              Our model analyzes various property characteristics to generate accurate price predictions. Here are the most important features:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white p-3 rounded shadow-sm">
                <h4 className="text-estate-secondary font-medium">Location</h4>
                <p className="text-sm">Specific neighborhoods within ABHA have different value profiles</p>
              </div>
              <div className="bg-white p-3 rounded shadow-sm">
                <h4 className="text-estate-secondary font-medium">Property Size</h4>
                <p className="text-sm">Total area in square meters is heavily weighted in the model</p>
              </div>
              <div className="bg-white p-3 rounded shadow-sm">
                <h4 className="text-estate-secondary font-medium">Rooms Configuration</h4>
                <p className="text-sm">Number of bedrooms and bathrooms affect property valuation</p>
              </div>
              <div className="bg-white p-3 rounded shadow-sm">
                <h4 className="text-estate-secondary font-medium">Property Type</h4>
                <p className="text-sm">Different types (apartment, villa, etc.) have distinct pricing patterns</p>
              </div>
              <div className="bg-white p-3 rounded shadow-sm">
                <h4 className="text-estate-secondary font-medium">Age of Building</h4>
                <p className="text-sm">Year built contributes to the depreciation calculation</p>
              </div>
              <div className="bg-white p-3 rounded shadow-sm">
                <h4 className="text-estate-secondary font-medium">Proximity Factors</h4>
                <p className="text-sm">Distance to amenities, schools, and city center</p>
              </div>
            </div>
          </TabsContent>
          <TabsContent value="accuracy" className="mt-6 p-4 bg-estate-light rounded-lg">
            <h3 className="text-estate-primary font-semibold text-lg mb-3">Model Performance</h3>
            <p className="mb-4">
              Our predictive model has been rigorously tested and validated using real ABHA real estate data. Here's how it performs:
            </p>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="font-medium">Mean Absolute Error (MAE)</span>
                  <span className="text-estate-secondary">45,320 SAR</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div className="bg-estate-secondary h-2.5 rounded-full" style={{ width: '85%' }}></div>
                </div>
                <p className="text-xs mt-1 text-gray-500">Lower is better - average error in price prediction</p>
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span className="font-medium">R² Score</span>
                  <span className="text-estate-secondary">0.87</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div className="bg-estate-secondary h-2.5 rounded-full" style={{ width: '87%' }}></div>
                </div>
                <p className="text-xs mt-1 text-gray-500">Higher is better - model explains 87% of price variation</p>
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span className="font-medium">Confidence Within 10%</span>
                  <span className="text-estate-secondary">84%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div className="bg-estate-secondary h-2.5 rounded-full" style={{ width: '84%' }}></div>
                </div>
                <p className="text-xs mt-1 text-gray-500">84% of predictions are within 10% of actual sale price</p>
              </div>
              
              <div className="p-3 bg-white rounded-lg mt-4">
                <p className="text-sm">
                  <span className="font-medium">Note:</span> Our model is continuously improving as more data becomes available. 
                  The prediction system is retrained quarterly with the latest market data to maintain accuracy.
                </p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default MachineLearningInfo;
