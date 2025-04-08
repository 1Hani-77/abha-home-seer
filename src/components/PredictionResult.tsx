
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface PredictionResultProps {
  predictedPrice: number;
  confidence: number;
  similarProperties: Array<{
    price: number;
    area: number;
    bedrooms: number;
    location: string;
  }>;
}

const PredictionResult: React.FC<PredictionResultProps> = ({ 
  predictedPrice, 
  confidence, 
  similarProperties 
}) => {
  const formatCurrency = (value: number) => {
    return `${value.toLocaleString()} SAR`;
  };

  return (
    <Card className="w-full">
      <CardHeader className="bg-estate-secondary text-white rounded-t-lg">
        <CardTitle className="text-xl">Prediction Results</CardTitle>
      </CardHeader>
      <CardContent className="pt-6 space-y-6">
        <div className="text-center p-6 bg-estate-light rounded-lg">
          <h3 className="text-lg font-medium text-estate-dark mb-2">Estimated Property Value</h3>
          <p className="text-4xl font-bold text-estate-primary mb-2">{formatCurrency(predictedPrice)}</p>
          <p className="text-sm text-estate-secondary">
            Confidence: {confidence}% based on current market data
          </p>
        </div>

        <div>
          <h3 className="text-lg font-medium text-estate-dark mb-4">Comparable Properties</h3>
          <div className="space-y-4">
            {similarProperties.map((property, index) => (
              <div key={index} className="bg-gray-50 p-4 rounded-lg">
                <div className="flex justify-between">
                  <span className="font-medium">Property {index + 1}</span>
                  <span className="font-bold text-estate-primary">{formatCurrency(property.price)}</span>
                </div>
                <div className="text-sm text-gray-500 mt-1">
                  {property.area} sqm • {property.bedrooms} bedrooms • {property.location}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-estate-light rounded-lg p-4">
          <h3 className="text-lg font-medium text-estate-dark mb-2">What Affects This Prediction?</h3>
          <ul className="space-y-2 text-sm">
            <li className="flex items-start space-x-2">
              <span className="text-estate-secondary font-bold">•</span>
              <span>Location has a strong influence on property values in ABHA</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-estate-secondary font-bold">•</span>
              <span>Property size (area in sqm) correlates directly with price</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-estate-secondary font-bold">•</span>
              <span>Property features (bedrooms, bathrooms) impact valuation</span>
            </li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};

export default PredictionResult;
