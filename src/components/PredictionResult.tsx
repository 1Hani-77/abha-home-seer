
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';

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
  const chartData = similarProperties.map((property, index) => ({
    name: `Property ${index + 1}`,
    price: property.price,
    area: property.area,
  }));

  // Add the predicted price to the chart data
  chartData.push({
    name: 'Your Property',
    price: predictedPrice,
    area: similarProperties[0]?.area || 200,
  });

  // Custom styles for the chart
  const yourPropertyIndex = chartData.length - 1;
  
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

        <div className="w-full">
          <h3 className="text-lg font-medium text-estate-dark mb-4">Comparable Properties</h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={chartData}
                margin={{
                  top: 10,
                  right: 30,
                  left: 0,
                  bottom: 0,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis tickFormatter={(value) => `${(value / 1000)}k`} />
                <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                <Area 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#1a365d" 
                  fill="#63b3ed" 
                  fillOpacity={0.8}
                />
              </AreaChart>
            </ResponsiveContainer>
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
              <span>Recent sales of similar properties in the neighborhood</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-estate-secondary font-bold">•</span>
              <span>Property features (bedrooms, bathrooms, etc.) impact valuation</span>
            </li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};

export default PredictionResult;
