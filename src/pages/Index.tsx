
import React, { useState } from 'react';
import Header from "@/components/Header";
import Hero from "@/components/Hero";
import PropertyForm from "@/components/PropertyForm";
import PredictionResult from "@/components/PredictionResult";
import MachineLearningInfo from "@/components/MachineLearningInfo";
import Footer from "@/components/Footer";
import { PropertyData, PredictionResult as PredictionResultType, predictPropertyPrice } from "@/services/predictionService";
import { useToast } from "@/components/ui/use-toast";

const Index = () => {
  const [predictionResult, setPredictionResult] = useState<PredictionResultType | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handlePredict = async (data: PropertyData) => {
    setIsLoading(true);
    try {
      // In a real app, this would call your Python/Streamlit ML backend
      const result = await predictPropertyPrice(data);
      setPredictionResult(result);
      toast({
        title: "Prediction Complete",
        description: "Your property valuation has been calculated successfully.",
      });
    } catch (error) {
      console.error("Prediction error:", error);
      toast({
        variant: "destructive",
        title: "Prediction Failed",
        description: "Unable to calculate property valuation. Please try again.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Header />
      <Hero />
      
      <main className="flex-grow">
        <section className="py-12 bg-white">
          <div className="container mx-auto px-4">
            <div className="text-center mb-10">
              <h2 className="text-3xl font-bold text-estate-primary mb-4">Get Your Property Valuation</h2>
              <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                Enter your property details below and our machine learning model will predict its market value based on current ABHA real estate data.
              </p>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className={`space-y-8 ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}>
                <PropertyForm onPredict={handlePredict} />
                <MachineLearningInfo />
              </div>
              
              <div>
                {isLoading ? (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-estate-secondary mx-auto mb-4"></div>
                      <p className="text-lg font-medium text-estate-primary">Calculating property value...</p>
                      <p className="text-gray-500">Our ML model is analyzing market data</p>
                    </div>
                  </div>
                ) : predictionResult ? (
                  <PredictionResult 
                    predictedPrice={predictionResult.predictedPrice}
                    confidence={predictionResult.confidence}
                    similarProperties={predictionResult.similarProperties}
                  />
                ) : (
                  <div className="h-full flex items-center justify-center bg-gray-100 rounded-lg p-8">
                    <div className="text-center">
                      <div className="bg-estate-light p-6 rounded-full inline-block mb-4">
                        <svg 
                          xmlns="http://www.w3.org/2000/svg" 
                          className="h-12 w-12 text-estate-secondary"
                          fill="none" 
                          viewBox="0 0 24 24" 
                          stroke="currentColor" 
                          strokeWidth={2}
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                        </svg>
                      </div>
                      <h3 className="text-xl font-medium text-estate-primary mb-2">Ready for Your Prediction</h3>
                      <p className="text-gray-600 max-w-md">
                        Fill in your property details on the left to get an accurate valuation using our machine learning model.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
        
        <section className="py-12 bg-estate-light">
          <div className="container mx-auto px-4 text-center">
            <h2 className="text-3xl font-bold text-estate-primary mb-6">How to Use ABHA HomeSeer</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="w-12 h-12 bg-estate-primary text-white rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">1</div>
                <h3 className="text-lg font-medium text-estate-primary mb-2">Enter Property Details</h3>
                <p className="text-gray-600">
                  Provide information about the property including size, location, number of rooms, and more.
                </p>
              </div>
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="w-12 h-12 bg-estate-secondary text-white rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">2</div>
                <h3 className="text-lg font-medium text-estate-primary mb-2">Get AI Prediction</h3>
                <p className="text-gray-600">
                  Our machine learning model analyzes the data and generates an accurate price prediction.
                </p>
              </div>
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="w-12 h-12 bg-estate-primary text-white rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">3</div>
                <h3 className="text-lg font-medium text-estate-primary mb-2">Make Informed Decisions</h3>
                <p className="text-gray-600">
                  Use the valuation results and comparative market analysis for your real estate decisions.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>
      
      <Footer />
    </div>
  );
};

export default Index;
