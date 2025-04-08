
import React from 'react';
import { BuildingIcon, ChartIcon } from './Icons';

const Hero = () => {
  return (
    <div className="bg-gradient-to-r from-estate-primary to-estate-secondary text-white py-16">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row items-center justify-between">
          <div className="md:w-1/2 mb-8 md:mb-0">
            <h1 className="text-4xl md:text-5xl font-bold mb-4 leading-tight">
              ABHA HomeSeer
            </h1>
            <h2 className="text-2xl md:text-3xl font-semibold mb-6 text-estate-accent">
              AI-Powered Real Estate Price Prediction
            </h2>
            <p className="text-lg mb-8">
              Get accurate property valuations in ABHA using our cutting-edge machine learning model. 
              Make informed decisions for buying or selling real estate with confidence.
            </p>
            <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4">
              <div className="flex items-center bg-white/10 rounded-lg p-4">
                <ChartIcon className="h-10 w-10 text-estate-accent mr-3" />
                <div>
                  <h3 className="font-semibold">Data-Driven</h3>
                  <p className="text-sm opacity-90">Powered by ML algorithms</p>
                </div>
              </div>
              <div className="flex items-center bg-white/10 rounded-lg p-4">
                <BuildingIcon className="h-10 w-10 text-estate-accent mr-3" />
                <div>
                  <h3 className="font-semibold">Local Insight</h3>
                  <p className="text-sm opacity-90">ABHA market expertise</p>
                </div>
              </div>
            </div>
          </div>
          <div className="md:w-2/5 relative">
            <div className="bg-white rounded-lg shadow-xl p-6 md:p-8 transform rotate-3 animate-float">
              <div className="bg-estate-light p-4 rounded-md mb-4">
                <div className="flex justify-between items-center mb-2">
                  <div className="font-medium text-estate-primary">Property Value</div>
                  <div className="text-estate-secondary font-bold">1,450,000 SAR</div>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div className="bg-estate-secondary h-2.5 rounded-full" style={{ width: '70%' }}></div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4 text-gray-600">
                <div className="text-sm">
                  <div className="font-medium text-estate-primary">Area</div>
                  <div>320 sqm</div>
                </div>
                <div className="text-sm">
                  <div className="font-medium text-estate-primary">Bedrooms</div>
                  <div>4</div>
                </div>
                <div className="text-sm">
                  <div className="font-medium text-estate-primary">Location</div>
                  <div>Al Aziziyah</div>
                </div>
                <div className="text-sm">
                  <div className="font-medium text-estate-primary">Built in</div>
                  <div>2015</div>
                </div>
              </div>
            </div>
            <div className="absolute -bottom-6 -left-6 bg-estate-accent/20 w-32 h-32 rounded-full -z-10"></div>
            <div className="absolute -top-6 -right-6 bg-estate-accent/20 w-24 h-24 rounded-full -z-10"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
