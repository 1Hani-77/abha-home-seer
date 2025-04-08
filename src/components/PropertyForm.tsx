
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { BedIcon, BathIcon, AreaIcon, LocationIcon } from "@/components/Icons";

type PropertyData = {
  area: number;
  bedrooms: number;
  bathrooms: number;
  location: string;
  propertyType: string;
  yearBuilt: number;
};

interface PropertyFormProps {
  onPredict: (data: PropertyData) => void;
}

const PropertyForm: React.FC<PropertyFormProps> = ({ onPredict }) => {
  const [propertyData, setPropertyData] = useState<PropertyData>({
    area: 200,
    bedrooms: 3,
    bathrooms: 2,
    location: "Abha City Center",
    propertyType: "Apartment",
    yearBuilt: 2010,
  });

  const handleChange = (field: keyof PropertyData, value: string | number) => {
    setPropertyData({
      ...propertyData,
      [field]: value,
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPredict(propertyData);
  };

  const abhaLocations = [
    "Abha City Center",
    "Al Sad",
    "Al Numas",
    "Al Aziziyah",
    "Al Marooj",
    "Al Mansak",
    "Al Qabel",
    "Al Warood"
  ];

  const propertyTypes = [
    "Apartment",
    "Villa",
    "Duplex",
    "Townhouse",
    "Studio"
  ];

  return (
    <Card className="w-full">
      <CardHeader className="bg-estate-secondary text-white rounded-t-lg">
        <CardTitle className="text-xl">Property Details</CardTitle>
      </CardHeader>
      <CardContent className="pt-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center">
                <AreaIcon className="h-5 w-5 mr-2 text-estate-secondary" />
                <Label htmlFor="area">Area (sqm)</Label>
              </div>
              <div className="flex space-x-4 items-center">
                <Slider
                  id="area-slider"
                  min={50}
                  max={800}
                  step={10}
                  value={[propertyData.area]}
                  onValueChange={(value) => handleChange("area", value[0])}
                  className="flex-grow"
                />
                <Input
                  id="area"
                  type="number"
                  value={propertyData.area}
                  onChange={(e) => handleChange("area", parseInt(e.target.value))}
                  className="w-20"
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center">
                <LocationIcon className="h-5 w-5 mr-2 text-estate-secondary" />
                <Label htmlFor="location">Location</Label>
              </div>
              <Select
                value={propertyData.location}
                onValueChange={(value) => handleChange("location", value)}
              >
                <SelectTrigger id="location">
                  <SelectValue placeholder="Select Location" />
                </SelectTrigger>
                <SelectContent>
                  {abhaLocations.map((location) => (
                    <SelectItem key={location} value={location}>
                      {location}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <div className="flex items-center">
                <BedIcon className="h-5 w-5 mr-2 text-estate-secondary" />
                <Label htmlFor="bedrooms">Bedrooms</Label>
              </div>
              <div className="flex space-x-4 items-center">
                <Slider
                  id="bedroom-slider"
                  min={1}
                  max={8}
                  step={1}
                  value={[propertyData.bedrooms]}
                  onValueChange={(value) => handleChange("bedrooms", value[0])}
                  className="flex-grow"
                />
                <Input
                  id="bedrooms"
                  type="number"
                  value={propertyData.bedrooms}
                  onChange={(e) => handleChange("bedrooms", parseInt(e.target.value))}
                  className="w-20"
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center">
                <BathIcon className="h-5 w-5 mr-2 text-estate-secondary" />
                <Label htmlFor="bathrooms">Bathrooms</Label>
              </div>
              <div className="flex space-x-4 items-center">
                <Slider
                  id="bathroom-slider"
                  min={1}
                  max={8}
                  step={1}
                  value={[propertyData.bathrooms]}
                  onValueChange={(value) => handleChange("bathrooms", value[0])}
                  className="flex-grow"
                />
                <Input
                  id="bathrooms"
                  type="number"
                  value={propertyData.bathrooms}
                  onChange={(e) => handleChange("bathrooms", parseInt(e.target.value))}
                  className="w-20"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="propertyType">Property Type</Label>
              <Select
                value={propertyData.propertyType}
                onValueChange={(value) => handleChange("propertyType", value)}
              >
                <SelectTrigger id="propertyType">
                  <SelectValue placeholder="Select Type" />
                </SelectTrigger>
                <SelectContent>
                  {propertyTypes.map((type) => (
                    <SelectItem key={type} value={type}>
                      {type}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="yearBuilt">Year Built</Label>
              <Select
                value={propertyData.yearBuilt.toString()}
                onValueChange={(value) => handleChange("yearBuilt", parseInt(value))}
              >
                <SelectTrigger id="yearBuilt">
                  <SelectValue placeholder="Select Year" />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: 31 }, (_, i) => 2030 - i).map((year) => (
                    <SelectItem key={year} value={year.toString()}>
                      {year}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <Button 
            type="submit" 
            className="w-full bg-estate-primary hover:bg-estate-primary/90 mt-6"
          >
            Predict Price
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};

export default PropertyForm;
