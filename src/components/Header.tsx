
import React from 'react';
import { Button } from "@/components/ui/button";
import { HomeIcon, GitHubIcon } from "@/components/Icons";

const Header = () => {
  return (
    <header className="bg-estate-primary text-white py-4 px-6 shadow-md">
      <div className="container mx-auto flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <HomeIcon className="h-6 w-6" />
          <h1 className="text-xl font-bold">ABHA HomeSeer</h1>
        </div>
        <a href="https://github.com/your-username/abha-homeseer" target="_blank" rel="noopener noreferrer">
          <Button variant="outline" className="border-white text-white hover:bg-white hover:text-estate-primary">
            <GitHubIcon className="h-5 w-5 mr-2" />
            <span>View Python ML Model</span>
          </Button>
        </a>
      </div>
    </header>
  );
};

export default Header;
