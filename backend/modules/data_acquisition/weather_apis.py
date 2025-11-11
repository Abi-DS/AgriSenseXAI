"""
Module 1: Data Acquisition Module - Weather APIs
Implementation Status: ~70% complete
Technologies: Python, FastAPI, Requests library
"""

import httpx
import asyncio
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
# Try backend/.env first, then parent directory .env
env_path = Path(__file__).parent.parent.parent / ".env"  # Go up to project root
if not env_path.exists():
    env_path = Path(__file__).parent.parent / ".env"  # Try backend/.env
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

class WeatherDataAcquisition:
    """Handles weather data acquisition from multiple APIs"""
    
    def __init__(self):
        # Get API keys from environment variables, fallback to demo_key if not set
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY", "demo_key")
        self.weatherapi_key = os.getenv("WEATHERAPI_KEY", "demo_key")
        
        # Log API key status (without exposing actual keys)
        if self.openweather_api_key == "demo_key":
            logger.warning("OpenWeatherMap API key not set - using fallback data")
        else:
            logger.info("OpenWeatherMap API key is set")
        
        if self.weatherapi_key == "demo_key":
            logger.warning("WeatherAPI key not set - using fallback data")
        else:
            logger.info("WeatherAPI key is set")
        
    async def fetch_openweather_data(self, lat: float, lon: float) -> Optional[Dict]:
        """Fetch weather data from OpenWeatherMap API"""
        if self.openweather_api_key == "demo_key":
            logger.debug("Skipping OpenWeatherMap API - using demo_key")
            return None
        
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.openweather_api_key,
                "units": "metric"
            }
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"Successfully fetched weather from OpenWeatherMap for {lat},{lon}")
                # Estimate rainfall based on weather description and location
                rainfall_estimate = self._estimate_rainfall_from_description(
                    data["weather"][0]["description"], lat, lon
                )
                
                return {
                    "temperature": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "pressure": data["main"]["pressure"],
                    "wind_speed": data["wind"]["speed"],
                    "weather_description": data["weather"][0]["description"],
                    "rainfall": rainfall_estimate,  # Add estimated rainfall
                    "source": "OpenWeatherMap",
                    "timestamp": datetime.utcnow().isoformat()
                }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("OpenWeatherMap API: Invalid API key (401 Unauthorized)")
            else:
                logger.error(f"OpenWeatherMap API HTTP error {e.response.status_code}: {e}")
            return None
        except Exception as e:
            logger.error(f"OpenWeatherMap API error: {e}")
            return None
    
    async def fetch_weatherapi_data(self, lat: float, lon: float) -> Optional[Dict]:
        """Fetch weather data from WeatherAPI.com"""
        if self.weatherapi_key == "demo_key":
            logger.debug("Skipping WeatherAPI - using demo_key")
            return None
        
        try:
            url = "http://api.weatherapi.com/v1/current.json"
            params = {
                "key": self.weatherapi_key,
                "q": f"{lat},{lon}",
                "aqi": "no"
            }
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"Successfully fetched weather from WeatherAPI for {lat},{lon}")
                # WeatherAPI doesn't provide rainfall in current.json, estimate it
                rainfall_estimate = self._estimate_rainfall_from_description(
                    data["current"]["condition"]["text"], lat, lon
                )
                
                return {
                    "temperature": data["current"]["temp_c"],
                    "humidity": data["current"]["humidity"],
                    "pressure": data["current"]["pressure_mb"],
                    "wind_speed": data["current"]["wind_kph"],
                    "weather_description": data["current"]["condition"]["text"],
                    "rainfall": rainfall_estimate,  # Add estimated rainfall
                    "source": "WeatherAPI",
                    "timestamp": datetime.utcnow().isoformat()
                }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("WeatherAPI: Invalid API key (401 Unauthorized)")
            else:
                logger.error(f"WeatherAPI HTTP error {e.response.status_code}: {e}")
            return None
        except Exception as e:
            logger.error(f"WeatherAPI error: {e}")
            return None
    
    async def get_weather_data(self, lat: float, lon: float) -> Dict:
        """
        Get weather data from multiple sources with fallback
        Returns combined weather information
        """
        # Try both APIs concurrently
        openweather_task = self.fetch_openweather_data(lat, lon)
        weatherapi_task = self.fetch_weatherapi_data(lat, lon)
        
        results = await asyncio.gather(openweather_task, weatherapi_task, return_exceptions=True)
        
        openweather_data = results[0] if not isinstance(results[0], Exception) else None
        weatherapi_data = results[1] if not isinstance(results[1], Exception) else None
        
        # Prioritize WeatherAPI.com for better Indian data coverage
        # Use WeatherAPI first if available, then OpenWeatherMap as fallback
        if weatherapi_data:
            return weatherapi_data
        elif openweather_data:
            return openweather_data
        else:
            # Fallback data - use location-based estimates for better differentiation
            # Different regions have different climates
            # Estimate based on latitude (tropical vs temperate)
            if lat < 15:  # Southern India (tropical)
                base_temp = 28.0
                base_humidity = 75.0
                base_rainfall = 150.0
            elif lat < 20:  # Central India
                base_temp = 26.0
                base_humidity = 65.0
                base_rainfall = 120.0
            else:  # Northern India (more temperate)
                base_temp = 24.0
                base_humidity = 55.0
                base_rainfall = 100.0
            
            # Add some variation based on longitude (east vs west)
            lon_factor = (lon - 75) / 10  # Center around 75°E (central India)
            temp_variation = lon_factor * 2  # ±2°C variation
            
            return {
                "temperature": base_temp + temp_variation,
                "humidity": base_humidity + (lon_factor * 5),  # ±5% variation
                "pressure": 1013.25,
                "wind_speed": 5.0,
                "rainfall": base_rainfall + (lon_factor * 20),  # ±20mm variation
                "weather_description": "Clear sky",
                "source": "Fallback (Location-based estimate)",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def search_cities_openweather(self, query: str, state: str = "", limit: int = 20) -> List[Dict]:
        """
        Search for cities using OpenWeatherMap Geocoding API
        Returns list of cities with name, state, country, lat, lon
        """
        if self.openweather_api_key == "demo_key":
            return []
        
        try:
            # OpenWeatherMap Geocoding API
            url = "http://api.openweathermap.org/geo/1.0/direct"
            search_query = f"{query}, India"
            if state:
                search_query = f"{query}, {state}, India"
            
            params = {
                "q": search_query,
                "limit": limit,
                "appid": self.openweather_api_key
            }
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                cities = []
                for city in data:
                    if city.get("country", "").upper() == "IN":
                        # Extract state from the response (might be in name or state field)
                        city_state = state or city.get("state", "")
                        cities.append({
                            "name": city.get("name", ""),
                            "state": city_state,
                            "country": "India",
                            "latitude": city.get("lat", 0),
                            "longitude": city.get("lon", 0)
                        })
                
                logger.info(f"Found {len(cities)} cities from OpenWeatherMap for query: {query}")
                return cities
                
        except Exception as e:
            logger.error(f"OpenWeatherMap city search error: {e}")
            return []
    
    async def search_cities(self, query: str, country: str = "IN") -> List[Dict]:
        """
        Search for cities using WeatherAPI search endpoint
        Returns list of cities with name, region (state), country, lat, lon
        """
        cities = []
        
        # Try WeatherAPI first if key is available
        if self.weatherapi_key != "demo_key":
            try:
                url = "http://api.weatherapi.com/v1/search.json"
                params = {
                    "key": self.weatherapi_key,
                    "q": query
                }
                
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Filter for Indian cities and format response
                    for city in data:
                        if city.get("country", "").upper() == country.upper():
                            cities.append({
                                "name": city.get("name", ""),
                                "state": city.get("region", ""),
                                "country": city.get("country", ""),
                                "latitude": city.get("lat", 0),
                                "longitude": city.get("lon", 0)
                            })
                    
                    logger.info(f"Found {len(cities)} cities from WeatherAPI for query: {query}")
            except Exception as e:
                logger.warning(f"WeatherAPI city search error: {e}")
        
        # Fallback to OpenWeatherMap if WeatherAPI didn't return results
        if len(cities) == 0 and self.openweather_api_key != "demo_key":
            openweather_cities = await self.search_cities_openweather(query)
            cities.extend(openweather_cities)
        
        return cities
    
    def _estimate_rainfall_from_description(self, description: str, lat: float, lon: float) -> float:
        """Estimate rainfall based on weather description and location"""
        desc_lower = description.lower()
        
        # Base rainfall estimates by description
        if any(word in desc_lower for word in ['rain', 'drizzle', 'shower', 'storm', 'thunder']):
            base_rainfall = 200.0  # High rainfall
        elif any(word in desc_lower for word in ['cloud', 'overcast', 'mist', 'fog']):
            base_rainfall = 100.0  # Moderate
        else:
            base_rainfall = 50.0   # Low (clear, sunny)
        
        # Adjust based on location (monsoon regions get more)
        if lat < 15:  # Southern India (high rainfall regions)
            base_rainfall *= 1.5
        elif lat < 20:  # Central India
            base_rainfall *= 1.2
        # Northern India keeps base
        
        return round(base_rainfall, 1)
    
    def get_weather_summary(self, weather_data: Dict, language: str = "en") -> str:
        """Generate a human-readable weather summary"""
        temp = weather_data.get("temperature", 0)
        humidity = weather_data.get("humidity", 0)
        desc = weather_data.get("weather_description", "Unknown")
        source = weather_data.get("source", "Unknown")
        rainfall = weather_data.get("rainfall", 0)
        
        # Note: Translation will be handled recursively in main.py
        # This method just formats the summary string
        if rainfall:
            return f"Weather: {desc}, {temp}°C, {humidity}% humidity, ~{rainfall}mm rainfall (Source: {source})"
        else:
            return f"Weather: {desc}, {temp}°C, {humidity}% humidity (Source: {source})"


