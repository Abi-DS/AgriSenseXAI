from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from datetime import datetime
import httpx
import json
import re
import os
import numpy as np
from pathlib import Path

from .schemas import HealthResponse, RecommendRequest, RecommendResponse
from pydantic import BaseModel

# Import modular components
from modules.data_acquisition.weather_apis import WeatherDataAcquisition
from modules.data_acquisition.soil_data_handler import SoilDataHandler
from modules.ml_engine.crop_recommendation_model import CropRecommendationModel
from modules.xai_engine.explanation_generator import ExplanationGenerator
from modules.ui_module.flutter_mockups import FlutterUIMockups
from modules.multilingual.translation_service import TranslationService


app = FastAPI(title="AgriSense XAI API", version="0.1.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Initialize modular components
weather_acquisition = WeatherDataAcquisition()
soil_handler = SoilDataHandler()
ml_model = CropRecommendationModel()
ui_mockups = FlutterUIMockups()
translation_service = TranslationService()

# Initialize ML model and generate training data for XAI
print("Initializing ML model...")
if not ml_model.load_model():
    print("Training new ML model...")
    X_train, y_train = ml_model.generate_synthetic_data(5000)
    ml_model.train_model(X_train, y_train)

# Initialize XAI engine with model and training data
print("Initializing XAI engine with SHAP and LIME...")
X_train, _ = ml_model.generate_synthetic_data(1000)  # Generate sample data for explainers
explanation_generator = ExplanationGenerator(
    ml_model=ml_model.get_model(),
    training_data=X_train
)
print("[SUCCESS] ML model and XAI engine initialized successfully")

DATA_DIR = Path(__file__).parent.parent / "data"
LOCATIONS_FILE = DATA_DIR / "locations.json"


async def _fetch_weather(lat: float, lon: float) -> str:
	# Open-Meteo free API
	url = "https://api.open-meteo.com/v1/forecast"
	params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m,relative_humidity_2m", "current": "temperature_2m,relative_humidity_2m"}
	async with httpx.AsyncClient(timeout=5) as client:
		resp = await client.get(url, params=params)
		resp.raise_for_status()
		js = resp.json()
		c = js.get("current", {})
		temp = c.get("temperature_2m")
		rh = c.get("relative_humidity_2m")
		return f"Temp {temp}°C, RH {rh}%"


def _load_locations() -> list:
	try:
		with open(LOCATIONS_FILE, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return []


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
	return HealthResponse(status="ok")


@app.get("/locations")
def list_locations():
	return {"locations": _load_locations(), "updated_at": datetime.utcnow().isoformat()}


@app.get("/cities/search")
async def search_cities(query: str, country: str = "IN"):
	"""Search for cities using WeatherAPI"""
	try:
		cities = await weather_acquisition.search_cities(query, country)
		return {
			"cities": cities,
			"query": query,
			"count": len(cities)
		}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"City search failed: {str(e)}")


@app.get("/cities/states")
async def get_states():
	"""Get list of Indian states with cities"""
	# Use expanded static locations first (fast and reliable)
	locations = _load_locations()
	states_dict = {}
	for loc in locations:
		state = loc.get("state", "Unknown")
		if state not in states_dict:
			states_dict[state] = []
		district = loc.get("district", "")
		if district and district not in states_dict[state]:
			states_dict[state].append(district)
	
	states_list = [
		{
			"state": state,
			"cities": sorted(list(set(districts)), key=str.lower)
		}
		for state, districts in sorted(states_dict.items())
		if len(districts) > 0
	]
	
	return {
		"states": states_list,
		"total_states": len(states_list),
		"source": "static"
	}


@app.get("/modules/status")
def get_modules_status():
	"""Get implementation status of all modules"""
	return {
		"module_1_data_acquisition": {
			"name": "Data Acquisition Module",
			"functionality": "Weather APIs and soil data handling",
			"technologies": ["Python", "FastAPI", "Requests library"],
			"implementation_status": "70%",
			"features": ["OpenWeatherMap API", "WeatherAPI.com", "Soil data validation", "Weather data fallback"]
		},
		"module_2_ml_engine": {
			"name": "ML Recommendation Engine", 
			"functionality": "Crop recommendation using machine learning",
			"technologies": ["Python", "Scikit-learn", "LightGBM"],
			"implementation_status": "40%",
			"features": ["LightGBM model", "Synthetic data generation", "Feature importance", "Multi-class prediction"]
		},
		"module_3_xai_engine": {
			"name": "XAI Explanation Engine",
			"functionality": "Explainable AI for crop recommendations",
			"technologies": ["Python", "SHAP", "LIME"],
			"implementation_status": "30%",
			"features": ["Rule-based explanations", "Feature impact analysis", "SHAP-style explanations", "Human-readable descriptions"]
		},
		"module_4_ui_module": {
			"name": "User Interface & Interaction Module",
			"functionality": "Mobile and web application interface",
			"technologies": ["Flutter", "Python backend"],
			"implementation_status": "20%",
			"features": ["Complete UI mockups", "Screen designs", "Navigation flow", "Component specifications"]
		},
		"module_5_multilingual": {
			"name": "Multilingual Support Module",
			"functionality": "Voice-driven interface for regional languages",
			"technologies": ["Python", "Text-to-Speech", "Speech-to-Text"],
			"implementation_status": "60%",
			"features": ["10 Indian languages", "Voice commands", "Text translation", "Rural-friendly interface"]
		}
	}


@app.get("/ui/mockups")
def get_ui_mockups():
	"""Get Flutter UI mockups and designs"""
	return {
		"mockups": [mockup.__dict__ for mockup in ui_mockups.get_all_mockups()],
		"total_screens": len(ui_mockups.get_all_mockups()),
		"design_status": "100% complete",
		"implementation_status": "20% complete"
	}


@app.get("/languages")
def get_supported_languages():
	"""Get supported languages for multilingual interface"""
	return {
		"languages": translation_service.get_supported_languages(),
		"voice_supported": [lang for lang in translation_service.get_supported_languages().keys() 
		                   if translation_service.is_voice_supported(lang)],
		"total_languages": len(translation_service.get_supported_languages())
	}


@app.get("/translate/{language}")
def get_translations(language: str):
	"""Get translations for specific language"""
	if language not in translation_service.get_supported_languages():
		raise HTTPException(status_code=400, detail="Language not supported")
	
	return {
		"language": language,
		"translations": translation_service.translations.get(language, {}),
		"voice_supported": translation_service.is_voice_supported(language),
		"voice_commands": translation_service.get_voice_commands(language),
		"dynamic_translation_available": translation_service.use_dynamic_translation
	}

class TranslateRequest(BaseModel):
	text: str
	target_language: str = "hi"
	source_language: str = "en"

@app.post("/translate/text")
async def translate_text(request: TranslateRequest):
	"""Dynamically translate any text to target language"""
	if request.target_language not in translation_service.get_supported_languages():
		raise HTTPException(status_code=400, detail="Target language not supported")
	
	if request.source_language not in translation_service.get_supported_languages():
		raise HTTPException(status_code=400, detail="Source language not supported")
	
	try:
		translated = translation_service.translate_dynamic(request.text, request.target_language, request.source_language)
		return {
			"original": request.text,
			"translated": translated,
			"source_language": request.source_language,
			"target_language": request.target_language,
			"method": "dynamic" if translation_service.use_dynamic_translation else "static"
		}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest) -> RecommendResponse:
	"""Enhanced recommendation using modular architecture"""
	try:
		# Module 1: Data Acquisition
		weather_data = None
		weather_summary = None
		
		if payload.use_weather:
			try:
				lat, lon = payload.latitude, payload.longitude
				# If lat/lon are zero, try to find from provided season/state/district via dataset
				if (not lat and not lon) or (lat == 0 and lon == 0):
					for loc in _load_locations():
						if payload.season and payload.season.lower() in loc.get("state", "").lower():
							lat, lon = loc["lat"], loc["lon"]
							break
				
				# Use modular weather acquisition
				weather_data = await weather_acquisition.get_weather_data(lat, lon)
				weather_summary = weather_acquisition.get_weather_summary(weather_data)
				print(f"Weather data for ({lat}, {lon}): temp={weather_data.get('temperature', 'N/A')}, humidity={weather_data.get('humidity', 'N/A')}, rainfall={weather_data.get('rainfall', 'N/A')}")
			except Exception as e:
				print(f"Weather acquisition error: {e}")
				weather_summary = None
				weather_data = None
		
		# Get location-based soil data if available (for Simple Mode with default params)
		# If user provided specific soil params, use those; otherwise estimate from location
		soil_params = {
			'ph': payload.ph,
			'nitrogen': payload.nitrogen,
			'phosphorus': payload.phosphorus,
			'potassium': payload.potassium,
			'moisture': payload.moisture
		}
		
		# Check if using default values (Simple Mode) - if so, estimate from location
		default_values = {'ph': 6.5, 'nitrogen': 40, 'phosphorus': 30, 'potassium': 30, 'moisture': 50}
		using_defaults = all(
			abs(soil_params.get(k, 0) - default_values.get(k, 0)) < 0.1 
			for k in default_values.keys()
		)
		
		if using_defaults and payload.use_weather:
			# Estimate soil parameters from location (with Ambee API if available)
			lat, lon = payload.latitude, payload.longitude
			estimated_soil = await soil_handler.estimate_soil_from_location(
				lat, lon, 
				state=getattr(payload, 'state', ''),
				district=getattr(payload, 'district', '')
			)
			soil_params.update(estimated_soil)
			source = estimated_soil.get('source', 'Estimate')
			print(f"Using {source} soil data for ({lat}, {lon}): ph={soil_params['ph']:.1f}, N={soil_params['nitrogen']:.1f}, P={soil_params['phosphorus']:.1f}, K={soil_params['potassium']:.1f}")
		
		# Prepare features for ML model
		# Use actual weather data if available, otherwise use defaults
		temperature = weather_data.get('temperature', 25.0) if weather_data else 25.0
		humidity = weather_data.get('humidity', 65.0) if weather_data else 65.0
		rainfall = weather_data.get('rainfall', 100.0) if weather_data else 100.0
		
		features = {
			'ph': soil_params['ph'],
			'nitrogen': soil_params['nitrogen'],
			'phosphorus': soil_params['phosphorus'],
			'potassium': soil_params['potassium'],
			'moisture': soil_params['moisture'],
			'temperature': temperature,
			'humidity': humidity,
			'rainfall': rainfall,
			'organic_matter': soil_params.get('organic_matter', 3.0)
		}
		
		print(f"Features for prediction: temp={temperature:.1f}, humidity={humidity:.1f}, rainfall={rainfall:.1f}, ph={features['ph']:.1f}, N={features['nitrogen']:.1f}, P={features['phosphorus']:.1f}, K={features['potassium']:.1f}")
		
		# Module 2: ML Recommendation Engine
		try:
			crop_predictions = ml_model.predict_crops(features, top_k=5)
			print(f"Generated {len(crop_predictions)} crop predictions")
		except Exception as e:
			print(f"Error in ML prediction: {e}")
			raise
		
		# Module 3: XAI Explanation Engine
		try:
			explanations = explanation_generator.generate_explanation(features, crop_predictions)
			print(f"Generated {len(explanations)} explanations")
		except Exception as e:
			print(f"Error in explanation generation: {e}")
			import traceback
			traceback.print_exc()
			raise
		
		# Convert to response format - build complete response first
		language = payload.language or "en"
		
		top_crops = []
		explanations_list = []
		
		# Build the response object with ALL English text from XAI
		for i, pred in enumerate(crop_predictions):
			top_crops.append({
				"crop": pred['crop'],
				"crop_translated": pred['crop'],  # Will be translated recursively
				"score": pred['confidence']
			})
			
			if i < len(explanations):
				explanation = explanations[i]
				
				# Build attributions with ALL original English text (using SHAP/LIME contributions)
				attributions = []
				# Handle both primary_factors and secondary_factors
				all_factors = (explanation.primary_factors or []) + (explanation.secondary_factors or [])
				if not all_factors:
					# If no factors, create a default one
					all_factors = []
				
				for factor in all_factors[:5]:  # Limit to top 5 factors
					try:
						# Safely extract importance (handle if it's a list or array)
						importance_val = factor.importance if hasattr(factor, 'importance') else 0.5
						if isinstance(importance_val, (list, np.ndarray)):
							importance_val = float(importance_val[0]) if len(importance_val) > 0 else 0.5
						else:
							importance_val = float(importance_val)
						importance_val = abs(importance_val)
						
						# Safely extract contribution (handle if it's a list or array)
						contribution_val = factor.contribution if hasattr(factor, 'contribution') else 0.0
						if isinstance(contribution_val, (list, np.ndarray)):
							contribution_val = float(contribution_val[0]) if len(contribution_val) > 0 else 0.0
						else:
							contribution_val = float(contribution_val)
						
						attributions.append({
							"feature": factor.feature_name,
							"feature_translated": factor.feature_name,  # Will be translated
							"importance": importance_val,
							"contribution": contribution_val,  # SHAP/LIME value
							"direction": factor.impact if hasattr(factor, 'impact') else 'positive',
							"direction_translated": factor.impact if hasattr(factor, 'impact') else 'positive',  # Will be translated
							"description": factor.description if hasattr(factor, 'description') else '',  # XAI-generated description
							"description_translated": factor.description if hasattr(factor, 'description') else '',  # Will be translated
							"method": factor.method if hasattr(factor, 'method') else 'rule_based'  # 'shap', 'lime', or 'rule_based'
						})
					except Exception as e:
						print(f"Error processing factor: {e}")
						import traceback
						traceback.print_exc()
						continue
				
				explanations_list.append({
					"crop": explanation.crop_name,
					"crop_translated": explanation.crop_name,  # Will be translated
					"text": explanation.overall_explanation,  # XAI-generated explanation
					"text_translated": explanation.overall_explanation,  # Will be translated
					"attributions": attributions
				})
		
		# Build the complete response object
		response_dict = {
			"top_crops": top_crops,
			"explanations": explanations_list,
			"model_version": "modular_v1.0",
			"weather_summary": weather_summary
		}
		
		# NOW translate ALL _translated fields AND weather_summary recursively - this catches EVERY piece of text
		print(f"\n=== TRANSLATING ALL TEXT (XAI + WEATHER + FALLBACK) TO {language.upper()} ===")
		print(f"Response object has {len(top_crops)} crops and {len(explanations_list)} explanations")
		if weather_summary:
			print(f"Weather summary: {weather_summary}")
		
		if language != 'en':
			# Create a translation-only object with ALL fields that need translation
			# Include ALL text fields, not just _translated ones
			translation_target = {
				"top_crops": [
					{"crop_translated": crop["crop_translated"], "crop": crop["crop"]} 
					for crop in top_crops
				],
				"explanations": [
					{
						"crop_translated": exp["crop_translated"],
						"crop": exp["crop"],  # Also translate crop name directly
						"text_translated": exp["text_translated"],
						"text": exp["text"],  # Also translate main text
						"attributions": [
							{
								"feature_translated": attr["feature_translated"],
								"feature": attr["feature"],  # Also translate feature name
								"direction_translated": attr["direction_translated"],
								"direction": attr["direction"],  # Also translate direction
								"description_translated": attr["description_translated"],
								"description": attr.get("description", "")  # Also translate description
							}
							for attr in exp["attributions"]
						]
					}
					for exp in explanations_list
				],
				"weather_summary": weather_summary if weather_summary else ""  # Include weather summary for translation
			}
			
			# Translate the entire translation target object recursively
			# This will translate weather_summary including "Source: Fallback" text
			translated_target = translation_service.translate_object_recursive(
				translation_target,
				target_language=language,
				source_language='en'
			)
			
			# Now update the response_dict with translated values
			# The recursive translation should have translated all strings, so we use the translated versions
			for i, crop in enumerate(response_dict["top_crops"]):
				# Use translated crop name (from either crop or crop_translated field)
				translated_crop = translated_target["top_crops"][i]
				crop["crop_translated"] = translated_crop.get("crop", translated_crop.get("crop_translated", crop["crop"]))
			
			for i, exp in enumerate(response_dict["explanations"]):
				translated_exp = translated_target["explanations"][i]
				# Use translated crop name
				exp["crop_translated"] = translated_exp.get("crop", translated_exp.get("crop_translated", exp["crop"]))
				
				# Use translated text
				exp["text_translated"] = translated_exp.get("text", translated_exp.get("text_translated", exp["text"]))
				
				# Update attributions
				for j, attr in enumerate(exp["attributions"]):
					translated_attr = translated_exp["attributions"][j]
					# Use translated feature name
					attr["feature_translated"] = translated_attr.get("feature", translated_attr.get("feature_translated", attr["feature"]))
					# Use translated direction
					attr["direction_translated"] = translated_attr.get("direction", translated_attr.get("direction_translated", attr["direction"]))
					# Use translated description
					attr["description_translated"] = translated_attr.get("description", translated_attr.get("description_translated", attr.get("description", "")))
			
			# For non-English, also update the primary text field to show translated version
			for exp in response_dict["explanations"]:
				exp["text"] = exp["text_translated"]  # Use translated text as primary
			
			# Translate weather_summary including fallback messages
			if weather_summary:
				response_dict["weather_summary"] = translated_target["weather_summary"]
				print(f"Translated weather summary: {response_dict['weather_summary']}")
		
		print(f"=== TRANSLATION COMPLETE ===\n")
		
		# Return the translated response
		return RecommendResponse(**response_dict)
		
	except ValidationError as ve:
		raise HTTPException(status_code=400, detail=str(ve))
	except Exception as e:
		import traceback
		error_details = traceback.format_exc()
		print(f"ERROR in /recommend endpoint: {str(e)}")
		print(f"Traceback: {error_details}")
		raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}") from e




