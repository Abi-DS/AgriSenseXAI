"""
AgroXAI - Self-Contained Streamlit App
All backend functionality integrated directly
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import asyncio

# Add backend to path - fix the path setup
current_dir = Path(__file__).parent
backend_dir = current_dir / "backend"
sys.path.insert(0, str(backend_dir))

# Import backend modules with correct path
try:
    from modules.data_acquisition.weather_apis import WeatherDataAcquisition
    from modules.data_acquisition.soil_data_handler import SoilDataHandler
    from modules.ml_engine.crop_recommendation_model import CropRecommendationModel
    from modules.xai_engine.explanation_generator import ExplanationGenerator
    from modules.multilingual.translation_service import TranslationService
except ImportError as e:
    # Try alternative import path
    sys.path.insert(0, str(current_dir))
    from backend.modules.data_acquisition.weather_apis import WeatherDataAcquisition
    from backend.modules.data_acquisition.soil_data_handler import SoilDataHandler
    from backend.modules.ml_engine.crop_recommendation_model import CropRecommendationModel
    from backend.modules.xai_engine.explanation_generator import ExplanationGenerator
    from backend.modules.multilingual.translation_service import TranslationService

# Load environment variables - prioritize Streamlit secrets, then .env file
from dotenv import load_dotenv
import os

# First, try to get from Streamlit secrets (for Streamlit Cloud)
# Note: This must be done before st.set_page_config() in some cases
try:
    if hasattr(st, 'secrets'):
        # Streamlit Cloud secrets - get and set in environment
        weatherapi_secret = st.secrets.get('WEATHERAPI_KEY', '')
        openweather_secret = st.secrets.get('OPENWEATHER_API_KEY', '')
        ambee_secret = st.secrets.get('AMBEE_API_KEY', '')
        
        if weatherapi_secret:
            os.environ['WEATHERAPI_KEY'] = weatherapi_secret
        if openweather_secret:
            os.environ['OPENWEATHER_API_KEY'] = openweather_secret
        if ambee_secret:
            os.environ['AMBEE_API_KEY'] = ambee_secret
except (FileNotFoundError, AttributeError, KeyError, Exception) as e:
    # Secrets not available (local dev or not configured)
    pass

# Then try .env file (for local development)
env_path = current_dir / ".env"
if not env_path.exists():
    env_path = current_dir / "backend" / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
# Also try loading from root
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AgroXAI - Crop Recommendation System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language options
LANGUAGES = {
    "en": "English",
    "hi": "हिंदी (Hindi)",
    "ta": "தமிழ் (Tamil)",
    "te": "తెలుగు (Telugu)",
    "bn": "বাংলা (Bengali)",
    "ml": "മലയാളം (Malayalam)"
}

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'translations' not in st.session_state:
    st.session_state.translations = {}
if 'states' not in st.session_state:
    st.session_state.states = []
if 'selected_state' not in st.session_state:
    st.session_state.selected_state = ''
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = ''
if 'selected_city_data' not in st.session_state:
    st.session_state.selected_city_data = None
if 'mode' not in st.session_state:
    st.session_state.mode = 'simple'
if 'backend_initialized' not in st.session_state:
    st.session_state.backend_initialized = False

# Initialize backend components (only once)
@st.cache_resource
def initialize_backend():
    """Initialize all backend components"""
    try:
        # Check API keys before initializing - re-read from environment
        weatherapi_key = os.getenv('WEATHERAPI_KEY', '')
        openweather_key = os.getenv('OPENWEATHER_API_KEY', '')
        ambee_key = os.getenv('AMBEE_API_KEY', '')
        
        # Log API key status (for debugging in Streamlit Cloud logs)
        if weatherapi_key and weatherapi_key != 'demo_key' and len(weatherapi_key) > 10:
            print(f"[INFO] WeatherAPI key is set (length: {len(weatherapi_key)})")
        else:
            print("[WARNING] WeatherAPI key not found or invalid - will use fallback")
            if weatherapi_key:
                print(f"[DEBUG] WEATHERAPI_KEY value: {weatherapi_key[:5]}...")
        
        if openweather_key and openweather_key != 'demo_key' and len(openweather_key) > 10:
            print(f"[INFO] OpenWeatherMap key is set (length: {len(openweather_key)})")
        else:
            print("[WARNING] OpenWeatherMap key not found - will use fallback")
        
        if ambee_key and len(ambee_key) > 10:
            print(f"[INFO] Ambee API key is set (length: {len(ambee_key)})")
        else:
            print("[INFO] Ambee API key not set - will use location estimates")
        
        # Initialize components (they will read from os.getenv)
        weather_acquisition = WeatherDataAcquisition()
        soil_handler = SoilDataHandler()
        ml_model = CropRecommendationModel()
        translation_service = TranslationService()
        
        # Initialize ML model
        if not ml_model.load_model():
            with st.spinner("Training ML model (first time only)..."):
                X_train, y_train = ml_model.generate_synthetic_data(5000)
                ml_model.train_model(X_train, y_train)
        
        # Initialize XAI engine
        X_train, _ = ml_model.generate_synthetic_data(1000)
        explanation_generator = ExplanationGenerator(
            ml_model=ml_model.get_model(),
            training_data=X_train
        )
        
        return {
            'weather': weather_acquisition,
            'soil': soil_handler,
            'ml_model': ml_model,
            'xai': explanation_generator,
            'translation': translation_service
        }
    except Exception as e:
        st.error(f"Failed to initialize backend: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Load backend components
backend = initialize_backend()
if backend is None:
    st.error("Backend initialization failed. Please check the logs.")
    st.stop()

# Load locations data
@st.cache_data
def load_locations():
    """Load states and cities from data file"""
    try:
        data_dir = current_dir / "backend" / "data"
        locations_file = data_dir / "locations.json"
        if locations_file.exists():
            import json
            with open(locations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Group by state
                states_dict = {}
                for loc in data:
                    state = loc.get('state', 'Unknown')
                    if not state or state == 'Unknown':
                        continue
                    if state not in states_dict:
                        states_dict[state] = {'state': state, 'cities': []}
                    # Try district first (as in locations.json), then name, then city
                    city_name = loc.get('district', loc.get('name', loc.get('city', '')))
                    if city_name and city_name not in states_dict[state]['cities']:
                        states_dict[state]['cities'].append(city_name)
                # Sort cities for each state
                for state_data in states_dict.values():
                    state_data['cities'].sort()
                return sorted(list(states_dict.values()), key=lambda x: x['state'])
        else:
            st.error(f"Locations file not found at: {locations_file}")
    except Exception as e:
        st.error(f"Failed to load locations: {e}")
        import traceback
        st.code(traceback.format_exc())
    return []

def load_translations(language: str) -> Dict:
    """Load translations from translation service"""
    try:
        if language == 'en':
            return {}
        # Get translations from translation service
        translations = backend['translation']._get_explanation_translations(language)
        return translations
    except Exception as e:
        st.warning(f"Failed to load translations: {e}")
    return {}

async def search_city_async(city_name: str, state: str) -> Optional[Dict]:
    """Search for city coordinates using WeatherAPI (prioritized) with fallback to static data"""
    # Try WeatherAPI first (like Next.js frontend does)
    try:
        cities = await backend['weather'].search_cities(city_name, country="IN")
        if cities:
            # Find exact match for city and state
            city = next(
                (c for c in cities 
                 if c.get('name', '').lower() == city_name.lower() and 
                    c.get('state', '').lower() == state.lower()),
                None
            ) or next(
                (c for c in cities 
                 if c.get('name', '').lower() == city_name.lower()),
                None
            ) or cities[0]
            
            if city and city.get('latitude') and city.get('longitude'):
                return {
                    'name': city.get('name', city_name),
                    'state': city.get('state', state),
                    'latitude': city.get('latitude', 20.5937),
                    'longitude': city.get('longitude', 78.9629)
                }
    except Exception as e:
        st.warning(f"WeatherAPI city search failed: {e}")
    
    # Fallback to static locations.json (uses "district" field)
    try:
        data_dir = current_dir / "backend" / "data"
        locations_file = data_dir / "locations.json"
        if locations_file.exists():
            import json
            with open(locations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for loc in data:
                    # Check district, name, or city fields
                    loc_city = loc.get('district', loc.get('name', loc.get('city', '')))
                    if loc_city.lower() == city_name.lower():
                        if loc.get('state', '').lower() == state.lower():
                            return {
                                'name': city_name,
                                'state': state,
                                'latitude': loc.get('lat', loc.get('latitude', 20.5937)),
                                'longitude': loc.get('lon', loc.get('longitude', 78.9629))
                            }
    except Exception as e:
        st.warning(f"Static city search failed: {e}")
    
    return None

def search_city(city_name: str, state: str) -> Optional[Dict]:
    """Wrapper to run async city search"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(search_city_async(city_name, state))

async def get_recommendation_async(payload: Dict) -> Optional[Dict]:
    """Get crop recommendation using backend modules directly"""
    try:
        # Get weather data
        weather_data = None
        weather_summary = None
        
        if payload.get('use_weather', True):
            lat = payload.get('latitude', 20.5937)
            lon = payload.get('longitude', 78.9629)
            weather_data = await backend['weather'].get_weather_data(lat, lon)
            weather_summary = backend['weather'].get_weather_summary(weather_data)
        
        # Get soil parameters
        soil_params = {
            'ph': payload.get('ph', 6.5),
            'nitrogen': payload.get('nitrogen', 40),
            'phosphorus': payload.get('phosphorus', 30),
            'potassium': payload.get('potassium', 30),
            'moisture': payload.get('moisture', 50)
        }
        
        # Check if using defaults - estimate from location
        default_values = {'ph': 6.5, 'nitrogen': 40, 'phosphorus': 30, 'potassium': 30, 'moisture': 50}
        using_defaults = all(
            abs(soil_params.get(k, 0) - default_values.get(k, 0)) < 0.1 
            for k in default_values.keys()
        )
        
        if using_defaults and payload.get('use_weather', True):
            lat = payload.get('latitude', 20.5937)
            lon = payload.get('longitude', 78.9629)
            estimated_soil = await backend['soil'].estimate_soil_from_location(
                lat, lon,
                state=payload.get('state', ''),
                district=payload.get('district', '')
            )
            soil_params.update(estimated_soil)
        
        # Prepare features
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
        
        # ML prediction
        crop_predictions = backend['ml_model'].predict_crops(features, top_k=5)
        
        # XAI explanations
        explanations = backend['xai'].generate_explanation(features, crop_predictions)
        
        # Convert to response format
        language = payload.get('language', 'en')
        
        top_crops = []
        for pred in crop_predictions:
            crop_name = pred['crop']
            crop_translated = backend['translation'].translate_crop_name(crop_name, language) if language != 'en' else crop_name
            top_crops.append({
                'crop': crop_name,
                'crop_translated': crop_translated,
                'score': pred['confidence']
            })
        
        explanations_list = []
        for exp in explanations:
            crop_name = exp.crop_name
            crop_translated = backend['translation'].translate_crop_name(crop_name, language) if language != 'en' else crop_name
            
            # Translate explanation
            explanation_text = exp.overall_explanation
            if language != 'en':
                explanation_text = backend['translation'].translate_dynamic(explanation_text, language)
            
            attributions = []
            for factor in exp.primary_factors[:5]:
                feature = factor.feature_name
                feature_translated = backend['translation'].translate_feature_name(feature, language) if language != 'en' else feature
                
                direction = factor.impact
                direction_translated = backend['translation']._get_explanation_translations(language).get(direction, direction) if language != 'en' else direction
                
                description = factor.description
                description_translated = backend['translation'].translate_dynamic(description, language) if language != 'en' and description else description
                
                attributions.append({
                    'feature': feature,
                    'feature_translated': feature_translated,
                    'direction': direction,
                    'direction_translated': direction_translated,
                    'description': description,
                    'description_translated': description_translated,
                    'importance': float(factor.importance),
                    'contribution': float(factor.contribution),
                    'method': factor.method
                })
            
            explanations_list.append({
                'crop': crop_name,
                'crop_translated': crop_translated,
                'text': explanation_text,
                'text_translated': explanation_text if language != 'en' else explanation_text,
                'attributions': attributions
            })
        
        # Translate weather summary
        if weather_summary and language != 'en':
            weather_summary = backend['translation'].translate_dynamic(weather_summary, language)
        
        return {
            'top_crops': top_crops,
            'explanations': explanations_list,
            'model_version': 'modular_v1.0',
            'weather_summary': weather_summary
        }
    except Exception as e:
        st.error(f"Recommendation failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def get_recommendation(payload: Dict) -> Optional[Dict]:
    """Wrapper to run async function"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(get_recommendation_async(payload))

# Load translations
st.session_state.translations = load_translations(st.session_state.language)

# Load states if not loaded
if not st.session_state.states:
    with st.spinner("Loading location data..."):
        loaded_states = load_locations()
        st.session_state.states = loaded_states
        if len(loaded_states) == 0:
            st.error("No states loaded! Please check that backend/data/locations.json exists and contains valid data.")

# Sidebar for language and mode selection
with st.sidebar:
    st.title("AgroXAI")
    st.markdown("---")
    
    # Show API key status
    weatherapi_key = os.getenv('WEATHERAPI_KEY', '')
    openweather_key = os.getenv('OPENWEATHER_API_KEY', '')
    ambee_key = os.getenv('AMBEE_API_KEY', '')
    
    st.markdown("### API Status")
    if weatherapi_key and weatherapi_key != 'demo_key' and len(weatherapi_key) > 10:
        st.success("WeatherAPI: Configured")
    else:
        st.warning("WeatherAPI: Not set (using fallback)")
    
    if openweather_key and openweather_key != 'demo_key' and len(openweather_key) > 10:
        st.success("OpenWeatherMap: Configured")
    else:
        st.info("OpenWeatherMap: Not set")
    
    if ambee_key and len(ambee_key) > 10:
        st.success("Ambee Soil API: Configured")
    else:
        st.info("Ambee Soil API: Not set (using estimates)")
    
    st.markdown("---")
    
    # Language selector
    selected_lang = st.selectbox(
        "Language / भाषा",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        index=list(LANGUAGES.keys()).index(st.session_state.language) if st.session_state.language in LANGUAGES else 0
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.session_state.translations = load_translations(selected_lang)
        st.rerun()
    
    st.markdown("---")
    
    # Mode selector
    mode = st.radio(
        "Mode",
        options=["simple", "manual"],
        format_func=lambda x: "Simple Mode" if x == "simple" else "Manual Mode",
        index=0 if st.session_state.mode == "simple" else 1
    )
    st.session_state.mode = mode
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **AgroXAI** provides AI-powered crop recommendations with:
    - Real-time weather data
    - Soil parameter analysis
    - LightGBM ML model
    - SHAP & LIME explanations
    - Multilingual support
    """)
    
    st.markdown("---")
    st.markdown("**Self-contained app** - No separate backend needed!")

# Main content
st.title("AgroXAI - Crop Recommendation System")
st.markdown("Get intelligent crop recommendations with explainable AI")

if st.session_state.mode == "simple":
    # Simple Mode - State/City Selection
    st.header("Simple Mode - Select Your Location")
    st.info("Just select your state and city. We'll automatically get weather and soil data!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # State selector
        state_options = [""] + [s['state'] for s in st.session_state.states]
        previous_state = st.session_state.selected_state
        selected_state = st.selectbox(
            st.session_state.translations.get('state', 'State'),
            options=state_options,
            index=0 if not st.session_state.selected_state else (state_options.index(st.session_state.selected_state) if st.session_state.selected_state in state_options else 0),
            key="state_selectbox"
        )
        
        # Reset city if state changed
        if previous_state != selected_state:
            st.session_state.selected_city = ''
            st.session_state.selected_city_data = None
        
        st.session_state.selected_state = selected_state
        
        # City selector (depends on state)
        cities = []
        if selected_state:
            state_data = next((s for s in st.session_state.states if s['state'] == selected_state), None)
            if state_data:
                cities = state_data.get('cities', [])
        
        # Debug info (can be removed later)
        if selected_state and len(cities) == 0:
            st.warning(f"No cities found for {selected_state}. Check locations.json file.")
        
        # Calculate index for city selectbox
        city_index = 0
        if st.session_state.selected_city and cities and st.session_state.selected_city in cities:
            city_index = cities.index(st.session_state.selected_city) + 1
        
        # City options
        city_options = [""] + cities if cities else [""]
        
        selected_city = st.selectbox(
            st.session_state.translations.get('city', 'City'),
            options=city_options,
            disabled=not selected_state or len(cities) == 0,
            index=city_index,
            key="city_selectbox"
        )
        st.session_state.selected_city = selected_city
        
        # Get city coordinates (using WeatherAPI prioritized)
        if selected_state and selected_city:
            if (not st.session_state.selected_city_data or 
                st.session_state.selected_city_data.get('name') != selected_city):
                with st.spinner("Getting location coordinates from WeatherAPI..."):
                    city_data = search_city(selected_city, selected_state)
                    if city_data:
                        st.session_state.selected_city_data = city_data
                    else:
                        # Last resort: use default coordinates
                        st.session_state.selected_city_data = {
                            'name': selected_city,
                            'state': selected_state,
                            'latitude': 20.5937,
                            'longitude': 78.9629
                        }
    
    with col2:
        if st.session_state.selected_city_data:
            st.success("Location Selected")
            city_data = st.session_state.selected_city_data
            st.write(f"**City:** {city_data.get('name', selected_city)}")
            st.write(f"**State:** {city_data.get('state', selected_state)}")
            st.write(f"**Coordinates:** {city_data.get('latitude', 0):.4f}, {city_data.get('longitude', 0):.4f}")
    
    # Submit button
    if st.button(
        st.session_state.translations.get('get_recommendation', 'Get Crop Recommendation'),
        type="primary",
        disabled=not (selected_state and selected_city),
        use_container_width=True
    ):
        if st.session_state.selected_city_data:
            with st.spinner("Getting recommendations with AI analysis..."):
                payload = {
                    "ph": 6.5,
                    "nitrogen": 40,
                    "phosphorus": 30,
                    "potassium": 30,
                    "moisture": 50,
                    "latitude": st.session_state.selected_city_data.get('latitude', 20.5937),
                    "longitude": st.session_state.selected_city_data.get('longitude', 78.9629),
                    "use_weather": True,
                    "language": st.session_state.language,
                    "state": selected_state
                }
                
                result = get_recommendation(payload)
                if result:
                    st.session_state.recommendation_result = result
                    st.rerun()
    
    # Display results
    if 'recommendation_result' in st.session_state and st.session_state.recommendation_result:
        result = st.session_state.recommendation_result
        st.markdown("---")
        st.header("Recommendations")
        
        # Weather summary (like Next.js frontend)
        if result.get('weather_summary'):
            st.info(f"{result['weather_summary']}")
        
        # Model version (like Next.js frontend)
        model_version = result.get('model_version', 'modular_v1.0')
        st.caption(f"Model: {model_version}")
        
        # Top crops
        st.subheader(st.session_state.translations.get('top_crops', 'Top Recommended Crops'))
        top_crops = result.get('top_crops', [])
        
        cols = st.columns(min(len(top_crops), 5))
        for idx, crop in enumerate(top_crops[:5]):
            with cols[idx]:
                crop_name = crop.get('crop_translated') or crop.get('crop', 'Unknown')
                score = crop.get('score', 0)
                st.metric(
                    label=crop_name,
                    value=f"{score:.1%}",
                    help=f"Confidence: {score:.2%}"
                )
        
        # Explanations
        st.subheader(st.session_state.translations.get('explanations', 'Why These Crops?'))
        explanations = result.get('explanations', [])
        
        for exp in explanations:
            crop_name = exp.get('crop_translated') or exp.get('crop', 'Unknown')
            explanation_text = exp.get('text_translated') or exp.get('text', '')
            
            with st.expander(f"{crop_name}", expanded=(exp == explanations[0])):
                st.write(explanation_text)
                
                # Attributions
                attributions = exp.get('attributions', [])
                if attributions:
                    st.markdown("**Key Factors:**")
                    for attr in attributions[:5]:
                        feature = attr.get('feature_translated') or attr.get('feature', '')
                        direction = attr.get('direction_translated') or attr.get('direction', '')
                        description = attr.get('description_translated') or attr.get('description', '')
                        importance = attr.get('importance', 0)
                        method = attr.get('method', 'rule_based')
                        
                        direction_text = "[Positive]" if direction == 'positive' else "[Negative]" if direction == 'negative' else "[Neutral]"
                        method_badge = f" ({method.upper()})" if method != 'rule_based' else ""
                        
                        # Display like Next.js frontend: Feature: Direction — Description
                        description_text = f" — {description}" if description else ""
                        st.write(f"{direction_text} **{feature}**: {direction}{description_text}{method_badge}")
                        
                        # Normalize importance to 0-1 range for progress bar
                        normalized_importance = max(0.0, min(1.0, float(importance)))
                        st.progress(normalized_importance, text=f"Importance: {importance:.1%}")

else:
    # Manual Mode - Full Parameter Input
    st.header("Manual Mode - Enter Soil Parameters")
    st.info("Enter your soil parameters manually for precise recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Soil Parameters")
        ph = st.slider(
            st.session_state.translations.get('ph_level', 'pH Level'),
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1
        )
        
        nitrogen = st.slider(
            st.session_state.translations.get('nitrogen', 'Nitrogen'),
            min_value=0.0,
            max_value=200.0,
            value=40.0,
            step=1.0
        )
        
        phosphorus = st.slider(
            st.session_state.translations.get('phosphorus', 'Phosphorus'),
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=1.0
        )
        
        potassium = st.slider(
            st.session_state.translations.get('potassium', 'Potassium'),
            min_value=0.0,
            max_value=300.0,
            value=30.0,
            step=1.0
        )
        
        moisture = st.slider(
            st.session_state.translations.get('moisture', 'Moisture %'),
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )
    
    with col2:
        st.subheader("Location")
        
        # State selector
        state_options = [""] + [s['state'] for s in st.session_state.states]
        manual_state = st.selectbox(
            "State",
            options=state_options,
            key="manual_state"
        )
        
        # City selector
        manual_cities = []
        if manual_state:
            state_data = next((s for s in st.session_state.states if s['state'] == manual_state), None)
            if state_data:
                manual_cities = state_data.get('cities', [])
        
        manual_city = st.selectbox(
            "City",
            options=[""] + manual_cities,
            disabled=not manual_state,
            key="manual_city"
        )
        
        # Get coordinates
        manual_lat = 20.5937
        manual_lon = 78.9629
        
        if manual_state and manual_city:
            city_data = search_city(manual_city, manual_state)
            if city_data:
                manual_lat = city_data.get('latitude', 20.5937)
                manual_lon = city_data.get('longitude', 78.9629)
                st.success(f"Location: {manual_city}, {manual_state}")
                st.caption(f"Coordinates: {manual_lat:.4f}, {manual_lon:.4f}")
        
        use_weather = st.checkbox(
            "Use weather data",
            value=True,
            help="Enable to get location-specific weather data"
        )
    
    # Submit button
    if st.button(
        st.session_state.translations.get('get_recommendation', 'Get Crop Recommendation'),
        type="primary",
        use_container_width=True
    ):
        with st.spinner("Analyzing with AI..."):
            payload = {
                "ph": ph,
                "nitrogen": nitrogen,
                "phosphorus": phosphorus,
                "potassium": potassium,
                "moisture": moisture,
                "latitude": manual_lat,
                "longitude": manual_lon,
                "use_weather": use_weather,
                "language": st.session_state.language,
                "state": manual_state
            }
            
            result = get_recommendation(payload)
            if result:
                st.session_state.recommendation_result = result
                st.rerun()
    
    # Display results (same as Simple Mode)
    if 'recommendation_result' in st.session_state and st.session_state.recommendation_result:
        result = st.session_state.recommendation_result
        st.markdown("---")
        st.header("Recommendations")
        
        if result.get('weather_summary'):
            st.info(f"{result['weather_summary']}")
        
        # Model version
        model_version = result.get('model_version', 'modular_v1.0')
        st.caption(f"Model: {model_version}")
        
        st.subheader(st.session_state.translations.get('top_crops', 'Top Recommended Crops'))
        top_crops = result.get('top_crops', [])
        
        cols = st.columns(min(len(top_crops), 5))
        for idx, crop in enumerate(top_crops[:5]):
            with cols[idx]:
                crop_name = crop.get('crop_translated') or crop.get('crop', 'Unknown')
                score = crop.get('score', 0)
                st.metric(
                    label=crop_name,
                    value=f"{score:.1%}",
                    help=f"Confidence: {score:.2%}"
                )
        
        st.subheader(st.session_state.translations.get('explanations', 'Why These Crops?'))
        explanations = result.get('explanations', [])
        
        for exp in explanations:
            crop_name = exp.get('crop_translated') or exp.get('crop', 'Unknown')
            explanation_text = exp.get('text_translated') or exp.get('text', '')
            
            with st.expander(f"{crop_name}", expanded=(exp == explanations[0])):
                st.write(explanation_text)
                
                attributions = exp.get('attributions', [])
                if attributions:
                    st.markdown("**Key Factors:**")
                    for attr in attributions[:5]:
                        feature = attr.get('feature_translated') or attr.get('feature', '')
                        direction = attr.get('direction_translated') or attr.get('direction', '')
                        description = attr.get('description_translated') or attr.get('description', '')
                        importance = attr.get('importance', 0)
                        method = attr.get('method', 'rule_based')
                        
                        direction_text = "[Positive]" if direction == 'positive' else "[Negative]" if direction == 'negative' else "[Neutral]"
                        method_badge = f" ({method.upper()})" if method != 'rule_based' else ""
                        
                        # Display like Next.js frontend: Feature: Direction — Description
                        description_text = f" — {description}" if description else ""
                        st.write(f"{direction_text} **{feature}**: {direction}{description_text}{method_badge}")
                        
                        # Normalize importance to 0-1 range for progress bar
                        normalized_importance = max(0.0, min(1.0, float(importance)))
                        st.progress(normalized_importance, text=f"Importance: {importance:.1%}")

# Footer
st.markdown("---")
st.markdown("**AgroXAI v1.0** | Powered by LightGBM, SHAP, and LIME | Built with Streamlit")
