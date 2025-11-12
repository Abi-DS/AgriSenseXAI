"""
AgroXAI - Self-Contained Streamlit App
All backend functionality integrated directly
"""

import streamlit as st
import streamlit.components.v1 as components
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
import json

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

# TTS component with proper controls (Play/Pause/Stop, voice selection)
def create_tts_button_simple(text: str, lang_code: str):
    """Create a TTS button with Play/Pause/Stop controls and voice selection"""
    component_id = f"tts_{hash(text) % 100000}"
    text_escaped = json.dumps(text)
    lang_code_map = {
        'hi': 'hi-IN', 'ta': 'ta-IN', 'te': 'te-IN', 
        'bn': 'bn-IN', 'ml': 'ml-IN', 'en': 'en-US'
    }
    lang_final = lang_code_map.get(lang_code, 'en-US')
    
    # Create TTS component with proper controls
    components.html(f"""
    <div id="tts-container-{component_id}" style="width: 100%;">
        <div id="tts-error-{component_id}" style="color: red; display: none; font-size: 12px; margin-bottom: 4px;"></div>
        <div style="display: flex; gap: 4px; align-items: center;">
            <button id="tts-play-{component_id}" style="background: #0b7; color: white; border: none; border-radius: 4px; padding: 6px 12px; cursor: pointer; font-size: 12px; flex: 1;">
                🔊 Play
            </button>
            <button id="tts-pause-{component_id}" style="background: #f90; color: white; border: none; border-radius: 4px; padding: 6px 12px; cursor: pointer; font-size: 12px; display: none;">
                ⏸️ Pause
            </button>
            <button id="tts-stop-{component_id}" style="background: #c33; color: white; border: none; border-radius: 4px; padding: 6px 12px; cursor: pointer; font-size: 12px; display: none;">
                ⏹️ Stop
            </button>
        </div>
    </div>
    
    <script>
    (function() {{
        const containerId = '{component_id}';
        const synth = window.speechSynthesis;
        const textToRead = {text_escaped};
        const langCode = '{lang_final}';
        
        // Get elements
        const errorDiv = document.getElementById('tts-error-' + containerId);
        const playBtn = document.getElementById('tts-play-' + containerId);
        const pauseBtn = document.getElementById('tts-pause-' + containerId);
        const stopBtn = document.getElementById('tts-stop-' + containerId);
        
        let currentUtterance = null;
        let voicesLoaded = false;
        
        // Check if speechSynthesis is supported
        if (!('speechSynthesis' in window)) {{
            errorDiv.textContent = 'Text-to-speech not supported in your browser';
            errorDiv.style.display = 'block';
            playBtn.disabled = true;
            playBtn.style.opacity = '0.5';
            playBtn.style.cursor = 'not-allowed';
            return;
        }}
        
        // Function to populate and get voices (handles async loading)
        function getVoices() {{
            const voices = synth.getVoices();
            if (voices.length > 0 && !voicesLoaded) {{
                voicesLoaded = true;
                console.log('TTS: Loaded', voices.length, 'voices');
                // Log available languages for debugging
                const langVoices = voices.filter(v => v.lang.includes('ta') || v.lang.includes('hi') || v.lang.includes('te') || v.lang.includes('bn') || v.lang.includes('ml'));
                if (langVoices.length > 0) {{
                    console.log('TTS: Found Indian language voices:', langVoices.map(v => v.name + ' (' + v.lang + ')'));
                }}
            }}
            return voices;
        }}
        
        // Populate voices on load and when they become available
        getVoices();
        if (synth.onvoiceschanged !== undefined) {{
            synth.onvoiceschanged = function() {{
                getVoices();
            }};
        }}
        
        // Also try loading voices after a short delay (some browsers need this)
        setTimeout(function() {{
            getVoices();
        }}, 500);
        
        // Function to reset UI to initial state
        function resetUI() {{
            playBtn.style.display = 'block';
            playBtn.textContent = '🔊 Play';
            pauseBtn.style.display = 'none';
            stopBtn.style.display = 'none';
        }}
        
        // Function to show playing state
        function showPlayingState() {{
            playBtn.style.display = 'none';
            pauseBtn.style.display = 'block';
            stopBtn.style.display = 'block';
        }}
        
        // Play button click handler
        playBtn.addEventListener('click', function() {{
            // If paused, resume
            if (synth.paused && currentUtterance) {{
                synth.resume();
                showPlayingState();
                return;
            }}
            
            // Cancel any ongoing speech
            synth.cancel();
            
            // Create new utterance
            currentUtterance = new SpeechSynthesisUtterance(textToRead);
            currentUtterance.lang = langCode;
            currentUtterance.rate = 0.9;
            currentUtterance.pitch = 1;
            currentUtterance.volume = 1.0;
            
            // Try to select appropriate voice for language (get fresh voices list)
            const voices = getVoices();
            if (voices.length > 0) {{
                const langPrefix = langCode.split('-')[0];
                
                // First try exact match (e.g., 'ta-IN')
                let matchingVoice = voices.find(v => v.lang === langCode);
                
                // If no exact match, try language prefix match (e.g., 'ta')
                if (!matchingVoice) {{
                    matchingVoice = voices.find(v => v.lang.startsWith(langPrefix + '-'));
                }}
                
                // If still no match, try any voice with language prefix
                if (!matchingVoice) {{
                    matchingVoice = voices.find(v => v.lang.toLowerCase().includes(langPrefix.toLowerCase()));
                }}
                
                if (matchingVoice) {{
                    currentUtterance.voice = matchingVoice;
                    console.log('TTS: Using voice:', matchingVoice.name, '(' + matchingVoice.lang + ')', 'for language', langCode);
                }} else {{
                    console.log('TTS: No matching voice found for', langCode, '- using default voice');
                    console.log('TTS: Available voices:', voices.map(v => v.name + ' (' + v.lang + ')').slice(0, 10));
                }}
            }} else {{
                console.log('TTS: No voices available yet, will use default');
            }}
            
            // Event handlers
            currentUtterance.onend = function() {{
                resetUI();
                currentUtterance = null;
            }};
            
            currentUtterance.onerror = function(event) {{
                console.error('TTS Error:', event.error);
                errorDiv.textContent = 'Speech error: ' + event.error;
                errorDiv.style.display = 'block';
                resetUI();
                currentUtterance = null;
            }};
            
            currentUtterance.onpause = function() {{
                playBtn.style.display = 'block';
                playBtn.textContent = '▶️ Resume';
                pauseBtn.style.display = 'none';
            }};
            
            currentUtterance.onresume = function() {{
                showPlayingState();
            }};
            
            // Speak
            synth.speak(currentUtterance);
            showPlayingState();
        }});
        
        // Pause button click handler
        pauseBtn.addEventListener('click', function() {{
            if (synth.speaking && !synth.paused) {{
                synth.pause();
            }}
        }});
        
        // Stop button click handler
        stopBtn.addEventListener('click', function() {{
            synth.cancel();
            resetUI();
            currentUtterance = null;
        }});
        
        // Stop speech if page is being unloaded
        window.addEventListener('beforeunload', function() {{
            synth.cancel();
        }});
    }})();
    </script>
    """, height=50)

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
if 'ui_translations' not in st.session_state:
    st.session_state.ui_translations = {}  # Cached UI translations for fast access
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
        all_translations = backend['translation']._get_explanation_translations()
        translations = all_translations.get(language, {}) if language != 'en' else {}
        return translations
    except Exception as e:
        st.warning(f"Failed to load translations: {e}")
    return {}

def preload_ui_translations(language: str):
    """Pre-translate and cache all UI text using STATIC translations (FAST - no API calls)"""
    # Skip preloading - we'll use static translations directly in t() function
    # This avoids any potential blocking issues
    pass

# Translation cache to avoid re-translating same content
_translation_cache = {}

def t(text: str, default: str = None) -> str:
    """Fast translation helper - uses static translations + cached dynamic translations"""
    language = st.session_state.get('language', 'en')
    if language == 'en':
        # For English, return the actual text (not translation keys)
        # Priority: actual text > formatted default key
        if text and text.strip() and not text.startswith('_'):
            # Use the actual text provided (already in proper English)
            return text
        # If only default (translation key) is provided, format it nicely
        if default:
            # Format default key: "simple_mode_description" -> "Simple Mode Description"
            formatted = default.replace('_', ' ').title()
            return formatted
        return text or ""
    
    # Check cache first (for previously translated content)
    cache_key = f"{language}:{text}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]
    
    # Try to get static translations directly from backend (if available)
    try:
        if backend and 'translation' in backend:
            translation_service = backend['translation']
            static_translations = translation_service.translations.get(language, {})
            
            # Map common UI text to static translation keys
            ui_mapping = {
                "API Status": "api_status",
                "Language": "language",
                "Mode": "mode",
                "Simple Mode": "simple_mode",
                "Manual Mode": "manual_mode",
                "About": "about",
                "City": "city",
                "State": "state",
                "Coordinates": "coordinates",
                "Recommendations": "recommendations",
                "Key Factors": "key_factors",
                "Loading location data...": "loading",
                "Soil Parameters": "soil_parameters",
                "Location": "location",
                "Location Selected": "location_selected",
                "Get Crop Recommendation": "get_recommendation",
                "Top Recommended Crops": "top_crops",
                "Why These Crops?": "explanations",
                "provides AI-powered crop recommendations with:": None,
                "Real-time weather data": None,
                "Soil parameter analysis": None,
                "LightGBM ML model": None,
                "SHAP & LIME explanations": None,
                "Multilingual support": None,
                "Self-contained app": None,
                "No separate backend needed!": None,
            }
            
            # Check if text matches a mapping
            if text in ui_mapping:
                key = ui_mapping[text]
                if key and key in static_translations:
                    result = static_translations[key]
                    _translation_cache[cache_key] = result  # Cache it
                    return result
            
            # Check static translations directly
            if text in static_translations:
                result = static_translations[text]
                _translation_cache[cache_key] = result
                return result
            
            # Check with default (this handles translation keys like "simple_mode_description")
            if default and default in static_translations:
                result = static_translations[default]
                _translation_cache[cache_key] = result
                return result
            
            # Also check session state translations (loaded from load_translations)
            if default and default in st.session_state.translations:
                result = st.session_state.translations[default]
                _translation_cache[cache_key] = result
                return result
            
            # NEVER call API for UI text - only return static translations or original
            # UI text should already be in static translations
            # Only dynamic content (explanations, descriptions) should use API, and that's handled elsewhere
    except Exception:
        pass
    
    # Fallback: if default looks like a translation key, try to get from session state
    if default and default in st.session_state.translations:
        return st.session_state.translations[default]
    
    # Final fallback: return original text (better than showing the key)
    result = text if text else (default if default else "")
    _translation_cache[cache_key] = result  # Cache even the original
    return result

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
        
        # OPTIMIZED: Collect all text to translate, then batch translate (much faster!)
        translation_map = {}
        if language != 'en':
            texts_to_translate = []
            
            # Collect all explanation texts and descriptions
            for exp in explanations:
                texts_to_translate.append(exp.overall_explanation)
                for factor in exp.primary_factors[:5]:
                    if factor.description:
                        texts_to_translate.append(factor.description)
            
            # Add weather summary
            if weather_summary:
                texts_to_translate.append(weather_summary)
            
            # Batch translate all texts at once (1-2 API calls instead of 20+)
            if texts_to_translate:
                try:
                    translated_texts = backend['translation'].translate_batch(texts_to_translate, language, 'en')
                    for orig, trans in zip(texts_to_translate, translated_texts):
                        translation_map[orig] = trans
                except Exception:
                    # Fallback: translate individually (still better than before)
                    for text in texts_to_translate:
                        try:
                            translation_map[text] = backend['translation'].translate_dynamic(text, language)
                        except:
                            translation_map[text] = text
        
        # Build explanations using translated text from map
        for exp in explanations:
            crop_name = exp.crop_name
            crop_translated = backend['translation'].translate_crop_name(crop_name, language) if language != 'en' else crop_name
            
            # Get translated explanation from map, or translate dynamically if not found
            explanation_text = exp.overall_explanation
            explanation_text_translated = explanation_text
            if language != 'en':
                if explanation_text in translation_map:
                    explanation_text_translated = translation_map[explanation_text]
                else:
                    # If not in map, translate it dynamically (shouldn't happen if batch worked, but fallback)
                    explanation_text_translated = backend['translation'].translate_dynamic(explanation_text, language)
            
            attributions = []
            for factor in exp.primary_factors[:5]:
                feature = factor.feature_name
                feature_translated = backend['translation'].translate_feature_name(feature, language) if language != 'en' else feature
                
                direction = factor.impact
                all_translations = backend['translation']._get_explanation_translations()
                direction_translated = all_translations.get(language, {}).get(direction, direction) if language != 'en' else direction
                
                description = factor.description
                # Get translated description from map, or translate dynamically if not found
                description_translated = description
                if language != 'en' and description:
                    if description in translation_map:
                        description_translated = translation_map[description]
                    else:
                        # If not in map, translate it dynamically
                        description_translated = backend['translation'].translate_dynamic(description, language)
                
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
                'text_translated': explanation_text_translated,
                'attributions': attributions
            })
        
        # Get translated weather summary from map, or translate if not found
        if weather_summary and language != 'en':
            if weather_summary in translation_map:
                weather_summary = translation_map[weather_summary]
            else:
                # Fallback: translate dynamically
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

# No preloading needed - translations accessed directly when needed in t() function

# Load states if not loaded
if not st.session_state.states:
    with st.spinner(t("Loading location data...", "Loading location data...")):
        loaded_states = load_locations()
        st.session_state.states = loaded_states
        if len(loaded_states) == 0:
            st.error(t("No states loaded! Please check that backend/data/locations.json exists and contains valid data.", "No states loaded! Please check that backend/data/locations.json exists and contains valid data."))

# Sidebar for language and mode selection
with st.sidebar:
    st.title("AgroXAI")
    st.markdown("---")
    
    # Language selector (moved above API status)
    selected_lang = st.selectbox(
        t("Language", "language"),
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        index=list(LANGUAGES.keys()).index(st.session_state.language) if st.session_state.language in LANGUAGES else 0
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.session_state.translations = load_translations(selected_lang)
        # No preloading needed - translations accessed directly in t() function
        st.rerun()
    
    st.markdown("---")
    
    # Show API key status
    st.markdown(f"### {t('API Status', 'api_status')}")
    weatherapi_key = os.getenv('WEATHERAPI_KEY', '')
    openweather_key = os.getenv('OPENWEATHER_API_KEY', '')
    ambee_key = os.getenv('AMBEE_API_KEY', '')
    
    if weatherapi_key and weatherapi_key != 'demo_key' and len(weatherapi_key) > 10:
        st.success(t("WeatherAPI: Configured", "WeatherAPI: Configured"))
    else:
        st.warning(t("WeatherAPI: Not set (using fallback)", "WeatherAPI: Not set (using fallback)"))
    
    if openweather_key and openweather_key != 'demo_key' and len(openweather_key) > 10:
        st.success(t("OpenWeatherMap: Configured", "OpenWeatherMap: Configured"))
    else:
        st.info(t("OpenWeatherMap: Not set", "OpenWeatherMap: Not set"))
    
    if ambee_key and len(ambee_key) > 10:
        st.success(t("Ambee Soil API: Configured", "Ambee Soil API: Configured"))
    else:
        st.info(t("Ambee Soil API: Not set (using estimates)", "Ambee Soil API: Not set (using estimates)"))
    
    st.markdown("---")
    
    # Mode selector
    mode = st.radio(
        t("Mode", "mode"),
        options=["simple", "manual"],
        format_func=lambda x: t("Simple Mode", "simple_mode") if x == "simple" else t("Manual Mode", "manual_mode"),
        index=0 if st.session_state.mode == "simple" else 1
    )
    st.session_state.mode = mode
    
    st.markdown("---")
    st.markdown(f"### {t('About', 'About')}")
    st.markdown(f"""
    **{t('AgroXAI', 'AgroXAI')}** {t('provides AI-powered crop recommendations with:', 'provides AI-powered crop recommendations with:')}
    - {t('Real-time weather data', 'Real-time weather data')}
    - {t('Soil parameter analysis', 'Soil parameter analysis')}
    - {t('LightGBM ML model', 'LightGBM ML model')}
    - {t('SHAP & LIME explanations', 'SHAP & LIME explanations')}
    - {t('Multilingual support', 'Multilingual support')}
    """)
    
    st.markdown("---")
    st.markdown(f"**{t('Self-contained app', 'Self-contained app')}** - {t('No separate backend needed!', 'No separate backend needed!')}")

# Main content
st.title(t("AgroXAI - Crop Recommendation System", "AgroXAI - Crop Recommendation System"))
st.markdown(t("Get intelligent crop recommendations with explainable AI", "Get intelligent crop recommendations with explainable AI"))

if st.session_state.mode == "simple":
    # Simple Mode - State/City Selection
    st.header(t("Simple Mode - Select Your Location", "Simple Mode - Select Your Location"))
    st.info(t("Just select your state and city. We'll automatically get weather and soil data!", "simple_mode_description"))
    
    # Translate instruction text
    instruction_text = t("Select your location and click 'Get Recommendation' to see crop suggestions.", "select_and_submit")
    if instruction_text:
        st.caption(instruction_text)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # State selector - translate state names for display
        language = st.session_state.get('language', 'en')
        state_options_original = [""] + [s['state'] for s in st.session_state.states] if st.session_state.states else [""]
        
        # Create translated state options for display
        state_options_display = state_options_original
        if language != 'en' and backend and 'translation' in backend and st.session_state.states:
            try:
                translation_service = backend['translation']
                if hasattr(translation_service, 'translate_state_name'):
                    state_options_display = [""] + [translation_service.translate_state_name(s['state'], language) for s in st.session_state.states]
            except Exception as e:
                # Fallback to original if translation fails
                st.warning(f"Translation error: {e}")
                state_options_display = state_options_original
        
        previous_state = st.session_state.selected_state
        selected_state_display = st.selectbox(
            t("State", "state"),
            options=state_options_display,
            index=0 if not st.session_state.selected_state else (state_options_original.index(st.session_state.selected_state) if st.session_state.selected_state in state_options_original else 0),
            key="state_selectbox"
        )
        
        # Map displayed state back to original state name
        if selected_state_display:
            selected_state_index = state_options_display.index(selected_state_display)
            selected_state = state_options_original[selected_state_index] if selected_state_index < len(state_options_original) else ""
        else:
            selected_state = ""
        
        # Reset city if state changed
        if previous_state != selected_state:
            st.session_state.selected_city = ''
            st.session_state.selected_city_data = None
        
        st.session_state.selected_state = selected_state
        
        # City selector (depends on state) - translate city names for display
        cities_original = []
        if selected_state:
            state_data = next((s for s in st.session_state.states if s['state'] == selected_state), None)
            if state_data:
                cities_original = state_data.get('cities', [])
        
        # Debug info (can be removed later)
        if selected_state and len(cities_original) == 0:
            st.warning(t(f"No cities found for {selected_state}. Check locations.json file.", f"No cities found for {selected_state}. Check locations.json file."))
        
        # Create translated city options for display
        cities_display = cities_original
        if language != 'en' and backend and 'translation' in backend and cities_original:
            try:
                translation_service = backend['translation']
                if hasattr(translation_service, 'translate_city_name'):
                    cities_display = [translation_service.translate_city_name(city, language) for city in cities_original]
            except Exception as e:
                # Fallback to original if translation fails
                cities_display = cities_original
        
        # Calculate index for city selectbox
        city_index = 0
        if st.session_state.selected_city and cities_original and st.session_state.selected_city in cities_original:
            city_index = cities_original.index(st.session_state.selected_city) + 1
        
        # City options (display translated, but track original)
        city_options_display = [""] + cities_display if cities_display else [""]
        city_options_original = [""] + cities_original if cities_original else [""]
        
        selected_city_display = st.selectbox(
            t("City", "city"),
            options=city_options_display,
            disabled=not selected_state or len(cities_original) == 0,
            index=city_index,
            key="city_selectbox"
        )
        
        # Map displayed city back to original city name
        if selected_city_display:
            selected_city_index = city_options_display.index(selected_city_display)
            selected_city = city_options_original[selected_city_index] if selected_city_index < len(city_options_original) else ""
        else:
            selected_city = ""
        
        st.session_state.selected_city = selected_city
        
        # Get city coordinates (using WeatherAPI prioritized)
        if selected_state and selected_city:
            if (not st.session_state.selected_city_data or 
                st.session_state.selected_city_data.get('name') != selected_city):
                with st.spinner(t("Getting location coordinates from WeatherAPI...", "Getting location coordinates from WeatherAPI...")):
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
            st.success(t("Location Selected", "location_selected"))
            city_data = st.session_state.selected_city_data
            language = st.session_state.get('language', 'en')
            
            # Translate city and state names for display
            city_display = city_data.get('name', selected_city)
            state_display = city_data.get('state', selected_state)
            if language != 'en' and backend and 'translation' in backend:
                try:
                    translation_service = backend['translation']
                    if hasattr(translation_service, 'translate_city_name'):
                        city_display = translation_service.translate_city_name(city_display, language)
                    if hasattr(translation_service, 'translate_state_name'):
                        state_display = translation_service.translate_state_name(state_display, language)
                except Exception:
                    # Keep original names if translation fails
                    pass
            
            st.write(f"**{t('City', 'city')}:** {city_display}")
            st.write(f"**{t('State', 'state')}:** {state_display}")
            st.write(f"**{t('Coordinates', 'coordinates')}:** {city_data.get('latitude', 0):.4f}, {city_data.get('longitude', 0):.4f}")
    
    # Submit button
    if st.button(
        st.session_state.translations.get('get_recommendation', 'Get Crop Recommendation'),
        type="primary",
        disabled=not (selected_state and selected_city),
        use_container_width=True
    ):
        if st.session_state.selected_city_data:
            with st.spinner(t("Getting recommendations with AI analysis...", "Getting recommendations with AI analysis...")):
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
        st.header(t("Recommendations", "Recommendations"))
        
        # Weather summary (like Next.js frontend) with TTS
        if result.get('weather_summary'):
            weather_text = result['weather_summary']
            col1, col2 = st.columns([10, 1])
            with col1:
                st.info(f"{weather_text}")
            with col2:
                language = st.session_state.get('language', 'en')
                lang_code = language if language in ['en', 'hi', 'ta', 'te', 'bn', 'ml'] else 'en'
                create_tts_button_simple(weather_text, lang_code)
        
        # Model version (like Next.js frontend)
        model_version = result.get('model_version', 'modular_v1.0')
        st.caption(f"{t('Model', 'Model')}: {model_version}")
        
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
                    help=f"{t('Confidence', 'Confidence')}: {score:.2%}"
                )
        
        # Explanations
        st.subheader(st.session_state.translations.get('explanations', 'Why These Crops?'))
        explanations = result.get('explanations', [])
        
        for exp in explanations:
            crop_name = exp.get('crop_translated') or exp.get('crop', 'Unknown')
            explanation_text = exp.get('text_translated') or exp.get('text', '')
            
            with st.expander(f"{crop_name}", expanded=(exp == explanations[0])):
                # Add TTS button next to explanation
                col1, col2 = st.columns([10, 1])
                with col1:
                    st.write(explanation_text)
                with col2:
                    language = st.session_state.get('language', 'en')
                    lang_code = language if language in ['en', 'hi', 'ta', 'te', 'bn', 'ml'] else 'en'
                    create_tts_button_simple(explanation_text, lang_code)
                
                # Attributions
                attributions = exp.get('attributions', [])
                if attributions:
                    st.markdown(f"**{t('Key Factors', 'Key Factors')}:**")
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
                        st.progress(normalized_importance, text=f"{t('Importance', 'Importance')}: {importance:.1%}")

else:
    # Manual Mode - Full Parameter Input
    st.header(t("Manual Mode - Enter Soil Parameters", "Manual Mode - Enter Soil Parameters"))
    st.info(t("Enter your soil parameters manually for precise recommendations", "Enter your soil parameters manually for precise recommendations"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t("Soil Parameters", "Soil Parameters"))
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
        st.subheader(t("Location", "location"))
        
        # State selector - translate state names for display
        language = st.session_state.get('language', 'en')
        state_options_original = [""] + [s['state'] for s in st.session_state.states]
        
        # Create translated state options for display
        state_options_display = state_options_original
        if language != 'en' and backend and 'translation' in backend and st.session_state.states:
            try:
                translation_service = backend['translation']
                if hasattr(translation_service, 'translate_state_name'):
                    state_options_display = [""] + [translation_service.translate_state_name(s['state'], language) for s in st.session_state.states]
            except Exception as e:
                # Fallback to original if translation fails
                state_options_display = state_options_original
        
        manual_state_display = st.selectbox(
            t("State", "state"),
            options=state_options_display,
            key="manual_state"
        )
        
        # Map displayed state back to original state name
        if manual_state_display:
            manual_state_index = state_options_display.index(manual_state_display)
            manual_state = state_options_original[manual_state_index] if manual_state_index < len(state_options_original) else ""
        else:
            manual_state = ""
        
        # City selector - translate city names for display
        manual_cities_original = []
        if manual_state:
            state_data = next((s for s in st.session_state.states if s['state'] == manual_state), None)
            if state_data:
                manual_cities_original = state_data.get('cities', [])
        
        # Create translated city options for display
        manual_cities_display = manual_cities_original
        if language != 'en' and backend and 'translation' in backend and manual_cities_original:
            try:
                translation_service = backend['translation']
                if hasattr(translation_service, 'translate_city_name'):
                    manual_cities_display = [translation_service.translate_city_name(city, language) for city in manual_cities_original]
            except Exception:
                # Fallback to original if translation fails
                manual_cities_display = manual_cities_original
        
        manual_city_display = st.selectbox(
            t("City", "city"),
            options=[""] + manual_cities_display,
            disabled=not manual_state,
            key="manual_city"
        )
        
        # Map displayed city back to original city name
        if manual_city_display:
            manual_city_index = ([""] + manual_cities_display).index(manual_city_display)
            manual_city = ([""] + manual_cities_original)[manual_city_index] if manual_city_index < len([""] + manual_cities_original) else ""
        else:
            manual_city = ""
        
        # Get coordinates
        manual_lat = 20.5937
        manual_lon = 78.9629
        
        if manual_state and manual_city:
            city_data = search_city(manual_city, manual_state)
            if city_data:
                manual_lat = city_data.get('latitude', 20.5937)
                manual_lon = city_data.get('longitude', 78.9629)
                # Translate names for display
                city_display = manual_city
                state_display = manual_state
                if language != 'en' and backend and 'translation' in backend:
                    try:
                        translation_service = backend['translation']
                        if hasattr(translation_service, 'translate_city_name'):
                            city_display = translation_service.translate_city_name(manual_city, language)
                        if hasattr(translation_service, 'translate_state_name'):
                            state_display = translation_service.translate_state_name(manual_state, language)
                    except Exception:
                        # Keep original names if translation fails
                        pass
                st.success(t(f"Location: {city_display}, {state_display}", f"Location: {city_display}, {state_display}"))
                st.caption(t(f"Coordinates: {manual_lat:.4f}, {manual_lon:.4f}", f"Coordinates: {manual_lat:.4f}, {manual_lon:.4f}"))
        
        use_weather = st.checkbox(
            t("Use weather data", "Use weather data"),
            value=True,
            help=t("Enable to get location-specific weather data", "Enable to get location-specific weather data")
        )
    
    # Submit button
    if st.button(
        st.session_state.translations.get('get_recommendation', 'Get Crop Recommendation'),
        type="primary",
        use_container_width=True
    ):
        with st.spinner(t("Analyzing with AI...", "Analyzing with AI...")):
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
        st.header(t("Recommendations", "Recommendations"))
        
        if result.get('weather_summary'):
            weather_text = result['weather_summary']
            col1, col2 = st.columns([10, 1])
            with col1:
                st.info(f"{weather_text}")
            with col2:
                language = st.session_state.get('language', 'en')
                lang_code = language if language in ['en', 'hi', 'ta', 'te', 'bn', 'ml'] else 'en'
                create_tts_button_simple(weather_text, lang_code)
        
        # Model version
        model_version = result.get('model_version', 'modular_v1.0')
        st.caption(f"{t('Model', 'Model')}: {model_version}")
        
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
                    help=f"{t('Confidence', 'Confidence')}: {score:.2%}"
                )
        
        st.subheader(st.session_state.translations.get('explanations', 'Why These Crops?'))
        explanations = result.get('explanations', [])
        
        for exp in explanations:
            crop_name = exp.get('crop_translated') or exp.get('crop', 'Unknown')
            explanation_text = exp.get('text_translated') or exp.get('text', '')
            
            with st.expander(f"{crop_name}", expanded=(exp == explanations[0])):
                # Add TTS button next to explanation
                col1, col2 = st.columns([10, 1])
                with col1:
                    st.write(explanation_text)
                with col2:
                    language = st.session_state.get('language', 'en')
                    lang_code = language if language in ['en', 'hi', 'ta', 'te', 'bn', 'ml'] else 'en'
                    create_tts_button_simple(explanation_text, lang_code)
                
                attributions = exp.get('attributions', [])
                if attributions:
                    st.markdown(f"**{t('Key Factors', 'Key Factors')}:**")
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
                        st.progress(normalized_importance, text=f"{t('Importance', 'Importance')}: {importance:.1%}")

# Footer
st.markdown("---")
st.markdown(f"**{t('AgroXAI v1.0', 'AgroXAI v1.0')}** | {t('Powered by LightGBM, SHAP, and LIME', 'Powered by LightGBM, SHAP, and LIME')} | {t('Built with Streamlit', 'Built with Streamlit')}")
