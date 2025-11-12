"""
Multilingual Support Module - Translation Service
Implementation Status: ~60% complete
Technologies: Python, Text-to-Speech, Speech-to-Text
"""

from typing import Dict, List, Optional, Any
import json
import re
from pathlib import Path
import logging

# Try to import dynamic translator
try:
    from deep_translator import GoogleTranslator
    DYNAMIC_TRANSLATION_AVAILABLE = True
except ImportError:
    DYNAMIC_TRANSLATION_AVAILABLE = False
    logging.warning("deep-translator not available. Install with: pip install deep-translator")

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

class TranslationService:
    """Handles multilingual support for rural users"""
    
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi (हिंदी)',
            'ta': 'Tamil (தமிழ்)',
            'te': 'Telugu (తెలుగు)',
            'bn': 'Bengali (বাংলা)',
            'mr': 'Marathi (मराठी)',
            'gu': 'Gujarati (ગુજરાતી)',
            'kn': 'Kannada (ಕನ್ನಡ)',
            'ml': 'Malayalam (മലയാളം)',
            'pa': 'Punjabi (ਪੰਜਾਬੀ)'
        }
        
        # Language code mapping for dynamic translator
        self.lang_code_map = {
            'en': 'en',
            'hi': 'hi',
            'ta': 'ta',
            'te': 'te',
            'bn': 'bn',
            'mr': 'mr',
            'gu': 'gu',
            'kn': 'kn',
            'ml': 'ml',
            'pa': 'pa'
        }
        
        self.translations = self._load_translations()
        self.use_dynamic_translation = DYNAMIC_TRANSLATION_AVAILABLE
        
        # Translation cache to avoid redundant API calls
        # Format: {(text, target_lang): translated_text}
        self._translation_cache: Dict[tuple, str] = {}
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation data for all supported languages"""
        return {
            'en': {
                'welcome': 'Welcome to AgriSense XAI',
                'select_location': 'Select your location',
                'state': 'State',
                'district': 'District',
                'soil_parameters': 'Soil Parameters',
                'ph_level': 'pH Level',
                'nitrogen': 'Nitrogen',
                'phosphorus': 'Phosphorus',
                'potassium': 'Potassium',
                'moisture': 'Moisture',
                'get_recommendation': 'Get Crop Recommendation',
                'recommendations': 'Crop Recommendations',
                'explanations': 'Why this crop?',
                'weather': 'Weather Information',
                'loading': 'Loading...',
                'error': 'Error occurred',
                'rice': 'Rice',
                'wheat': 'Wheat',
                'maize': 'Maize',
                'cotton': 'Cotton',
                'sugarcane': 'Sugarcane',
                'banana': 'Banana',
                'mango': 'Mango',
                'grapes': 'Grapes',
                'watermelon': 'Watermelon',
                'coconut': 'Coconut',
                'simple_mode': 'Simple Mode',
                'manual_mode': 'Manual Mode',
                'select_location': 'Select Your Location',
                'city': 'City',
                'select_city': '-- Select City --',
                'select_state_first': '-- Select State First --',
                'location_selected': 'Location selected',
                'coordinates': 'Coordinates',
                'select_location_error': 'Please select both state and city',
                'select_and_submit': 'Select your location and click "Get Recommendation" to see crop suggestions.',
                'top_crops': 'Top Recommended Crops',
                'key_factors': 'Key Factors',
                'simple_mode_description': 'Just select your state and city. We\'ll automatically get weather data and provide crop recommendations!',
                'select_state': '-- Select State --',
                'optional': 'optional',
                'location': 'Location',
                'select_location_warning': 'Please select State and District to get location-specific weather data',
                'submit_to_see': 'Submit the form to see recommendations.',
                'api_status': 'API Status',
                'language': 'Language',
                'mode': 'Mode',
                'about': 'About'
            },
            'hi': {
                'welcome': 'अग्रीसेंस एक्सएआई में आपका स्वागत है',
                'select_location': 'अपना स्थान चुनें',
                'state': 'राज्य',
                'district': 'जिला',
                'soil_parameters': 'मिट्टी के मापदंड',
                'ph_level': 'पीएच स्तर',
                'nitrogen': 'नाइट्रोजन',
                'phosphorus': 'फॉस्फोरस',
                'potassium': 'पोटैशियम',
                'moisture': 'नमी',
                'get_recommendation': 'फसल सुझाव प्राप्त करें',
                'recommendations': 'फसल सुझाव',
                'explanations': 'यह फसल क्यों?',
                'weather': 'मौसम की जानकारी',
                'loading': 'लोड हो रहा है...',
                'error': 'त्रुटि हुई',
                'rice': 'चावल',
                'wheat': 'गेहूं',
                'maize': 'मक्का',
                'cotton': 'कपास',
                'sugarcane': 'गन्ना',
                'banana': 'केला',
                'mango': 'आम',
                'grapes': 'अंगूर',
                'watermelon': 'तरबूज',
                'coconut': 'नारियल',
                'simple_mode': 'सरल मोड',
                'manual_mode': 'मैनुअल मोड',
                'select_location': 'अपना स्थान चुनें',
                'city': 'शहर',
                'select_city': '-- शहर चुनें --',
                'select_state_first': '-- पहले राज्य चुनें --',
                'location_selected': 'स्थान चुना गया',
                'coordinates': 'निर्देशांक',
                'select_location_error': 'कृपया राज्य और शहर दोनों चुनें',
                'select_and_submit': 'अपना स्थान चुनें और "सुझाव प्राप्त करें" पर क्लिक करें',
                'top_crops': 'शीर्ष अनुशंसित फसलें',
                'key_factors': 'मुख्य कारक',
                'simple_mode_description': 'बस अपना राज्य और शहर चुनें। हम स्वचालित रूप से मौसम डेटा प्राप्त करेंगे और फसल सुझाव देंगे!',
                'select_state': '-- राज्य चुनें --',
                'optional': 'वैकल्पिक',
                'location': 'स्थान',
                'select_location_warning': 'कृपया स्थान-विशिष्ट मौसम डेटा प्राप्त करने के लिए राज्य और जिला चुनें',
                'submit_to_see': 'सुझाव देखने के लिए फॉर्म सबमिट करें।',
                'api_status': 'API स्थिति',
                'language': 'भाषा',
                'mode': 'मोड',
                'about': 'के बारे में'
            },
            'ta': {
                'welcome': 'அக்ரிசென்ஸ் எக்ஸ்ஏஐக்கு வரவேற்கிறோம்',
                'select_location': 'உங்கள் இடத்தைத் தேர்ந்தெடுக்கவும்',
                'state': 'மாநிலம்',
                'district': 'மாவட்டம்',
                'soil_parameters': 'மண் அளவுருக்கள்',
                'ph_level': 'pH நிலை',
                'nitrogen': 'நைட்ரஜன்',
                'phosphorus': 'பாஸ்பரஸ்',
                'potassium': 'பொட்டாசியம்',
                'moisture': 'ஈரப்பதம்',
                'get_recommendation': 'பயிர் பரிந்துரை பெறவும்',
                'recommendations': 'பயிர் பரிந்துரைகள்',
                'explanations': 'இந்த பயிர் ஏன்?',
                'weather': 'வானிலை தகவல்',
                'loading': 'ஏற்றுகிறது...',
                'error': 'பிழை ஏற்பட்டது',
                'rice': 'அரிசி',
                'wheat': 'கோதுமை',
                'maize': 'சோளம்',
                'cotton': 'பருத்தி',
                'sugarcane': 'கரும்பு',
                'banana': 'வாழை',
                'mango': 'மாம்பழம்',
                'grapes': 'திராட்சை',
                'watermelon': 'தர்பூசணி',
                'coconut': 'தேங்காய்',
                'simple_mode': 'எளிய முறை',
                'manual_mode': 'கைமுறை முறை',
                'select_location': 'உங்கள் இடத்தைத் தேர்ந்தெடுக்கவும்',
                'city': 'நகரம்',
                'select_city': '-- நகரத்தைத் தேர்ந்தெடுக்கவும் --',
                'select_state_first': '-- முதலில் மாநிலத்தைத் தேர்ந்தெடுக்கவும் --',
                'location_selected': 'இடம் தேர்ந்தெடுக்கப்பட்டது',
                'coordinates': 'ஆயத்தொலைவுகள்',
                'select_location_error': 'தயவுசெய்து மாநிலம் மற்றும் நகரம் இரண்டையும் தேர்ந்தெடுக்கவும்',
                'select_and_submit': 'உங்கள் இடத்தைத் தேர்ந்தெடுத்து "பரிந்துரை பெற" என்பதைக் கிளிக் செய்யவும்',
                'top_crops': 'முதன்மை பரிந்துரைக்கப்பட்ட பயிர்கள்',
                'key_factors': 'முக்கிய காரணிகள்',
                'simple_mode_description': 'உங்கள் மாநிலம் மற்றும் நகரத்தைத் தேர்ந்தெடுக்கவும்। நாங்கள் தானாக வானிலை மற்றும் மண் தரவைப் பெறுவோம்!',
                'select_state': '-- மாநிலத்தைத் தேர்ந்தெடுக்கவும் --',
                'optional': 'விருப்பமானது',
                'location': 'இடம்',
                'select_location_warning': 'இடம்-குறிப்பிட்ட வானிலை தரவைப் பெற மாநிலம் மற்றும் மாவட்டத்தைத் தேர்ந்தெடுக்கவும்',
                'submit_to_see': 'பரிந்துரைகளைப் பார்க்க படிவத்தை சமர்ப்பிக்கவும்.',
                'api_status': 'API நிலை',
                'language': 'மொழி',
                'mode': 'முறை',
                'about': 'பற்றி'
            },
            'te': {
                'welcome': 'అగ్రిసెన్స్ ఎక్స్ఏఐకు స్వాగతం',
                'select_location': 'మీ స్థానాన్ని ఎంచుకోండి',
                'state': 'రాష్ట్రం',
                'district': 'జిల్లా',
                'soil_parameters': 'నేల పరామితులు',
                'ph_level': 'pH స్థాయి',
                'nitrogen': 'నత్రజని',
                'phosphorus': 'భాస్వరం',
                'potassium': 'పొటాషియం',
                'moisture': 'తేమ',
                'get_recommendation': 'పంట సిఫారసు పొందండి',
                'recommendations': 'పంట సిఫారసులు',
                'explanations': 'ఈ పంట ఎందుకు?',
                'weather': 'వాతావరణ సమాచారం',
                'loading': 'లోడ్ అవుతోంది...',
                'error': 'లోపం సంభవించింది',
                'rice': 'వరి',
                'wheat': 'గోధుమ',
                'maize': 'మొక్కజొన్న',
                'cotton': 'పత్తి',
                'sugarcane': 'చెరకు',
                'banana': 'అరటి',
                'mango': 'మామిడి',
                'grapes': 'ద్రాక్ష',
                'watermelon': 'పుచ్చకాయ',
                'coconut': 'కొబ్బరి',
                'simple_mode': 'సాధారణ మోడ్',
                'manual_mode': 'మాన్యువల్ మోడ్',
                'select_location': 'మీ స్థానాన్ని ఎంచుకోండి',
                'city': 'నగరం',
                'select_city': '-- నగరాన్ని ఎంచుకోండి --',
                'select_state_first': '-- మొదట రాష్ట్రాన్ని ఎంచుకోండి --',
                'location_selected': 'స్థానం ఎంచుకోబడింది',
                'coordinates': 'సమన్వయాలు',
                'select_location_error': 'దయచేసి రాష్ట్రం మరియు నగరం రెండింటినీ ఎంచుకోండి',
                'select_and_submit': 'మీ స్థానాన్ని ఎంచుకొని "సిఫారసు పొందండి" క్లిక్ చేయండి',
                'top_crops': 'టాప్ సిఫారసు చేసిన పంటలు',
                'key_factors': 'ప్రధాన కారకాలు',
                'simple_mode_description': 'మీ రాష్ట్రం మరియు నగరాన్ని ఎంచుకోండి. మేము స్వయంచాలకంగా వాతావరణ డేటాను పొంది పంట సిఫారసులను అందిస్తాము!',
                'select_state': '-- రాష్ట్రాన్ని ఎంచుకోండి --',
                'optional': 'ఐచ్ఛికం',
                'location': 'స్థానం',
                'select_location_warning': 'స్థాన-నిర్దిష్ట వాతావరణ డేటాను పొందడానికి రాష్ట్రం మరియు జిల్లాను ఎంచుకోండి',
                'submit_to_see': 'సిఫారసులను చూడటానికి ఫారమ్ను సమర్పించండి.',
                'api_status': 'API స్థితి',
                'language': 'భాష',
                'mode': 'మోడ్',
                'about': 'గురించి'
            },
            'bn': {
                'welcome': 'অগ্রিসেন্স এক্সএআইতে স্বাগতম',
                'select_location': 'আপনার অবস্থান নির্বাচন করুন',
                'state': 'রাজ্য',
                'district': 'জেলা',
                'soil_parameters': 'মাটির পরামিতি',
                'ph_level': 'pH স্তর',
                'nitrogen': 'নাইট্রোজেন',
                'phosphorus': 'ফসফরাস',
                'potassium': 'পটাসিয়াম',
                'moisture': 'আর্দ্রতা',
                'get_recommendation': 'ফসলের সুপারিশ পান',
                'recommendations': 'ফসলের সুপারিশ',
                'explanations': 'এই ফসল কেন?',
                'weather': 'আবহাওয়ার তথ্য',
                'loading': 'লোড হচ্ছে...',
                'error': 'ত্রুটি ঘটেছে',
                'rice': 'ধান',
                'wheat': 'গম',
                'maize': 'ভুট্টা',
                'cotton': 'তুলা',
                'sugarcane': 'আখ',
                'banana': 'কলা',
                'mango': 'আম',
                'grapes': 'আঙ্গুর',
                'watermelon': 'তরমুজ',
                'coconut': 'নারকেল',
                'simple_mode': 'সরল মোড',
                'manual_mode': 'ম্যানুয়াল মোড',
                'select_location': 'আপনার অবস্থান নির্বাচন করুন',
                'city': 'শহর',
                'select_city': '-- শহর নির্বাচন করুন --',
                'select_state_first': '-- প্রথমে রাজ্য নির্বাচন করুন --',
                'location_selected': 'অবস্থান নির্বাচিত হয়েছে',
                'coordinates': 'স্থানাঙ্ক',
                'select_location_error': 'অনুগ্রহ করে রাজ্য এবং শহর উভয়ই নির্বাচন করুন',
                'select_and_submit': 'আপনার অবস্থান নির্বাচন করুন এবং "সুপারিশ পান" ক্লিক করুন',
                'top_crops': 'শীর্ষ সুপারিশকৃত ফসল',
                'key_factors': 'মূল কারণ',
                'simple_mode_description': 'শুধু আপনার রাজ্য এবং শহর নির্বাচন করুন। আমরা স্বয়ংক্রিয়ভাবে আবহাওয়ার ডেটা পাব এবং ফসলের সুপারিশ দেব!',
                'select_state': '-- রাজ্য নির্বাচন করুন --',
                'optional': 'ঐচ্ছিক',
                'location': 'অবস্থান',
                'select_location_warning': 'অবস্থান-নির্দিষ্ট আবহাওয়ার ডেটা পেতে রাজ্য এবং জেলা নির্বাচন করুন',
                'submit_to_see': 'সুপারিশ দেখতে ফর্ম জমা দিন।',
                'api_status': 'API অবস্থা',
                'language': 'ভাষা',
                'mode': 'মোড',
                'about': 'সম্পর্কে'
            },
            'ml': {
                'welcome': 'അഗ്രിസെൻസ് എക്സ്എഐയിലേക്ക് സ്വാഗതം',
                'select_location': 'നിങ്ങളുടെ സ്ഥാനം തിരഞ്ഞെടുക്കുക',
                'state': 'സംസ്ഥാനം',
                'district': 'ജില്ല',
                'soil_parameters': 'മണ്ണ് പാരാമീറ്ററുകൾ',
                'ph_level': 'pH നില',
                'nitrogen': 'നൈട്രജൻ',
                'phosphorus': 'ഫോസ്ഫറസ്',
                'potassium': 'പൊട്ടാസ്യം',
                'moisture': 'ഈർപ്പം',
                'get_recommendation': 'വിള ശുപാർശ നേടുക',
                'recommendations': 'വിള ശുപാർശകൾ',
                'explanations': 'ഈ വിള എന്തുകൊണ്ട്?',
                'weather': 'കാലാവസ്ഥാ വിവരം',
                'loading': 'ലോഡ് ചെയ്യുന്നു...',
                'error': 'പിശക് സംഭവിച്ചു',
                'rice': 'അരി',
                'wheat': 'ഗോതമ്പ്',
                'maize': 'ചോളം',
                'cotton': 'പരുത്തി',
                'sugarcane': 'ചെറുകച്ച',
                'banana': 'വാഴ',
                'mango': 'മാമ്പഴം',
                'grapes': 'മുന്തിരി',
                'watermelon': 'തണ്ണിമത്തൻ',
                'coconut': 'തെങ്ങ്',
                'simple_mode': 'ലളിത മോഡ്',
                'manual_mode': 'മാനുവൽ മോഡ്',
                'select_location': 'നിങ്ങളുടെ സ്ഥാനം തിരഞ്ഞെടുക്കുക',
                'city': 'നഗരം',
                'select_city': '-- നഗരം തിരഞ്ഞെടുക്കുക --',
                'select_state_first': '-- ആദ്യം സംസ്ഥാനം തിരഞ്ഞെടുക്കുക --',
                'location_selected': 'സ്ഥാനം തിരഞ്ഞെടുത്തു',
                'coordinates': 'കോർഡിനേറ്റുകൾ',
                'select_location_error': 'ദയവായി സംസ്ഥാനവും നഗരവും തിരഞ്ഞെടുക്കുക',
                'select_and_submit': 'നിങ്ങളുടെ സ്ഥാനം തിരഞ്ഞെടുത്ത് "ശുപാർശ നേടുക" ക്ലിക്ക് ചെയ്യുക',
                'top_crops': 'മികച്ച ശുപാർശ ചെയ്ത വിളകൾ',
                'key_factors': 'പ്രധാന ഘടകങ്ങൾ',
                'simple_mode_description': 'നിങ്ങളുടെ സംസ്ഥാനവും നഗരവും തിരഞ്ഞെടുക്കുക. ഞങ്ങൾ സ്വയം ക്രമീകരിച്ച് കാലാവസ്ഥാ ഡാറ്റ നേടുകയും വിള ശുപാർശകൾ നൽകുകയും ചെയ്യും!',
                'select_state': '-- സംസ്ഥാനം തിരഞ്ഞെടുക്കുക --',
                'optional': 'ഓപ്ഷണൽ',
                'location': 'സ്ഥാനം',
                'select_location_warning': 'സ്ഥാന-നിർദ്ദിഷ്ട കാലാവസ്ഥാ ഡാറ്റ നേടാൻ സംസ്ഥാനവും ജില്ലയും തിരഞ്ഞെടുക്കുക',
                'submit_to_see': 'ശുപാർശകൾ കാണാൻ ഫോം സമർപ്പിക്കുക.',
                'no_factors': 'വിശദമായ ഘടകങ്ങൾ ലഭ്യമല്ല',
                'api_status': 'API സ്ഥിതി',
                'language': 'ഭാഷ',
                'mode': 'മോഡ്',
                'about': 'കുറിച്ച്'
            }
        }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.supported_languages
    
    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target language"""
        if target_language not in self.translations:
            return text
        
        return self.translations[target_language].get(text, text)
    
    def translate_crop_name(self, crop_name: str, target_language: str) -> str:
        """Translate crop name to target language"""
        if target_language == 'en':
            return crop_name.capitalize()
        return self.translate_text(crop_name.lower(), target_language)
    
    def translate_feature_name(self, feature_name: str, target_language: str) -> str:
        """Translate feature name to target language"""
        if target_language == 'en':
            return feature_name
        
        feature_translations = {
            'hi': {
                'ph': 'पीएच',
                'nitrogen': 'नाइट्रोजन',
                'phosphorus': 'फॉस्फोरस',
                'potassium': 'पोटैशियम',
                'moisture': 'नमी',
                'temperature': 'तापमान',
                'humidity': 'आर्द्रता',
                'rainfall': 'वर्षा',
                'organic_matter': 'कार्बनिक पदार्थ'
            },
            'ta': {
                'ph': 'pH',
                'nitrogen': 'நைட்ரஜன்',
                'phosphorus': 'பாஸ்பரஸ்',
                'potassium': 'பொட்டாசியம்',
                'moisture': 'ஈரப்பதம்',
                'temperature': 'வெப்பநிலை',
                'humidity': 'ஈரப்பதம்',
                'rainfall': 'மழை',
                'organic_matter': 'கரிமப் பொருள்'
            },
            'te': {
                'ph': 'pH',
                'nitrogen': 'నత్రజని',
                'phosphorus': 'భాస్వరం',
                'potassium': 'పొటాషియం',
                'moisture': 'తేమ',
                'temperature': 'ఉష్ణోగ్రత',
                'humidity': 'తేమ',
                'rainfall': 'వర్షపాతం',
                'organic_matter': 'సేంద్రియ పదార్థం'
            },
            'bn': {
                'ph': 'pH',
                'nitrogen': 'নাইট্রোজেন',
                'phosphorus': 'ফসফরাস',
                'potassium': 'পটাসিয়াম',
                'moisture': 'আর্দ্রতা',
                'temperature': 'তাপমাত্রা',
                'humidity': 'আর্দ্রতা',
                'rainfall': 'বৃষ্টিপাত',
                'organic_matter': 'জৈব পদার্থ'
            }
        }
        
        return feature_translations.get(target_language, {}).get(feature_name, feature_name)
    
    def translate_state_name(self, state_name: str, target_language: str) -> str:
        """Translate state name to target language"""
        if target_language == 'en':
            return state_name
        
        # State name translations
        state_translations = {
            'hi': {
                'Karnataka': 'कर्नाटक',
                'Maharashtra': 'महाराष्ट्र',
                'Tamil Nadu': 'तमिलनाडु',
                'Uttar Pradesh': 'उत्तर प्रदेश',
                'West Bengal': 'पश्चिम बंगाल',
                'Gujarat': 'गुजरात',
                'Rajasthan': 'राजस्थान',
                'Madhya Pradesh': 'मध्य प्रदेश',
                'Andhra Pradesh': 'आंध्र प्रदेश',
                'Telangana': 'तेलंगाना',
                'Kerala': 'केरल',
                'Punjab': 'पंजाब',
                'Haryana': 'हरियाणा',
                'Bihar': 'बिहार',
                'Odisha': 'ओडिशा'
            },
            'ta': {
                'Karnataka': 'கர்நாடகா',
                'Maharashtra': 'மகாராஷ்டிரா',
                'Tamil Nadu': 'தமிழ்நாடு',
                'Uttar Pradesh': 'உத்தரப் பிரதேசம்',
                'West Bengal': 'மேற்கு வங்காளம்',
                'Gujarat': 'குஜராத்',
                'Rajasthan': 'ராஜஸ்தான்',
                'Madhya Pradesh': 'மத்திய பிரதேசம்',
                'Andhra Pradesh': 'ஆந்திர பிரதேசம்',
                'Telangana': 'தெலங்காணா',
                'Kerala': 'கேரளா',
                'Punjab': 'பஞ்சாப்',
                'Haryana': 'ஹரியானா',
                'Bihar': 'பீகார்',
                'Odisha': 'ஒடிசா'
            },
            'te': {
                'Karnataka': 'కర్ణాటక',
                'Maharashtra': 'మహారాష్ట్ర',
                'Tamil Nadu': 'తమిళనాడు',
                'Uttar Pradesh': 'ఉత్తర ప్రదేశ్',
                'West Bengal': 'పశ్చిమ బెంగాల్',
                'Gujarat': 'గుజరాత్',
                'Rajasthan': 'రాజస్థాన్',
                'Madhya Pradesh': 'మధ్య ప్రదేశ్',
                'Andhra Pradesh': 'ఆంధ్ర ప్రదేశ్',
                'Telangana': 'తెలంగాణ',
                'Kerala': 'కేరళ',
                'Punjab': 'పంజాబ్',
                'Haryana': 'హర్యానా',
                'Bihar': 'బీహార్',
                'Odisha': 'ఒడిశా'
            },
            'bn': {
                'Karnataka': 'কর্ণাটক',
                'Maharashtra': 'মহারাষ্ট্র',
                'Tamil Nadu': 'তামিলনাড়ু',
                'Uttar Pradesh': 'উত্তর প্রদেশ',
                'West Bengal': 'পশ্চিমবঙ্গ',
                'Gujarat': 'গুজরাট',
                'Rajasthan': 'রাজস্থান',
                'Madhya Pradesh': 'মধ্য প্রদেশ',
                'Andhra Pradesh': 'আন্ধ্র প্রদেশ',
                'Telangana': 'তেলেঙ্গানা',
                'Kerala': 'কেরালা',
                'Punjab': 'পাঞ্জাব',
                'Haryana': 'হরিয়ানা',
                'Bihar': 'বিহার',
                'Odisha': 'ওড়িশা'
            },
            'ml': {
                'Karnataka': 'കർണാടക',
                'Maharashtra': 'മഹാരാഷ്ട്ര',
                'Tamil Nadu': 'തമിഴ്നാട്',
                'Uttar Pradesh': 'ഉത്തർ പ്രദേശ്',
                'West Bengal': 'പശ്ചിമ ബംഗാൾ',
                'Gujarat': 'ഗുജറാത്ത്',
                'Rajasthan': 'രാജസ്ഥാൻ',
                'Madhya Pradesh': 'മധ്യ പ്രദേശ്',
                'Andhra Pradesh': 'ആന്ധ്ര പ്രദേശ്',
                'Telangana': 'തെലംഗാണ',
                'Kerala': 'കേരളം',
                'Punjab': 'പഞ്ചാബ്',
                'Haryana': 'ഹരിയാണ',
                'Bihar': 'ബിഹാർ',
                'Odisha': 'ഒഡീഷ'
            }
        }
        
        return state_translations.get(target_language, {}).get(state_name, state_name)
    
    def translate_city_name(self, city_name: str, target_language: str) -> str:
        """Translate city name to target language using dynamic translation"""
        if target_language == 'en':
            return city_name
        
        # Check cache first
        cache_key = (f"city:{city_name}", target_language)
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]
        
        # Use dynamic translation for city names (they may not be in static translations)
        try:
            translated = self.translate_dynamic(city_name, target_language)
            self._translation_cache[cache_key] = translated
            return translated
        except:
            return city_name
    
    def translate_dynamic(self, text: str, target_language: str, source_language: str = 'en') -> str:
        """
        Translate dynamic content using deep-translator (free Google Translate API).
        Translates ENTIRE sentences as complete text (not word-by-word).
        """
        if target_language == 'en' or not text or not text.strip():
            return text
        
        # Step 1: Check cache first to avoid redundant API calls
        cache_key = (text.strip(), target_language)
        if cache_key in self._translation_cache:
            logger.debug(f"Cache hit for translation: '{text[:30]}...'")
            return self._translation_cache[cache_key]
        
        # Step 2: For dynamic content, translate ENTIRE text via API (not partial word replacement)
        # Skip static translation check to avoid partial translations
        if not self.use_dynamic_translation:
            logger.warning("Dynamic translation not available")
            return text
        
        try:
            target_code = self.lang_code_map.get(target_language, target_language)
            source_code = self.lang_code_map.get(source_language, source_language)
            
            if target_code == source_code:
                return text
            
            # Use GoogleTranslator for dynamic translation (deep-translator is Python equivalent of google-translate-api)
            translator = GoogleTranslator(source=source_code, target=target_code)
            
            # Limit text length to avoid timeouts (max 5000 chars per Google Translate limit)
            text_to_translate = text[:5000] if len(text) > 5000 else text
            
            # Translate with timeout handling
            try:
                translated = translator.translate(text_to_translate)
                
                if translated and translated.strip() and translated != text_to_translate:
                    # Cache the translation
                    self._translation_cache[cache_key] = translated
                    return translated
                else:
                    # Translation failed or returned same - return original
                    return text
                    
            except Exception as translate_error:
                logger.warning(f"Translation API error for '{text[:30]}...': {translate_error}")
                # Return original text on error (don't try static to avoid recursion)
                return text
                
        except Exception as e:
            logger.error(f"Dynamic translation error: {e}")
            # Return original text on error
            return text
    
    def translate_batch(self, texts: List[str], target_language: str, source_language: str = 'en') -> List[str]:
        """
        Translate multiple texts in batches - ONLY for dynamic content.
        PRIORITIZES static translations (no API calls), then Gemini Pro, then deep-translator.
        OPTIMIZED: Checks cache first, checks static translations, deduplicates, and batches efficiently.
        """
        if target_language == 'en' or not texts:
            return texts
        
        # Step 1: Check cache for already translated texts
        cached_results = {}
        texts_to_check = []
        text_indices = []
        
        for i, text in enumerate(texts):
            cache_key = (text.strip(), target_language)
            if cache_key in self._translation_cache:
                cached_results[i] = self._translation_cache[cache_key]
            else:
                texts_to_check.append(text)
                text_indices.append(i)
        
        # Step 2: For dynamic content, translate ENTIRE texts via API (not partial word replacement)
        # Skip static translation check to ensure complete sentence translation
        texts_to_translate = texts_to_check
        
        # If all texts are cached, return immediately (no API calls!)
        if not texts_to_translate:
            logger.debug(f"All {len(texts)} texts found in cache - no API calls needed!")
            result = []
            for i in range(len(texts)):
                if i in cached_results:
                    result.append(cached_results[i])
                else:
                    result.append(texts[i])
            return result
        
        # Step 3: Deduplicate texts that need dynamic translation (same text = translate once)
        unique_texts = []
        text_to_unique_index = {}
        
        for text in texts_to_translate:
            text_stripped = text.strip()
            if text_stripped not in text_to_unique_index:
                unique_texts.append(text_stripped)
                text_to_unique_index[text_stripped] = len(unique_texts) - 1
        
        logger.info(f"Translating {len(unique_texts)} unique dynamic texts via deep-translator (from {len(texts_to_translate)} total, {len(texts)} original) - {len(cached_results)} cached")
        
        # Step 3: Batch translate unique texts using deep-translator (complete sentence translation)
        if not self.use_dynamic_translation:
            # If no dynamic translation available, return original texts
            translated_unique = unique_texts
        else:
            # Use deep-translator for unique texts
            try:
                target_code = self.lang_code_map.get(target_language, target_language)
                source_code = self.lang_code_map.get(source_language, source_language)
                
                if target_code == source_code:
                    translated_unique = unique_texts
                else:
                    translator = GoogleTranslator(source=source_code, target=target_code)
                    # Translate in batches for deep-translator
                    chunk_size = 10
                    translated_unique = []
                    for i in range(0, len(unique_texts), chunk_size):
                        chunk = unique_texts[i:i+chunk_size]
                        try:
                            sep = " |||SEP||| "
                            combined = sep.join(chunk)
                            translated_combined = translator.translate(combined)
                            chunk_translated = translated_combined.split(sep)
                            if len(chunk_translated) == len(chunk):
                                translated_unique.extend(chunk_translated)
                            else:
                                # Fallback to individual
                                for t in chunk:
                                    try:
                                        translated_unique.append(translator.translate(t[:5000]))
                                    except:
                                        translated_unique.append(t)
                        except Exception:
                            # Fallback to individual
                            for t in chunk:
                                try:
                                    translated_unique.append(translator.translate(t[:5000]))
                                except:
                                    translated_unique.append(t)
                    
                    # Cache translations
                    for orig, trans in zip(unique_texts, translated_unique):
                        if trans and trans.strip():
                            self._translation_cache[(orig, target_language)] = trans.strip()
            except Exception as e:
                logger.error(f"Deep-translator batch failed: {e}")
                translated_unique = unique_texts  # Return original on error
        
        # Step 4: Map unique translations back to all texts (including duplicates)
        unique_translation_map = dict(zip(unique_texts, translated_unique))
        
        # Build final result: use cached, then unique translations, then original
        result = []
        for i, text in enumerate(texts):
            if i in cached_results:
                result.append(cached_results[i])
            else:
                text_stripped = text.strip()
                if text_stripped in unique_translation_map:
                    result.append(unique_translation_map[text_stripped])
                else:
                    result.append(text)  # Fallback to original
        
        return result
    
    def translate_object_recursive(self, obj: Any, target_language: str, source_language: str = 'en') -> Any:
        """
        OPTIMIZED: Collect all strings first, then batch translate them.
        This reduces API calls from 20+ to 2-3 calls.
        """
        if target_language == 'en':
            return obj
        
        if obj is None:
            return obj
        
        # Collect all strings that need translation with their paths
        strings_to_translate = []
        paths = []
        
        def collect_strings(o, path=[]):
            if isinstance(o, str):
                text = o.strip()
                # Don't skip if it's a meaningful string (not just numbers or very short codes)
                if text and not text.replace('.', '').replace('-', '').replace(' ', '').replace('%', '').replace('°', '').isdigit():
                    # Only skip very short codes (1-2 chars) that are likely language codes, not crop names
                    # Crop names like "wheat", "rice" should be translated
                    if not (len(text) <= 2 and text.isalpha() and text.lower() in ['no', 'ok', 'hi', 'ta', 'te', 'bn', 'ml', 'en']):
                        strings_to_translate.append(text)
                        paths.append(path)
            elif isinstance(o, dict):
                for key, value in o.items():
                    collect_strings(value, path + [key])
            elif isinstance(o, list):
                for i, item in enumerate(o):
                    collect_strings(item, path + [i])
        
        collect_strings(obj)
        
        if not strings_to_translate:
            return obj
        
        # Batch translate all strings - split into chunks to avoid timeout
        logger.info(f"Batch translating {len(strings_to_translate)} strings to {target_language}")
        
        # OPTIMIZED: Use larger batches for Gemini (can handle 50+ texts in one call)
        # For deep-translator, keep smaller chunks
        if self.use_gemini:
            # Gemini can handle large batches - translate all at once (1 API call!)
            chunk_size = 50  # Large batch for Gemini
        else:
            chunk_size = 10  # Smaller for deep-translator
        
        translated_strings = []
        for i in range(0, len(strings_to_translate), chunk_size):
            chunk = strings_to_translate[i:i+chunk_size]
            logger.debug(f"Translating chunk {i//chunk_size + 1} ({len(chunk)} strings)")
            chunk_translated = self.translate_batch(chunk, target_language, source_language)
            translated_strings.extend(chunk_translated)
        
        # Create a mapping
        translation_map = dict(zip(strings_to_translate, translated_strings))
        
        # Apply translations back to the object
        def apply_translations(o, path=[]):
            if isinstance(o, str):
                text = o.strip()
                # Try exact match first
                if text in translation_map:
                    return translation_map[text]
                # Try without leading/trailing whitespace variations
                if text.strip() in translation_map:
                    return translation_map[text.strip()]
                return o
            elif isinstance(o, dict):
                return {key: apply_translations(value, path + [key]) for key, value in o.items()}
            elif isinstance(o, list):
                return [apply_translations(item, path + [i]) for i, item in enumerate(o)]
            else:
                return o
        
        return apply_translations(obj)
    
    def translate_explanation(self, explanation: str, target_language: str) -> str:
        """
        Translate explanation text using ONLY static translations (NO API calls, NO recursion).
        This is used for UI text, labels, and common phrases.
        """
        if target_language == 'en' or not explanation or not explanation.strip():
            return explanation
        
        if target_language not in self.translations:
            return explanation
        
        # Get full explanation translations (static only)
        explanation_translations = self._get_explanation_translations()
        lang_translations = explanation_translations.get(target_language, {})
        
        # Try to find exact match first
        if explanation in lang_translations:
            return lang_translations[explanation]
        
        # Start with the original explanation
        translated = explanation
        
        # Translate common patterns (longer patterns first to avoid partial matches)
        patterns = sorted(lang_translations.items(), key=lambda x: len(x[0]), reverse=True)
        for pattern, translation in patterns:
            if pattern.lower() in translated.lower():
                # Case-insensitive replacement
                translated = re.sub(re.escape(pattern), translation, translated, flags=re.IGNORECASE)
        
        # Translate crop names in the explanation
        for crop_key in ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'banana', 'mango', 'grapes', 'watermelon', 'coconut']:
            if crop_key in translated.lower():
                crop_translated = self.translate_crop_name(crop_key, target_language)
                translated = re.sub(re.escape(crop_key), crop_translated, translated, flags=re.IGNORECASE)
        
        # Translate feature names
        for feature_key in ['ph', 'nitrogen', 'phosphorus', 'potassium', 'moisture', 'temperature', 'humidity', 'rainfall', 'organic_matter']:
            if feature_key in translated.lower():
                feature_translated = self.translate_feature_name(feature_key, target_language)
                translated = re.sub(re.escape(feature_key), feature_translated, translated, flags=re.IGNORECASE)
        
        # Translate remaining keywords
        for key, value in self.translations[target_language].items():
            if key in translated.lower() and len(key) > 2:  # Only translate meaningful words
                translated = re.sub(r'\b' + re.escape(key) + r'\b', value, translated, flags=re.IGNORECASE)
        
        return translated
    
    def _get_explanation_translations(self) -> Dict[str, Dict[str, str]]:
        """Get full explanation translations for all languages"""
        return {
            'hi': {
                'is highly suitable for your location': 'आपके स्थान के लिए अत्यधिक उपयुक्त है',
                'is suitable for your location': 'आपके स्थान के लिए उपयुक्त है',
                'is moderately suitable for your location': 'आपके स्थान के लिए मध्यम रूप से उपयुक्त है',
                'requires warm, humid conditions with plenty of water': 'भरपूर पानी के साथ गर्म, आर्द्र परिस्थितियों की आवश्यकता है',
                'grows well in warm climates with moderate rainfall': 'मध्यम वर्षा वाली गर्म जलवायु में अच्छी तरह से बढ़ता है',
                'prefers cooler temperatures and moderate moisture': 'ठंडे तापमान और मध्यम नमी पसंद करता है',
                'needs warm weather and well-drained soil': 'गर्म मौसम और अच्छी जल निकासी वाली मिट्टी की आवश्यकता है',
                'requires tropical climate with high rainfall': 'उच्च वर्षा वाली उष्णकटिबंधीय जलवायु की आवश्यकता है',
                'is suitable for your conditions': 'आपकी परिस्थितियों के लिए उपयुक्त है',
                'Your': 'आपका',
                'level is particularly favorable': 'स्तर विशेष रूप से अनुकूल है',
                'level may need attention': 'स्तर पर ध्यान देने की आवश्यकता हो सकती है',
                'ph level': 'पीएच स्तर',
                'temperature': 'तापमान',
                'rainfall': 'वर्षा',
                'moisture': 'नमी',
                'humidity': 'आर्द्रता',
                'nitrogen': 'नाइट्रोजन',
                'phosphorus': 'फॉस्फोरस',
                'potassium': 'पोटैशियम',
                'organic matter': 'कार्बनिक पदार्थ',
                'positive': 'सकारात्मक',
                'negative': 'नकारात्मक',
                'neutral': 'तटस्थ'
            },
            'ta': {
                'is highly suitable for your location': 'உங்கள் இடத்திற்கு மிகவும் பொருத்தமானது',
                'is suitable for your location': 'உங்கள் இடத்திற்கு பொருத்தமானது',
                'is moderately suitable for your location': 'உங்கள் இடத்திற்கு மிதமாக பொருத்தமானது',
                'requires warm, humid conditions with plenty of water': 'நிறைய நீருடன் வெப்பமான, ஈரப்பதமான நிலைமைகள் தேவை',
                'grows well in warm climates with moderate rainfall': 'மிதமான மழையுடன் வெப்பமான காலநிலையில் நன்றாக வளரும்',
                'prefers cooler temperatures and moderate moisture': 'குளிர்ந்த வெப்பநிலை மற்றும் மிதமான ஈரப்பதத்தை விரும்புகிறது',
                'needs warm weather and well-drained soil': 'வெப்பமான வானிலை மற்றும் நன்றாக வடிகட்டப்பட்ட மண் தேவை',
                'requires tropical climate with high rainfall': 'அதிக மழையுடன் வெப்பமண்டல காலநிலை தேவை',
                'is suitable for your conditions': 'உங்கள் நிலைமைகளுக்கு பொருத்தமானது',
                'Your': 'உங்கள்',
                'level is particularly favorable': 'நிலை குறிப்பாக சாதகமானது',
                'level may need attention': 'நிலைக்கு கவனம் தேவைப்படலாம்',
                'ph level': 'pH நிலை',
                'temperature': 'வெப்பநிலை',
                'rainfall': 'மழை',
                'moisture': 'ஈரப்பதம்',
                'humidity': 'ஈரப்பதம்',
                'nitrogen': 'நைட்ரஜன்',
                'phosphorus': 'பாஸ்பரஸ்',
                'potassium': 'பொட்டாசியம்',
                'organic matter': 'கரிமப் பொருள்',
                'positive': 'நேர்மறை',
                'negative': 'எதிர்மறை',
                'neutral': 'நடுநிலை'
            },
            'te': {
                'is highly suitable for your location': 'మీ స్థానానికి చాలా అనుకూలంగా ఉంది',
                'is suitable for your location': 'మీ స్థానానికి అనుకూలంగా ఉంది',
                'is moderately suitable for your location': 'మీ స్థానానికి మధ్యస్థంగా అనుకూలంగా ఉంది',
                'requires warm, humid conditions with plenty of water': 'సమృద్ధమైన నీటితో వెచ్చని, తేమతో కూడిన పరిస్థితులు అవసరం',
                'grows well in warm climates with moderate rainfall': 'మధ్యస్థ వర్షపాతంతో వెచ్చని వాతావరణంలో బాగా పెరుగుతుంది',
                'prefers cooler temperatures and moderate moisture': 'చల్లని ఉష్ణోగ్రతలు మరియు మధ్యస్థ తేమను ఇష్టపడుతుంది',
                'needs warm weather and well-drained soil': 'వెచ్చని వాతావరణం మరియు బాగా నీరు వెళ్ళే నేల అవసరం',
                'requires tropical climate with high rainfall': 'అధిక వర్షపాతంతో ఉష్ణమండల వాతావరణం అవసరం',
                'is suitable for your conditions': 'మీ పరిస్థితులకు అనుకూలంగా ఉంది',
                'Your': 'మీ',
                'level is particularly favorable': 'స్థాయి ప్రత్యేకంగా అనుకూలంగా ఉంది',
                'level may need attention': 'స్థాయికి శ్రద్ధ అవసరం కావచ్చు',
                'ph level': 'pH స్థాయి',
                'temperature': 'ఉష్ణోగ్రత',
                'rainfall': 'వర్షపాతం',
                'moisture': 'తేమ',
                'humidity': 'తేమ',
                'nitrogen': 'నత్రజని',
                'phosphorus': 'భాస్వరం',
                'potassium': 'పొటాషియం',
                'organic matter': 'సేంద్రియ పదార్థం',
                'positive': 'సానుకూల',
                'negative': 'ప్రతికూల',
                'neutral': 'తటస్థ'
            },
            'bn': {
                'is highly suitable for your location': 'আপনার অবস্থানের জন্য অত্যন্ত উপযুক্ত',
                'is suitable for your location': 'আপনার অবস্থানের জন্য উপযুক্ত',
                'is moderately suitable for your location': 'আপনার অবস্থানের জন্য মাঝারিভাবে উপযুক্ত',
                'requires warm, humid conditions with plenty of water': 'প্রচুর জল সহ উষ্ণ, আর্দ্র অবস্থার প্রয়োজন',
                'grows well in warm climates with moderate rainfall': 'মাঝারি বৃষ্টিপাত সহ উষ্ণ জলবায়ুতে ভালোভাবে বৃদ্ধি পায়',
                'prefers cooler temperatures and moderate moisture': 'শীতল তাপমাত্রা এবং মাঝারি আর্দ্রতা পছন্দ করে',
                'needs warm weather and well-drained soil': 'উষ্ণ আবহাওয়া এবং ভালো নিষ্কাশনযুক্ত মাটি প্রয়োজন',
                'requires tropical climate with high rainfall': 'উচ্চ বৃষ্টিপাত সহ গ্রীষ্মমন্ডলীয় জলবায়ু প্রয়োজন',
                'is suitable for your conditions': 'আপনার অবস্থার জন্য উপযুক্ত',
                'Your': 'আপনার',
                'level is particularly favorable': 'স্তর বিশেষভাবে অনুকূল',
                'level may need attention': 'স্তরের মনোযোগ প্রয়োজন হতে পারে',
                'ph level': 'pH স্তর',
                'temperature': 'তাপমাত্রা',
                'rainfall': 'বৃষ্টিপাত',
                'moisture': 'আর্দ্রতা',
                'humidity': 'আর্দ্রতা',
                'nitrogen': 'নাইট্রোজেন',
                'phosphorus': 'ফসফরাস',
                'potassium': 'পটাসিয়াম',
                'organic matter': 'জৈব পদার্থ',
                'positive': 'ইতিবাচক',
                'negative': 'নেতিবাচক',
                'neutral': 'নিরপেক্ষ'
            }
        }
    
    def get_voice_commands(self, language: str) -> Dict[str, str]:
        """Get voice commands for specific language"""
        voice_commands = {
            'en': {
                'start': 'Start AgriSense',
                'select_location': 'Select location',
                'get_recommendation': 'Get recommendation',
                'explain': 'Explain recommendation',
                'change_language': 'Change language'
            },
            'hi': {
                'start': 'अग्रीसेंस शुरू करें',
                'select_location': 'स्थान चुनें',
                'get_recommendation': 'सुझाव प्राप्त करें',
                'explain': 'सुझाव समझाएं',
                'change_language': 'भाषा बदलें'
            },
            'ta': {
                'start': 'அக்ரிசென்ஸ் தொடங்கவும்',
                'select_location': 'இடத்தைத் தேர்ந்தெடுக்கவும்',
                'get_recommendation': 'பரிந்துரை பெறவும்',
                'explain': 'பரிந்துரையை விளக்கவும்',
                'change_language': 'மொழியை மாற்றவும்'
            }
        }
        
        return voice_commands.get(language, voice_commands['en'])
    
    def is_voice_supported(self, language: str) -> bool:
        """Check if voice support is available for language"""
        return language in ['en', 'hi', 'ta', 'te', 'bn']
    
    def get_tts_voice_id(self, language: str) -> str:
        """Get Text-to-Speech voice ID for language"""
        voice_mapping = {
            'en': 'en-US-Standard-A',
            'hi': 'hi-IN-Wavenet-A',
            'ta': 'ta-IN-Standard-A',
            'te': 'te-IN-Standard-A',
            'bn': 'bn-IN-Standard-A'
        }
        return voice_mapping.get(language, 'en-US-Standard-A')

