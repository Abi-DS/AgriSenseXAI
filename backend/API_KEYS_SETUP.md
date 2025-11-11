# API Keys Setup Guide

## Where to Get API Keys

### 1. OpenWeatherMap API (Recommended - Free Tier Available)
- **Website**: https://openweathermap.org/api
- **Sign up**: https://home.openweathermap.org/users/sign_up
- **Free tier**: 60 API calls/minute, 1,000,000 calls/month
- **Steps**:
  1. Sign up for a free account
  2. Go to "API keys" section
  3. Generate a new API key (it may take 10-15 minutes to activate)
  4. Copy your API key

### 2. WeatherAPI.com (Alternative - Free Tier Available)
- **Website**: https://www.weatherapi.com/
- **Sign up**: https://www.weatherapi.com/signup.aspx
- **Free tier**: 1 million calls/month
- **Steps**:
  1. Sign up for a free account
  2. Go to your dashboard
  3. Copy your API key from the dashboard

## Where to Put Your API Keys

### Option 1: Using .env file (Recommended - More Secure)

1. **Create a `.env` file** in the `backend/` directory:
   ```
   backend/
   ├── .env          <-- Create this file
   ├── app/
   ├── modules/
   └── requirements.txt
   ```

2. **Add your API keys** to the `.env` file:
   ```env
   OPENWEATHER_API_KEY=your_actual_openweather_api_key_here
   WEATHERAPI_KEY=your_actual_weatherapi_key_here
   ```

3. **Important**: Make sure `.env` is in your `.gitignore` file so you don't commit your keys to version control!

### Option 2: Direct Code Edit (Not Recommended for Production)

If you don't want to use environment variables, you can directly edit:
- **File**: `backend/modules/data_acquisition/weather_apis.py`
- **Lines 25-26**: Replace `"demo_key"` with your actual API keys

```python
self.openweather_api_key = "your_actual_key_here"
self.weatherapi_key = "your_actual_key_here"
```

## How It Works

- The app tries **both APIs** simultaneously
- If one fails, it automatically tries the other
- If both fail (or no keys are provided), it uses **fallback demo data**
- The app will work without API keys, but will use default weather values

## Testing Your API Keys

After adding your keys, restart the backend server:
```powershell
# Stop the server (Ctrl+C or kill the process)
# Then restart:
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

When you make a recommendation request, check the weather summary in the response. If it shows:
- `Source: OpenWeatherMap` or `Source: WeatherAPI` = Your API keys are working! ✅
- `Source: Fallback` = API keys not working, using demo data ⚠️

## Notes

- **You only need ONE API key** - either OpenWeatherMap OR WeatherAPI.com
- The app will work fine with just demo data if you don't add keys
- Free tiers are usually sufficient for development and testing
- Never share your API keys publicly or commit them to Git!

