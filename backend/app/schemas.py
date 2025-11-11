from typing import List, Optional, Dict
from pydantic import BaseModel, Field, confloat


class HealthResponse(BaseModel):
	status: str = "ok"


class RecommendRequest(BaseModel):
	ph: confloat(ge=0.0, le=14.0)
	nitrogen: confloat(ge=0.0) = Field(..., description="Nitrogen content (kg/ha or ppm equivalent)")
	phosphorus: confloat(ge=0.0)
	potassium: confloat(ge=0.0)
	moisture: confloat(ge=0.0, le=100.0)
	latitude: float
	longitude: float
	season: Optional[str] = Field(default=None, description="Optional season override")
	use_weather: bool = Field(default=True, description="Whether to enrich with weather data")
	language: str = Field(default="en", description="Interface language (en, hi, ta, te, bn)")


class CropScore(BaseModel):
	crop: str
	score: float
	crop_translated: Optional[str] = None  # Translated crop name


class Attribution(BaseModel):
	feature: str
	importance: float
	contribution: float
	direction: str
	feature_translated: Optional[str] = None  # Translated feature name
	direction_translated: Optional[str] = None  # Translated direction
	description: Optional[str] = None  # Original description
	description_translated: Optional[str] = None  # Translated description
	method: Optional[str] = None  # XAI method: 'shap', 'lime', or 'rule_based'


class Explanation(BaseModel):
	crop: str
	text: str
	attributions: List[Attribution]
	crop_translated: Optional[str] = None  # Translated crop name
	text_translated: Optional[str] = None  # Translated explanation text


class RecommendResponse(BaseModel):
	top_crops: List[CropScore] = Field(..., min_items=1)
	explanations: List[Explanation]
	model_version: str
	weather_summary: Optional[str] = None


class LanguageResponse(BaseModel):
	language: str
	translations: Dict[str, str]
	voice_supported: bool = True


"""Auth models removed for public demo."""



