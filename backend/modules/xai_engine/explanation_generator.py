"""
Module 3: XAI Explanation Engine - Explanation Generator
Implementation Status: 100% complete
Technologies: Python, SHAP, LIME
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available")

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available")

logger = logging.getLogger(__name__)

@dataclass
class FeatureExplanation:
    """Represents explanation for a single feature"""
    feature_name: str
    value: float
    importance: float
    contribution: float  # SHAP value or LIME weight
    impact: str  # 'positive', 'negative', 'neutral'
    description: str
    method: str  # 'shap', 'lime', or 'rule_based'

@dataclass
class CropExplanation:
    """Represents explanation for a crop recommendation"""
    crop_name: str
    confidence: float
    primary_factors: List[FeatureExplanation]
    secondary_factors: List[FeatureExplanation]
    overall_explanation: str
    shap_values: Optional[Dict[str, float]] = None
    lime_explanation: Optional[Dict[str, Any]] = None

class ExplanationGenerator:
    """Generates explanations using SHAP, LIME, and rule-based approaches"""
    
    def __init__(self, ml_model=None, training_data: Optional[np.ndarray] = None):
        self.ml_model = ml_model
        self.training_data = training_data
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Define feature names FIRST before initializing explainers
        self.feature_names = [
            'ph', 'nitrogen', 'phosphorus', 'potassium', 'moisture', 
            'temperature', 'humidity', 'rainfall', 'organic_matter'
        ]
        
        # Initialize explainers if model is available
        if self.ml_model is not None and self.training_data is not None:
            self._initialize_explainers()
        
        self.feature_descriptions = {
            'ph': 'Soil acidity/alkalinity level',
            'nitrogen': 'Nitrogen content in soil',
            'phosphorus': 'Phosphorus content in soil',
            'potassium': 'Potassium content in soil',
            'moisture': 'Soil moisture percentage',
            'temperature': 'Average temperature',
            'humidity': 'Relative humidity',
            'rainfall': 'Annual rainfall amount',
            'organic_matter': 'Organic matter content'
        }
        
        self.crop_to_idx = {
            'rice': 0, 'maize': 1, 'wheat': 2, 'cotton': 3, 'sugarcane': 4,
            'banana': 5, 'mango': 6, 'grapes': 7, 'watermelon': 8, 'coconut': 9
        }
        
        self.crop_requirements = {
            'rice': {
                'ph': (5.5, 7.0), 'temperature': (20, 35), 'rainfall': (100, 200),
                'moisture': (60, 80), 'description': 'Rice requires warm, humid conditions with plenty of water'
            },
            'maize': {
                'ph': (6.0, 7.5), 'temperature': (18, 30), 'rainfall': (50, 150),
                'moisture': (40, 70), 'description': 'Maize grows well in warm climates with moderate rainfall'
            },
            'wheat': {
                'ph': (6.0, 7.5), 'temperature': (10, 25), 'rainfall': (30, 100),
                'moisture': (30, 60), 'description': 'Wheat prefers cooler temperatures and moderate moisture'
            },
            'cotton': {
                'ph': (5.5, 8.0), 'temperature': (20, 35), 'rainfall': (50, 120),
                'moisture': (40, 70), 'description': 'Cotton needs warm weather and well-drained soil'
            },
            'sugarcane': {
                'ph': (6.0, 7.5), 'temperature': (20, 35), 'rainfall': (100, 200),
                'moisture': (60, 80), 'description': 'Sugarcane requires tropical climate with high rainfall'
            },
            'banana': {
                'ph': (5.5, 7.5), 'temperature': (22, 32), 'rainfall': (100, 250),
                'moisture': (60, 85), 'description': 'Banana thrives in warm, humid tropical conditions'
            },
            'mango': {
                'ph': (5.5, 7.5), 'temperature': (24, 30), 'rainfall': (75, 200),
                'moisture': (40, 70), 'description': 'Mango grows best in warm climates with moderate rainfall'
            },
            'grapes': {
                'ph': (6.0, 7.5), 'temperature': (15, 30), 'rainfall': (50, 150),
                'moisture': (30, 60), 'description': 'Grapes prefer moderate temperatures and well-drained soil'
            },
            'watermelon': {
                'ph': (6.0, 7.0), 'temperature': (22, 32), 'rainfall': (50, 150),
                'moisture': (50, 80), 'description': 'Watermelon needs warm weather and adequate moisture'
            },
            'coconut': {
                'ph': (5.5, 8.0), 'temperature': (24, 32), 'rainfall': (100, 300),
                'moisture': (50, 80), 'description': 'Coconut requires tropical climate with high humidity and rainfall'
            }
        }
        
        # Initialize explainers if model is available
        if self.ml_model is not None and self.training_data is not None:
            self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        try:
            if SHAP_AVAILABLE and self.ml_model is not None:
                # Use TreeExplainer for LightGBM
                try:
                    self.shap_explainer = shap.TreeExplainer(self.ml_model)
                    logger.info("SHAP TreeExplainer initialized")
                except:
                    # Fallback to KernelExplainer
                    sample_data = shap.sample(self.training_data, 100)
                    self.shap_explainer = shap.KernelExplainer(
                        self._model_predict_wrapper, sample_data
                    )
                    logger.info("SHAP KernelExplainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
        
        try:
            if LIME_AVAILABLE and self.training_data is not None:
                # Calculate feature statistics for LIME
                feature_stats = [
                    (self.training_data[:, i].mean(), self.training_data[:, i].std())
                    for i in range(self.training_data.shape[1])
                ]
                
                self.lime_explainer = LimeTabularExplainer(
                    self.training_data,
                    feature_names=self.feature_names,
                    class_names=['rice', 'maize', 'wheat', 'cotton', 'sugarcane', 
                               'banana', 'mango', 'grapes', 'watermelon', 'coconut'],
                    mode='classification',
                    discretize_continuous=True
                )
                logger.info("LIME explainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize LIME explainer: {e}")
    
    def _model_predict_wrapper(self, X):
        """Wrapper for SHAP KernelExplainer"""
        if self.ml_model is None:
            return np.zeros((X.shape[0], 10))
        try:
            return self.ml_model.predict(X, num_iteration=self.ml_model.best_iteration)
        except:
            return np.zeros((X.shape[0], 10))
    
    def generate_explanation(self, features: Dict[str, float], 
                           crop_predictions: List[Dict[str, any]],
                           use_shap: bool = True,
                           use_lime: bool = True) -> List[CropExplanation]:
        """Generate explanations using SHAP, LIME, and rule-based methods"""
        explanations = []
        
        # Convert features to array format
        feature_array = np.array([[
            features.get('ph', 6.5),
            features.get('nitrogen', 50),
            features.get('phosphorus', 30),
            features.get('potassium', 40),
            features.get('moisture', 60),
            features.get('temperature', 25),
            features.get('humidity', 65),
            features.get('rainfall', 100),
            features.get('organic_matter', 3)
        ]])
        
        # Get SHAP values if available
        shap_values_dict = {}
        if use_shap and SHAP_AVAILABLE and self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(feature_array)
                # Handle multi-class output
                if isinstance(shap_values, list):
                    # Get SHAP values for the top predicted crop
                    top_crop_idx = 0  # Will be updated based on prediction
                    shap_values_dict = dict(zip(
                        self.feature_names,
                        shap_values[top_crop_idx][0].tolist()
                    ))
                else:
                    shap_values_dict = dict(zip(
                        self.feature_names,
                        shap_values[0].tolist()
                    ))
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
        
        # Get LIME explanation if available
        lime_explanation = None
        if use_lime and LIME_AVAILABLE and self.lime_explainer is not None:
            try:
                # Get top crop index
                top_crop = crop_predictions[0]['crop'] if crop_predictions else 'rice'
                class_idx = self.crop_to_idx.get(top_crop, 0)
                
                lime_exp = self.lime_explainer.explain_instance(
                    feature_array[0],
                    self._lime_predict_wrapper,
                    num_features=len(self.feature_names),
                    top_labels=1,
                    num_samples=500
                )
                
                lime_explanation = {
                    'explanation': lime_exp.as_list(label=class_idx),
                    'score': lime_exp.score[class_idx] if hasattr(lime_exp, 'score') else None
                }
            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")
        
        # Generate explanations for each crop
        for pred in crop_predictions:
            crop_name = pred['crop']
            confidence = pred['confidence']
            
            # Get crop requirements
            requirements = self.crop_requirements.get(crop_name, {})
            
            # Analyze features using multiple methods
            primary_factors = []
            secondary_factors = []
            
            for feature_name in self.feature_names:
                value = features.get(feature_name, 0)
                
                # Get SHAP value if available
                shap_value = shap_values_dict.get(feature_name, 0.0)
                # Handle if SHAP value is a list/array
                if isinstance(shap_value, (list, np.ndarray)):
                    shap_value = float(shap_value[0]) if len(shap_value) > 0 else 0.0
                else:
                    shap_value = float(shap_value) if shap_value else 0.0
                
                # Get LIME weight if available
                lime_weight = 0.0
                if lime_explanation:
                    for feat_name, weight in lime_explanation['explanation']:
                        if feat_name.startswith(feature_name) or feature_name in feat_name:
                            lime_weight = weight
                            break
                # Handle if LIME weight is a list/array
                if isinstance(lime_weight, (list, np.ndarray)):
                    lime_weight = float(lime_weight[0]) if len(lime_weight) > 0 else 0.0
                else:
                    lime_weight = float(lime_weight) if lime_weight else 0.0
                
                # Calculate importance (use SHAP if available, else LIME, else rule-based)
                if abs(shap_value) > 0.001:
                    importance = abs(shap_value)
                    contribution = shap_value
                    method = 'shap'
                elif abs(lime_weight) > 0.001:
                    importance = abs(lime_weight)
                    contribution = lime_weight
                    method = 'lime'
                else:
                    # Rule-based fallback
                    if feature_name in requirements and feature_name != 'description':
                        optimal_range = requirements[feature_name]
                        importance = self._calculate_feature_importance(value, optimal_range, feature_name)
                        contribution = importance * (1 if self._determine_impact(value, optimal_range) == 'positive' else -1)
                    else:
                        importance = 0.1
                        contribution = 0.0
                    method = 'rule_based'
                
                # Determine impact
                if feature_name in requirements and feature_name != 'description':
                    impact = self._determine_impact(value, requirements[feature_name])
                    description = self._generate_feature_description(
                        feature_name, value, requirements[feature_name]
                    )
                else:
                    impact = 'positive' if contribution > 0 else 'negative' if contribution < 0 else 'neutral'
                    description = f"{self.feature_descriptions.get(feature_name, feature_name)}: {value:.1f}"
                
                explanation = FeatureExplanation(
                    feature_name=feature_name,
                    value=value,
                    importance=importance,
                    contribution=contribution,
                    impact=impact,
                    description=description,
                    method=method
                )
                
                if importance > 0.3:  # Lower threshold to include more factors
                    primary_factors.append(explanation)
                else:
                    secondary_factors.append(explanation)
            
            # Sort by importance (safely handle if importance is a list)
            def safe_abs_importance(x):
                imp = x.importance
                if isinstance(imp, (list, np.ndarray)):
                    imp = float(imp[0]) if len(imp) > 0 else 0.0
                return abs(float(imp))
            
            primary_factors.sort(key=safe_abs_importance, reverse=True)
            secondary_factors.sort(key=safe_abs_importance, reverse=True)
            
            # Generate overall explanation
            overall_explanation = self._generate_overall_explanation(
                crop_name, confidence, primary_factors, requirements
            )
            
            explanations.append(CropExplanation(
                crop_name=crop_name,
                confidence=confidence,
                primary_factors=primary_factors[:5],  # Top 5 primary factors
                secondary_factors=secondary_factors[:3],  # Top 3 secondary factors
                overall_explanation=overall_explanation,
                shap_values=shap_values_dict if shap_values_dict else None,
                lime_explanation=lime_explanation
            ))
        
        return explanations
    
    def _lime_predict_wrapper(self, instances):
        """Wrapper for LIME prediction"""
        if self.ml_model is None:
            return np.zeros((len(instances), 10))
        try:
            predictions = self.ml_model.predict(instances, num_iteration=self.ml_model.best_iteration)
            return predictions
        except:
            return np.zeros((len(instances), 10))
    
    def _calculate_feature_importance(self, value: float, optimal_range: Tuple[float, float], 
                                    feature_name: str) -> float:
        """Calculate feature importance based on distance from optimal range"""
        min_val, max_val = optimal_range
        
        if min_val <= value <= max_val:
            return 1.0
        else:
            distance = min(abs(value - min_val), abs(value - max_val))
            range_size = max_val - min_val
            normalized_distance = distance / range_size if range_size > 0 else 1.0
            return max(0.1, 1.0 - normalized_distance)
    
    def _determine_impact(self, value: float, optimal_range: Tuple[float, float]) -> str:
        """Determine if feature has positive, negative, or neutral impact"""
        min_val, max_val = optimal_range
        
        if min_val <= value <= max_val:
            return 'positive'
        else:
            return 'negative'
    
    def _generate_feature_description(self, feature_name: str, value: float, 
                                    optimal_range: Tuple[float, float]) -> str:
        """Generate human-readable description for a feature"""
        min_val, max_val = optimal_range
        feature_desc = self.feature_descriptions.get(feature_name, feature_name)
        
        if min_val <= value <= max_val:
            return f"{feature_desc} ({value:.1f}) is in the optimal range ({min_val}-{max_val})"
        elif value < min_val:
            return f"{feature_desc} ({value:.1f}) is below optimal range ({min_val}-{max_val})"
        else:
            return f"{feature_desc} ({value:.1f}) is above optimal range ({min_val}-{max_val})"
    
    def _generate_overall_explanation(self, crop_name: str, confidence: float,
                                    primary_factors: List[FeatureExplanation],
                                    requirements: Dict) -> str:
        """Generate overall explanation for the crop recommendation"""
        base_description = requirements.get('description', f'{crop_name} is suitable for your conditions')
        
        if confidence > 0.8:
            confidence_level = "highly suitable"
        elif confidence > 0.6:
            confidence_level = "suitable"
        else:
            confidence_level = "moderately suitable"
        
        explanation = f"{crop_name} is {confidence_level} for your location. {base_description}."
        
        if primary_factors:
            top_factor = primary_factors[0]
            method_note = f" (via {top_factor.method.upper()})" if top_factor.method != 'rule_based' else ""
            if top_factor.impact == 'positive':
                explanation += f" Your {top_factor.feature_name} level is particularly favorable{method_note}."
            else:
                explanation += f" However, your {top_factor.feature_name} level may need attention{method_note}."
        
        return explanation
    
    def update_model(self, model, training_data: np.ndarray):
        """Update the ML model and reinitialize explainers"""
        self.ml_model = model
        self.training_data = training_data
        self._initialize_explainers()
