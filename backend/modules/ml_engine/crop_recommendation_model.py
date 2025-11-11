"""
Module 2: ML Recommendation Engine - Crop Recommendation Model
Implementation Status: 100% complete
Technologies: Python, Scikit-learn, LightGBM
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import joblib
import os

try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available, using fallback model")

logger = logging.getLogger(__name__)

class CropRecommendationModel:
    """LightGBM-based crop recommendation model with proper training and persistence"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder() if LIGHTGBM_AVAILABLE else None
        self.feature_names = [
            'ph', 'nitrogen', 'phosphorus', 'potassium', 'moisture', 
            'temperature', 'humidity', 'rainfall', 'organic_matter'
        ]
        self.crop_labels = [
            'rice', 'maize', 'wheat', 'cotton', 'sugarcane',
            'banana', 'mango', 'grapes', 'watermelon', 'coconut'
        ]
        self.model_dir = Path(__file__).parent / "models"
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / "crop_model.pkl"
        self.encoder_path = self.model_dir / "label_encoder.pkl"
        
    def generate_synthetic_data(self, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic synthetic agricultural data for training"""
        np.random.seed(42)
        random.seed(42)
        
        data = []
        labels = []
        
        # Crop-specific parameter distributions
        crop_configs = {
            'rice': {
                'ph': (5.5, 7.0), 'nitrogen': (80, 150), 'phosphorus': (20, 50),
                'potassium': (100, 200), 'moisture': (60, 80), 'temperature': (20, 35),
                'humidity': (70, 90), 'rainfall': (100, 200), 'organic_matter': (2.0, 5.0)
            },
            'maize': {
                'ph': (6.0, 7.5), 'nitrogen': (100, 180), 'phosphorus': (30, 60),
                'potassium': (80, 150), 'moisture': (40, 70), 'temperature': (18, 30),
                'humidity': (50, 80), 'rainfall': (50, 150), 'organic_matter': (2.5, 5.5)
            },
            'wheat': {
                'ph': (6.0, 7.5), 'nitrogen': (60, 120), 'phosphorus': (20, 40),
                'potassium': (50, 100), 'moisture': (30, 60), 'temperature': (10, 25),
                'humidity': (40, 70), 'rainfall': (30, 100), 'organic_matter': (1.5, 4.0)
            },
            'cotton': {
                'ph': (5.5, 8.0), 'nitrogen': (70, 130), 'phosphorus': (25, 50),
                'potassium': (90, 180), 'moisture': (40, 70), 'temperature': (20, 35),
                'humidity': (50, 80), 'rainfall': (50, 120), 'organic_matter': (2.0, 5.0)
            },
            'sugarcane': {
                'ph': (6.0, 7.5), 'nitrogen': (90, 160), 'phosphorus': (30, 60),
                'potassium': (120, 250), 'moisture': (60, 80), 'temperature': (20, 35),
                'humidity': (65, 85), 'rainfall': (100, 200), 'organic_matter': (3.0, 6.0)
            },
            'banana': {
                'ph': (5.5, 7.5), 'nitrogen': (100, 200), 'phosphorus': (30, 70),
                'potassium': (200, 300), 'moisture': (60, 85), 'temperature': (22, 32),
                'humidity': (70, 90), 'rainfall': (100, 250), 'organic_matter': (3.0, 6.0)
            },
            'mango': {
                'ph': (5.5, 7.5), 'nitrogen': (60, 120), 'phosphorus': (20, 50),
                'potassium': (80, 150), 'moisture': (40, 70), 'temperature': (24, 30),
                'humidity': (60, 85), 'rainfall': (75, 200), 'organic_matter': (2.0, 5.0)
            },
            'grapes': {
                'ph': (6.0, 7.5), 'nitrogen': (50, 100), 'phosphorus': (20, 40),
                'potassium': (100, 200), 'moisture': (30, 60), 'temperature': (15, 30),
                'humidity': (50, 75), 'rainfall': (50, 150), 'organic_matter': (1.5, 4.0)
            },
            'watermelon': {
                'ph': (6.0, 7.0), 'nitrogen': (80, 150), 'phosphorus': (30, 60),
                'potassium': (100, 200), 'moisture': (50, 80), 'temperature': (22, 32),
                'humidity': (60, 85), 'rainfall': (50, 150), 'organic_matter': (2.0, 5.0)
            },
            'coconut': {
                'ph': (5.5, 8.0), 'nitrogen': (50, 100), 'phosphorus': (20, 40),
                'potassium': (100, 200), 'moisture': (50, 80), 'temperature': (24, 32),
                'humidity': (70, 90), 'rainfall': (100, 300), 'organic_matter': (2.0, 5.0)
            }
        }
        
        samples_per_crop = n_samples // len(self.crop_labels)
        
        for crop_name, config in crop_configs.items():
            for _ in range(samples_per_crop):
                # Generate features based on crop-specific distributions
                features = []
                for feature_name in self.feature_names:
                    if feature_name in config:
                        min_val, max_val = config[feature_name]
                        # Add some noise for realism
                        value = np.random.uniform(min_val * 0.8, max_val * 1.2)
                        features.append(value)
                    else:
                        # Default range if not specified
                        features.append(np.random.uniform(0, 100))
                
                data.append(features)
                labels.append(crop_name)
        
        # Add some random samples for diversity
        for _ in range(n_samples % len(self.crop_labels)):
            features = [
                np.random.uniform(4.0, 9.0),  # ph
                np.random.uniform(10, 200),   # nitrogen
                np.random.uniform(5, 100),    # phosphorus
                np.random.uniform(10, 300),   # potassium
                np.random.uniform(10, 90),    # moisture
                np.random.uniform(15, 40),    # temperature
                np.random.uniform(20, 90),    # humidity
                np.random.uniform(0, 500),   # rainfall
                np.random.uniform(0.5, 8.0)  # organic_matter
            ]
            data.append(features)
            labels.append(random.choice(self.crop_labels))
        
        return np.array(data), np.array(labels)
    
    def train_model(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, float]:
        """Train LightGBM model with proper validation"""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, using fallback")
            return {'error': 'LightGBM not available'}
        
        try:
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            
            # Create LightGBM dataset
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            # Model parameters - tuned for better sensitivity to weather features
            params = {
                'objective': 'multiclass',
                'num_class': len(self.crop_labels),
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 50,  # Increased for more complexity
                'learning_rate': 0.03,  # Lower learning rate for better generalization
                'feature_fraction': 0.85,  # Slightly lower to force use of all features
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 20,  # Prevent overfitting
                'max_depth': 8,  # Allow deeper trees to capture weather patterns
                'verbose': -1,
                'random_state': 42
            }
            
            # Train model
            self.model = lgb.train(
                params,
                train_data,
                valid_sets=[test_data],
                num_boost_round=200,
                callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
            )
            
            # Evaluate
            y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
            y_pred_class = np.argmax(y_pred, axis=1)
            accuracy = accuracy_score(y_test, y_pred_class)
            
            # Save model
            self.save_model()
            
            logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
            
            return {
                'accuracy': float(accuracy),
                'n_samples': len(X),
                'n_features': len(self.feature_names),
                'n_classes': len(self.crop_labels),
                'best_iteration': int(self.model.best_iteration)
            }
            
        except Exception as e:
            logger.error(f"Model training error: {e}", exc_info=True)
            return {'error': str(e)}
    
    def save_model(self):
        """Save trained model and encoder"""
        try:
            if self.model is not None:
                joblib.dump(self.model, self.model_path)
                if self.label_encoder is not None:
                    joblib.dump(self.label_encoder, self.encoder_path)
                logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self) -> bool:
        """Load pre-trained model"""
        try:
            if self.model_path.exists() and self.encoder_path.exists():
                self.model = joblib.load(self.model_path)
                self.label_encoder = joblib.load(self.encoder_path)
                logger.info("Model loaded successfully")
                return True
            else:
                logger.warning("Model files not found")
                return False
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            return False
    
    def predict_crops(self, features: Dict[str, float], top_k: int = 5) -> List[Dict[str, any]]:
        """Predict top-k crop recommendations"""
        try:
            if self.model is None:
                if not self.load_model():
                    # Train a new model if none exists
                    logger.info("Training new model...")
                    X, y = self.generate_synthetic_data(5000)
                    self.train_model(X, y)
            
            if self.model is None:
                raise ValueError("Model not available")
            
            # Prepare feature vector
            feature_vector = np.array([[
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
            
            # Predict probabilities
            probabilities = self.model.predict(feature_vector, num_iteration=self.model.best_iteration)[0]
            
            # Get top-k predictions
            top_indices = np.argsort(probabilities)[::-1][:top_k]
            
            results = []
            for rank, idx in enumerate(top_indices, 1):
                crop_name = self.label_encoder.inverse_transform([idx])[0]
                confidence = float(probabilities[idx])
                
                results.append({
                    'crop': crop_name,
                    'confidence': confidence,
                    'rank': rank
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            # Fallback to rule-based prediction
            return self._fallback_predict(features, top_k)
    
    def _fallback_predict(self, features: Dict[str, float], top_k: int) -> List[Dict[str, any]]:
        """Fallback rule-based prediction"""
        ph = features.get('ph', 6.5)
        temperature = features.get('temperature', 25)
        rainfall = features.get('rainfall', 100)
        moisture = features.get('moisture', 60)
        
        results = []
        if temperature > 30 and rainfall > 150:
            results.append({'crop': 'rice', 'confidence': 0.8, 'rank': 1})
        elif ph > 7 and features.get('nitrogen', 50) > 60:
            results.append({'crop': 'maize', 'confidence': 0.75, 'rank': 1})
        elif moisture < 40 and temperature < 25:
            results.append({'crop': 'wheat', 'confidence': 0.7, 'rank': 1})
        else:
            results.append({'crop': 'rice', 'confidence': 0.6, 'rank': 1})
        
        # Add more crops
        other_crops = [c for c in self.crop_labels if c != results[0]['crop']]
        for i, crop in enumerate(other_crops[:top_k-1], 2):
            results.append({'crop': crop, 'confidence': 0.5 - i*0.05, 'rank': i})
        
        return results[:top_k]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        if self.model is None:
            return {}
        
        try:
            importance = self.model.feature_importance(importance_type='gain')
            importance_dict = dict(zip(self.feature_names, importance.tolist()))
            # Normalize
            total = sum(importance_dict.values())
            if total > 0:
                importance_dict = {k: v/total for k, v in importance_dict.items()}
            return importance_dict
        except Exception as e:
            logger.error(f"Feature importance error: {e}")
            return {}
    
    def get_model(self):
        """Get the underlying model for SHAP/LIME"""
        return self.model
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names.copy()
