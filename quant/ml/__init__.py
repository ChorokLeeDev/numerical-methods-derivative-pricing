"""
Machine Learning module for factor return prediction.

Week 3: A → B → C Implementation Roadmap

A. LightGBM + Quantile Regression (Current)
   - Factor return prediction
   - Prediction interval for uncertainty
   - Confidence-based position sizing

B. LSTM + MC Dropout (Next)
   - Temporal pattern learning
   - Bayesian uncertainty via dropout

C. Temporal Fusion Transformer (Advanced)
   - Attention mechanism for interpretability
   - Built-in uncertainty quantification
"""

from .features import prepare_features, create_lagged_features
from .lgbm_model import QuantileGBM, train_quantile_models
from .position_sizing import confidence_weighted_portfolio
