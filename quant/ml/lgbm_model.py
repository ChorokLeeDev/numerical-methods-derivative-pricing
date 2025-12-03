"""
LightGBM Quantile Regression for factor return prediction.

Key idea: Instead of predicting a single return value, predict the distribution:
- q10: 10th percentile (pessimistic scenario)
- q50: 50th percentile (median prediction)
- q90: 90th percentile (optimistic scenario)

Confidence = 1 / (q90 - q10)  # Narrower interval = higher confidence
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import warnings

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Run: pip install lightgbm")


@dataclass
class QuantileGBM:
    """
    LightGBM Quantile Regression model.

    Trains separate models for different quantiles to estimate
    the prediction interval.

    Attributes:
        models: Dict of {quantile: fitted_model}
        feature_names: List of feature names used
        feature_importance: DataFrame with feature importances
    """
    models: dict = None
    feature_names: list = None
    feature_importance: pd.DataFrame = None
    quantiles: list = None

    def __post_init__(self):
        self.models = self.models or {}
        self.feature_names = self.feature_names or []
        self.quantiles = self.quantiles or [0.1, 0.5, 0.9]


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    quantiles: list[float] = [0.1, 0.5, 0.9],
    params: dict = None
) -> QuantileGBM:
    """
    Train LightGBM models for multiple quantiles.

    Parameters:
        X_train: Training features
        y_train: Training target (returns)
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        quantiles: Quantiles to estimate
        params: LightGBM parameters

    Returns:
        QuantileGBM with trained models

    Example:
        >>> model = train_quantile_models(X_train, y_train)
        >>> predictions = predict_quantiles(model, X_test)
    """
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    # Default parameters (tuned for financial data)
    default_params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 5,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    if params:
        default_params.update(params)

    # Handle NaN/inf values
    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
    y_train_clean = y_train.replace([np.inf, -np.inf], np.nan)

    # Keep only complete cases
    mask = X_train_clean.notna().all(axis=1) & y_train_clean.notna()
    X_train_clean = X_train_clean[mask]
    y_train_clean = y_train_clean[mask]

    # Similarly for validation
    if X_val is not None and y_val is not None:
        X_val_clean = X_val.replace([np.inf, -np.inf], np.nan)
        y_val_clean = y_val.replace([np.inf, -np.inf], np.nan)
        val_mask = X_val_clean.notna().all(axis=1) & y_val_clean.notna()
        X_val_clean = X_val_clean[val_mask]
        y_val_clean = y_val_clean[val_mask]
        eval_set = [(X_val_clean, y_val_clean)]
    else:
        eval_set = None

    models = {}
    importances = []

    for q in quantiles:
        print(f"Training quantile {q:.2f} model...")

        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=q,
            **default_params
        )

        # Fit with optional early stopping
        if eval_set:
            model.fit(
                X_train_clean, y_train_clean,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )
        else:
            model.fit(X_train_clean, y_train_clean)

        models[q] = model

        # Collect feature importance
        imp = pd.DataFrame({
            'feature': X_train_clean.columns,
            f'importance_q{int(q*100)}': model.feature_importances_
        })
        importances.append(imp.set_index('feature'))

    # Combine importances
    feature_importance = pd.concat(importances, axis=1)
    feature_importance['mean_importance'] = feature_importance.mean(axis=1)
    feature_importance = feature_importance.sort_values('mean_importance', ascending=False)

    return QuantileGBM(
        models=models,
        feature_names=list(X_train_clean.columns),
        feature_importance=feature_importance,
        quantiles=quantiles
    )


def predict_quantiles(
    model: QuantileGBM,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict return quantiles for given features.

    Parameters:
        model: Trained QuantileGBM
        X: Features to predict

    Returns:
        DataFrame with columns: q10, q50, q90, confidence

    Example:
        >>> preds = predict_quantiles(model, X_test)
        >>> high_conf = preds[preds['confidence'] > preds['confidence'].median()]
    """
    # Clean input
    X_clean = X.replace([np.inf, -np.inf], np.nan)

    # Predict each quantile
    predictions = {}
    for q, lgb_model in model.models.items():
        predictions[f'q{int(q*100)}'] = lgb_model.predict(X_clean)

    result = pd.DataFrame(predictions, index=X.index)

    # Calculate confidence (inverse of prediction interval width)
    q10_col = f'q{int(model.quantiles[0]*100)}'
    q90_col = f'q{int(model.quantiles[-1]*100)}'

    interval_width = result[q90_col] - result[q10_col]

    # Avoid division by zero and cap extreme confidence values
    interval_width = interval_width.clip(lower=0.001)
    result['interval_width'] = interval_width
    result['confidence'] = 1.0 / interval_width

    # Normalize confidence to [0, 1] range
    result['confidence_normalized'] = (
        (result['confidence'] - result['confidence'].min()) /
        (result['confidence'].max() - result['confidence'].min())
    ).fillna(0.5)

    return result


def rolling_train_predict(
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int = 36,  # 36 months
    step_size: int = 1,      # Retrain every month
    quantiles: list[float] = [0.1, 0.5, 0.9]
) -> pd.DataFrame:
    """
    Rolling window training and prediction.

    For each time period:
    1. Train on past `train_window` months
    2. Predict for current month

    Parameters:
        X: Features (multi-index: date, ticker)
        y: Target returns
        train_window: Number of months for training
        step_size: Months between retraining
        quantiles: Quantiles to predict

    Returns:
        DataFrame with predictions for each (date, ticker)
    """
    dates = X.index.get_level_values('date').unique().sort_values()
    all_predictions = []

    for i in range(train_window, len(dates), step_size):
        pred_date = dates[i]
        train_dates = dates[i-train_window:i]

        # Get training data
        train_mask = X.index.get_level_values('date').isin(train_dates)
        X_train = X[train_mask]
        y_train = y[train_mask]

        # Get prediction data (current date)
        pred_mask = X.index.get_level_values('date') == pred_date
        X_pred = X[pred_mask]

        if len(X_train) < 50 or len(X_pred) == 0:
            continue

        # Train and predict
        try:
            model = train_quantile_models(
                X_train, y_train,
                quantiles=quantiles,
                params={'n_estimators': 100, 'verbose': -1}
            )
            preds = predict_quantiles(model, X_pred)
            all_predictions.append(preds)
        except Exception as e:
            print(f"Error at {pred_date}: {e}")
            continue

    if not all_predictions:
        return pd.DataFrame()

    return pd.concat(all_predictions)


def evaluate_predictions(
    predictions: pd.DataFrame,
    actuals: pd.Series,
    quantiles: list[float] = [0.1, 0.5, 0.9]
) -> dict:
    """
    Evaluate quantile prediction quality.

    Metrics:
    - Coverage: % of actuals within predicted interval
    - Pinball loss: Standard quantile regression metric
    - IC (Information Coefficient): Correlation between prediction and actual

    Parameters:
        predictions: DataFrame from predict_quantiles
        actuals: Actual returns
        quantiles: Quantiles used

    Returns:
        Dict of evaluation metrics
    """
    # Align indices
    common = predictions.index.intersection(actuals.index)
    preds = predictions.loc[common]
    actual = actuals.loc[common]

    q10_col = f'q{int(quantiles[0]*100)}'
    q50_col = f'q{int(quantiles[1]*100)}'
    q90_col = f'q{int(quantiles[-1]*100)}'

    # Coverage: % within interval
    within_interval = (actual >= preds[q10_col]) & (actual <= preds[q90_col])
    coverage = within_interval.mean()

    # Pinball loss for median
    median_errors = actual - preds[q50_col]
    pinball_50 = np.mean(np.where(
        median_errors >= 0,
        0.5 * median_errors,
        (0.5 - 1) * median_errors
    ))

    # IC: rank correlation
    ic = preds[q50_col].corr(actual, method='spearman')

    # RMSE
    rmse = np.sqrt(((preds[q50_col] - actual) ** 2).mean())

    return {
        'coverage': coverage,
        'expected_coverage': quantiles[-1] - quantiles[0],  # e.g., 0.8 for 10-90
        'pinball_loss': pinball_50,
        'information_coefficient': ic,
        'rmse': rmse,
        'n_samples': len(common)
    }
