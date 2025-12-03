"""
Test script for ML pipeline (LightGBM + Quantile Regression).

This validates Step A of the ML roadmap before proceeding to Rust extensions.
"""

import sys
sys.path.insert(0, '/Users/i767700/Github/quant')

import numpy as np
import pandas as pd
from datetime import datetime

# Import only ML modules (avoid data modules that need pykrx)
from src.ml.lgbm_model import train_quantile_models, predict_quantiles, evaluate_predictions
from src.ml.position_sizing import confidence_weighted_portfolio


def create_synthetic_data():
    """Create synthetic data for testing when real data is slow."""
    np.random.seed(42)

    # 50 stocks, 60 months
    n_stocks = 50
    n_months = 60

    dates = pd.date_range('2019-01-01', periods=n_months, freq='ME')
    tickers = [f'STOCK_{i:03d}' for i in range(n_stocks)]

    # Create multi-index (date, ticker)
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])

    # Features: momentum, value, quality (random but with some structure)
    features = pd.DataFrame({
        'momentum': np.random.randn(len(index)),
        'value': np.random.randn(len(index)),
        'quality': np.random.randn(len(index)),
    }, index=index)

    # Add some signal: high momentum/value/quality → higher returns
    signal = 0.02 * features['momentum'] + 0.015 * features['value'] + 0.01 * features['quality']
    noise = np.random.randn(len(index)) * 0.05
    returns = signal + noise

    return features, pd.Series(returns, index=index, name='forward_return')


def test_quantile_model():
    """Test LightGBM quantile regression."""
    print("=" * 60)
    print("Testing LightGBM Quantile Regression")
    print("=" * 60)

    # Create synthetic data
    X, y = create_synthetic_data()

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Date range: {X.index.get_level_values('date').min()} to {X.index.get_level_values('date').max()}")

    # Split train/test (time-series aware)
    dates = X.index.get_level_values('date').unique()
    train_dates = dates[:48]  # First 48 months
    test_dates = dates[48:]   # Last 12 months

    train_mask = X.index.get_level_values('date').isin(train_dates)
    test_mask = X.index.get_level_values('date').isin(test_dates)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\nTrain: {len(X_train)} samples ({len(train_dates)} months)")
    print(f"Test:  {len(X_test)} samples ({len(test_dates)} months)")

    # Train quantile models
    print("\nTraining quantile models...")
    model = train_quantile_models(
        X_train, y_train,
        quantiles=[0.1, 0.5, 0.9],
        params={'n_estimators': 100, 'verbose': -1}
    )

    # Feature importance
    print("\n📊 Feature Importance:")
    print(model.feature_importance[['mean_importance']].head(10))

    # Predict on test set
    print("\nPredicting on test set...")
    predictions = predict_quantiles(model, X_test)

    print("\nPrediction sample:")
    print(predictions.head(10))

    # Evaluate
    print("\n📈 Evaluation Metrics:")
    metrics = evaluate_predictions(predictions, y_test)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return model, predictions, y_test


def test_position_sizing():
    """Test confidence-weighted portfolio construction."""
    print("\n" + "=" * 60)
    print("Testing Confidence-Weighted Position Sizing")
    print("=" * 60)

    # Create synthetic predictions
    np.random.seed(42)
    n_stocks = 100

    index = pd.MultiIndex.from_product(
        [[pd.Timestamp('2024-01-31')], [f'STOCK_{i:03d}' for i in range(n_stocks)]],
        names=['date', 'ticker']
    )

    # Create predictions with varying confidence
    predictions = pd.DataFrame({
        'q10': np.random.randn(n_stocks) * 0.03 - 0.02,
        'q50': np.random.randn(n_stocks) * 0.02,
        'q90': np.random.randn(n_stocks) * 0.03 + 0.02,
    }, index=index)

    predictions['interval_width'] = predictions['q90'] - predictions['q10']
    predictions['confidence'] = 1.0 / predictions['interval_width'].clip(lower=0.01)
    predictions['confidence_normalized'] = (
        (predictions['confidence'] - predictions['confidence'].min()) /
        (predictions['confidence'].max() - predictions['confidence'].min())
    )

    print(f"Predictions shape: {predictions.shape}")
    print("\nPrediction distribution:")
    print(predictions[['q50', 'confidence_normalized']].describe())

    # Construct portfolio
    long_w, short_w = confidence_weighted_portfolio(
        predictions,
        n_long=10,
        n_short=10
    )

    print(f"\n📊 Portfolio Construction:")
    print(f"Long positions: {len(long_w)}")
    print(f"Short positions: {len(short_w)}")
    print(f"Long weights sum: {long_w.sum():.4f}")
    print(f"Short weights sum: {short_w.sum():.4f}")

    print("\nTop 5 long positions:")
    print(long_w.sort_values(ascending=False).head())

    print("\nTop 5 short positions:")
    print(short_w.sort_values(ascending=False).head())

    # Check weight distribution
    print(f"\nWeight statistics (long):")
    print(f"  Min: {long_w.min():.4f}")
    print(f"  Max: {long_w.max():.4f}")
    print(f"  Std: {long_w.std():.4f}")

    return long_w, short_w


def main():
    """Run all tests."""
    print("\n🚀 ML Pipeline Test Suite")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test 1: Quantile Model
    try:
        model, preds, y_test = test_quantile_model()
        print("\n✅ Quantile model test PASSED")
    except Exception as e:
        print(f"\n❌ Quantile model test FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Position Sizing
    try:
        long_w, short_w = test_position_sizing()
        print("\n✅ Position sizing test PASSED")
    except Exception as e:
        print(f"\n❌ Position sizing test FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
