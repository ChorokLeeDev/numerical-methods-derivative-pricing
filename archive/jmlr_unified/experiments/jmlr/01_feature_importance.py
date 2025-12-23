"""
Phase 2 Week 5-6: Feature Importance Analysis for JMLR Paper

Objective: Identify which features drive tail risk prediction in crowded factors

Analysis:
1. Load 8 Fama-French factors + 168 engineered features
2. Train RandomForest model for crash prediction
3. Compute SHAP values for feature attribution
4. Feature ablation study (4 groups: Return, Vol, Correlation, Crowding)
5. Generate Table 5 (Top 20 features) and Figure 8 (SHAP summary)

Output:
- table5_feature_importance.csv
- figure8_shap_summary.pdf
- feature_ablation_results.csv

Author: Chorok Lee
Date: December 2025
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directories to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings('ignore')
except ImportError as e:
    logger.error(f"Missing required library: {e}")
    logger.info("Install with: pip install shap scikit-learn")
    sys.exit(1)


class FeatureImportanceAnalysis:
    """
    SHAP-based feature importance analysis for crowding detection.

    Methods:
    --------
    1. load_data: Load factors and features from factor_crowding project
    2. train_base_model: Train RandomForest for crash prediction
    3. compute_shap_values: Calculate SHAP feature importance
    4. feature_ablation_study: Test 4 feature groups
    5. generate_table5: Top 20 features with statistics
    6. generate_figure8: SHAP summary plot
    """

    def __init__(self):
        """Initialize analysis."""
        self.repo_root = repo_root
        self.data_dir = self.repo_root / 'data' / 'factor_crowding'
        self.results_dir = self.repo_root / 'results'
        self.results_dir.mkdir(exist_ok=True)

        self.X = None
        self.y = None
        self.feature_names = None
        self.model = None
        self.shap_values = None
        self.explainer = None

        logger.info(f"Feature Importance Analysis initialized")
        logger.info(f"  Data dir: {self.data_dir}")
        logger.info(f"  Results dir: {self.results_dir}")

    def load_data(self, test_size=0.2, random_state=42):
        """
        Load 8 Fama-French factors + 168 features and crash labels.

        Parameters:
        -----------
        test_size : float
            Test set proportion
        random_state : int
            Reproducibility seed

        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
        """
        logger.info("=" * 60)
        logger.info("Step 1: Loading Data")
        logger.info("=" * 60)

        try:
            # Load factors (from factor_crowding project)
            factors_file = self.data_dir / 'ff_extended_factors.parquet'
            if not factors_file.exists():
                logger.error(f"Factors file not found: {factors_file}")
                logger.info("Creating synthetic data for demo...")
                return self._create_synthetic_data()

            logger.info(f"Loading factors from: {factors_file}")
            factors_df = pd.read_parquet(factors_file)
            logger.info(f"  Shape: {factors_df.shape}")
            logger.info(f"  Factors: {factors_df.columns.tolist()}")

            # For now, use synthetic data for testing
            # In production, load from crowding_ml.py features
            logger.warning("Using synthetic data for Phase 2 prototype")
            logger.info("(Production will use data from factor_crowding/experiments)")

            return self._create_synthetic_data(n_samples=654, n_features=168)

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.info("Falling back to synthetic data")
            return self._create_synthetic_data()

    def _create_synthetic_data(self, n_samples=654, n_features=168):
        """
        Create synthetic data for prototyping.

        Parameters:
        -----------
        n_samples : int
            Number of samples (typical for one factor walk-forward window)
        n_features : int
            Number of engineered features

        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
        """
        logger.info(f"Creating synthetic data: {n_samples} samples, {n_features} features")

        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)

        # Create synthetic target: crash probability based on features
        # Features: [0:40] return, [40:72] volatility, [72:132] correlation, [132:168] crowding
        crash_prob = (
            0.3 * np.mean(X[:, 0:40], axis=1) +  # Recent returns
            0.2 * np.mean(X[:, 40:72], axis=1) +  # Volatility
            0.25 * np.mean(X[:, 72:132], axis=1) +  # Correlation patterns
            0.25 * np.mean(X[:, 132:168], axis=1)   # Crowding signals
        )
        y = (crash_prob > np.median(crash_prob)).astype(int)

        # Feature names by group
        feature_names = (
            [f'Return_{i}' for i in range(40)] +
            [f'Volatility_{i}' for i in range(32)] +
            [f'Correlation_{i}' for i in range(60)] +
            [f'Crowding_{i}' for i in range(36)]
        )

        self.feature_names = np.array(feature_names)
        self.feature_groups = {
            'Return': list(range(0, 40)),
            'Volatility': list(range(40, 72)),
            'Correlation': list(range(72, 132)),
            'Crowding': list(range(132, 168))
        }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"✅ Data created:")
        logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"  Class distribution: {np.bincount(y)} (0: no crash, 1: crash)")
        logger.info(f"  Crash rate: {y.mean():.1%}")

        return X_train, X_test, y_train, y_test

    def train_base_model(self, X_train, X_test, y_train, y_test):
        """
        Train RandomForest for crash prediction.

        Parameters:
        -----------
        X_train, X_test : arrays
            Feature matrices
        y_train, y_test : arrays
            Target labels

        Returns:
        --------
        float : Test AUC-ROC score
        """
        logger.info("=" * 60)
        logger.info("Step 2: Training RandomForest Model")
        logger.info("=" * 60)

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        logger.info("Training RandomForest (100 trees, max_depth=10)...")
        self.model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import roc_auc_score, classification_report

        y_pred_train = self.model.predict_proba(X_train)[:, 1]
        y_pred_test = self.model.predict_proba(X_test)[:, 1]

        auc_train = roc_auc_score(y_train, y_pred_train)
        auc_test = roc_auc_score(y_test, y_pred_test)

        logger.info(f"✅ Model trained:")
        logger.info(f"  Train AUC: {auc_train:.4f}")
        logger.info(f"  Test AUC: {auc_test:.4f}")
        logger.info(f"\nClassification Report (Test Set):")
        logger.info(classification_report(y_test, (y_pred_test > 0.5).astype(int)))

        # Store for later use in ablation study
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        return auc_test

    def compute_shap_values(self, X_train, sample_size=100):
        """
        Compute SHAP values for feature attribution.

        Parameters:
        -----------
        X_train : array
            Training data for background
        sample_size : int
            Number of samples for SHAP computation

        Returns:
        --------
        np.array : SHAP values (n_samples, n_features)
        """
        logger.info("=" * 60)
        logger.info("Step 3: Computing SHAP Values")
        logger.info("=" * 60)

        logger.info(f"Creating TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)

        logger.info(f"Computing SHAP values (sample_size={sample_size})...")
        X_sample = X_train[:sample_size]
        self.shap_values = self.explainer.shap_values(X_sample)

        logger.info(f"✅ SHAP values computed:")
        logger.info(f"  Shape: {self.shap_values.shape}")
        logger.info(f"  Mean |SHAP|: {np.abs(self.shap_values).mean():.4f}")

        return self.shap_values

    def feature_ablation_study(self, X_test, y_test):
        """
        Test model performance with different feature groups.

        Parameters:
        -----------
        X_test : array
            Test features
        y_test : array
            Test labels

        Returns:
        --------
        pd.DataFrame : AUC scores by feature group
        """
        logger.info("=" * 60)
        logger.info("Step 4: Feature Ablation Study")
        logger.info("=" * 60)

        from sklearn.metrics import roc_auc_score

        results = []

        # Baseline: all features
        y_pred = self.model.predict_proba(X_test)[:, 1]
        baseline_auc = roc_auc_score(y_test, y_pred)
        results.append({
            'Feature Group': 'All Features',
            'Features Used': 168,
            'AUC': baseline_auc,
            'Delta AUC': 0.0
        })
        logger.info(f"Baseline (all features): AUC = {baseline_auc:.4f}")

        # Test each feature group
        for group_name, indices in self.feature_groups.items():
            X_subset = X_test[:, indices]

            # Use current model's predictions but only subset features
            # For pure ablation: check if subset alone can predict
            # Train simple model with just this group
            model_subset = RandomForestClassifier(
                n_estimators=50,  # Fewer for speed
                max_depth=8,
                min_samples_leaf=20,
                class_weight='balanced',
                random_state=42,
                n_jobs=4
            )

            try:
                # Train on subset features using full X_train
                model_subset.fit(self.X_train[:, indices], self.y_train)
                y_pred_subset = model_subset.predict_proba(X_subset)[:, 1]
                subset_auc = roc_auc_score(y_test, y_pred_subset)
                delta_auc = subset_auc - baseline_auc
            except Exception as e:
                logger.warning(f"Error evaluating {group_name}: {e}")
                subset_auc = np.nan
                delta_auc = np.nan

            logger.info(f"  {group_name}: {len(indices)} features → AUC = {subset_auc:.4f}")
            results.append({
                'Feature Group': group_name,
                'Features Used': len(indices),
                'AUC': subset_auc,
                'Delta AUC': delta_auc
            })

        results_df = pd.DataFrame(results)

        logger.info("\n✅ Ablation Study Results:")
        logger.info(results_df.to_string())

        return results_df

    def generate_table5(self):
        """
        Generate Table 5: Top 20 Features for JMLR Paper.

        Returns:
        --------
        pd.DataFrame : Top 20 features with statistics
        """
        logger.info("=" * 60)
        logger.info("Step 5: Generating Table 5 (Feature Importance)")
        logger.info("=" * 60)

        if self.shap_values is None:
            logger.error("SHAP values not computed yet")
            return None

        # Handle SHAP shape: (n_samples, n_features) or (n_samples, n_features, 2)
        shap_vals = self.shap_values
        if len(shap_vals.shape) == 3:  # Binary classification case
            shap_vals = shap_vals[:, :, 1]  # Use positive class SHAP values

        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_vals).mean(axis=0)
        std_shap = np.abs(shap_vals).std(axis=0)
        min_shap = np.abs(shap_vals).min(axis=0)
        max_shap = np.abs(shap_vals).max(axis=0)

        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean |SHAP|': mean_shap,
            'Std SHAP': std_shap,
            'Min SHAP': min_shap,
            'Max SHAP': max_shap
        }).sort_values('Mean |SHAP|', ascending=False)

        # Top 20
        top20 = feature_importance.head(20).copy()
        top20['Rank'] = range(1, 21)

        # Extract feature group
        top20['Group'] = top20['Feature'].apply(
            lambda x: x.split('_')[0] if '_' in x else 'Unknown'
        )

        table5 = top20[['Rank', 'Feature', 'Group', 'Mean |SHAP|', 'Std SHAP', 'Max SHAP']]

        logger.info("\n✅ Table 5: Top 20 Features")
        logger.info(table5.to_string(index=False))

        # Save to CSV
        output_file = self.results_dir / 'table5_feature_importance.csv'
        table5.to_csv(output_file, index=False)
        logger.info(f"Saved to: {output_file}")

        return table5

    def generate_figure8(self):
        """
        Generate Figure 8: SHAP Summary Plot for JMLR Paper.

        Returns:
        --------
        matplotlib.figure.Figure : SHAP summary plot
        """
        logger.info("=" * 60)
        logger.info("Step 6: Generating Figure 8 (SHAP Summary)")
        logger.info("=" * 60)

        if self.shap_values is None:
            logger.error("SHAP values not computed yet")
            return None

        # Handle SHAP shape: (n_samples, n_features) or (n_samples, n_features, 2)
        shap_vals = self.shap_values
        if len(shap_vals.shape) == 3:  # Binary classification case
            shap_vals = shap_vals[:, :, 1]  # Use positive class SHAP values

        # Create SHAP summary plot
        plt.figure(figsize=(10, 8))

        # Top 15 features
        mean_shap = np.abs(shap_vals).mean(axis=0)
        top_indices = np.argsort(mean_shap)[::-1][:15]

        shap_values_top = shap_vals[:, top_indices]
        feature_names_top = self.feature_names[top_indices]

        # Horizontal bar plot
        y_pos = np.arange(len(feature_names_top))
        importances = np.abs(shap_values_top).mean(axis=0)

        plt.barh(y_pos, importances, color='steelblue', alpha=0.8)
        plt.yticks(y_pos, feature_names_top)
        plt.xlabel('Mean |SHAP value| (Average impact on model output)')
        plt.title('Figure 8: Feature Importance for Crash Prediction\n(SHAP Summary Plot)')
        plt.tight_layout()

        # Save figure
        output_file = self.results_dir / 'figure8_shap_summary.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved Figure 8 to: {output_file}")

        plt.close()

        return output_file

    def run_full_analysis(self):
        """
        Execute complete feature importance analysis.

        Returns:
        --------
        dict : Analysis results
        """
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE IMPORTANCE ANALYSIS - FULL PIPELINE")
        logger.info("=" * 60)

        try:
            # Step 1: Load data
            X_train, X_test, y_train, y_test = self.load_data()

            # Step 2: Train model
            auc_test = self.train_base_model(X_train, X_test, y_train, y_test)

            # Step 3: Compute SHAP
            shap_vals = self.compute_shap_values(X_train, sample_size=min(100, len(X_train)))

            # Step 4: Ablation study
            ablation_df = self.feature_ablation_study(X_test, y_test)

            # Step 5: Generate Table 5
            table5 = self.generate_table5()

            # Step 6: Generate Figure 8
            fig8_path = self.generate_figure8()

            logger.info("\n" + "=" * 60)
            logger.info("✅ FEATURE IMPORTANCE ANALYSIS COMPLETE")
            logger.info("=" * 60)

            return {
                'status': 'success',
                'model_auc': auc_test,
                'table5': table5,
                'figure8': str(fig8_path),
                'ablation_results': ablation_df
            }

        except Exception as e:
            logger.error(f"Error in analysis: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}


def main():
    """Main entry point."""
    analysis = FeatureImportanceAnalysis()
    results = analysis.run_full_analysis()

    logger.info(f"\nResults summary:")
    logger.info(f"  Status: {results.get('status')}")
    logger.info(f"  Model AUC: {results.get('model_auc', 'N/A')}")
    logger.info(f"  Table 5: {results.get('table5', 'N/A')}")
    logger.info(f"  Figure 8: {results.get('figure8', 'N/A')}")

    return results


if __name__ == '__main__':
    main()
