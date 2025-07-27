import pandas as pd
from sklearn.model_selection import KFold
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def kfold_cross_validation(df, n_splits=5, random_state=42):
    try:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = []
        for fold, (train_index, test_index) in enumerate(kf.split(df)):
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]
            folds.append((train_df, test_df))
            logger.info(f"Fold {fold + 1} split successfully.")
            logger.info(f"Training set size: {len(train_df)}")
            logger.info(f"Testing set size: {len(test_df)}")
        logger.info("K-fold cross-validation completed successfully.")
        return folds
    except Exception as e:
        logger.error(f"An error occurred during k-fold cross-validation: {e}")
        raise
