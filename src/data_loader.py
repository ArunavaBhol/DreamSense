import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGDataLoader:
    """Handles loading and preprocessing of EEG CSV datasets."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Loads data from the CSV file path."""
        try:
            logger.info(f"Loading EEG data from {self.file_path}...")
            self.data = pd.read_csv(self.file_path)
            logger.info(f"Successfully loaded dataset with shape: {self.data.shape}")
            return self.data
        except FileNotFoundError as e:
            logger.error(f"File not found at {self.file_path}")
            raise e
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise e

    def split_features_target(self, target_column: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """Splits the dataframe into features (X) and target labels (y)."""
        if self.data is None:
            self.load_data()
            
        if target_column not in self.data.columns:
            raise KeyError(f"Target column '{target_column}' not found in dataset.")
            
        X = self.data.drop(columns=[target_column]).values
        y = self.data[target_column].values
        return X, y
