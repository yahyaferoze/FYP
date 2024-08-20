import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score  # Importing accuracy_score
import time


class TestModelFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load and preprocess data just as in the main script
        cls.df = pd.read_csv('~/Desktop/Friday-WorkingHours-Morning.csv', low_memory=False)
        cls.df.columns = cls.df.columns.str.strip()

        cls.target_column = 'Label'
        cls.features = [
            'Destination Port', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
            'Bwd Packets/s', 'Bwd IAT Min', 'Flow IAT Mean', 'Fwd Packets/s',
            'Bwd Packet Length Mean', 'Bwd IAT Max', 'Avg Bwd Segment Size',
            'Flow Duration', 'Packet Length Mean', 'Subflow Bwd Bytes',
            'Fwd IAT Max', 'Flow Packets/s', 'Bwd IAT Std',
            'Flow IAT Max', 'Bwd IAT Total'
        ]

        # Encode target
        le = LabelEncoder()
        cls.y_encoded = le.fit_transform(cls.df[cls.target_column])

        # Prepare features
        X = cls.df[cls.features].replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)

        # Scale features
        scaler = StandardScaler()
        cls.X_scaled = scaler.fit_transform(X_imputed)

        # Initialise models
        cls.models = {
            'KNN': KNeighborsClassifier(),
            'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
            'DecisionTree': DecisionTreeClassifier()
        }

    def test_data_loading(self):
        """Test data loading and initial preprocessing."""
        self.assertFalse(self.df.empty)
        self.assertIn(self.target_column, self.df.columns)

    def test_feature_selection(self):
        """Ensure that all specified features are in the DataFrame."""
        for feature in self.features:
            self.assertIn(feature, self.df.columns)

    def test_preprocessing(self):
        """Test imputing and scaling."""
        self.assertEqual(self.X_scaled.shape[1], len(self.features))

    def test_model_initialization(self):
        """Verify that each model is initialized correctly."""
        self.assertIsInstance(self.models['KNN'], KNeighborsClassifier)
        self.assertIsInstance(self.models['AdaBoost'], AdaBoostClassifier)
        self.assertIsInstance(self.models['DecisionTree'], DecisionTreeClassifier)

    def test_time_series_cross_validation(self):
        """Check the time series cross-validation process."""
        tscv = TimeSeriesSplit(n_splits=5)
        model = self.models['KNN']
        for train_index, test_index in tscv.split(self.X_scaled):
            X_train, X_test = self.X_scaled[train_index], self.X_scaled[test_index]
            y_train, y_test = self.y_encoded[train_index], self.y_encoded[test_index]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            self.assertGreaterEqual(accuracy, 0)  # Basic check to make sure that it runs and returns a result


if __name__ == '__main__':
    unittest.main()
