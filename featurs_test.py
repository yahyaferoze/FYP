import unittest
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Load the dataset
        self.df = pd.read_csv('~/Desktop/Friday-WorkingHours-Morning.csv', low_memory=False)
        self.target_column = ' Label'
        self.X = self.df.drop(self.target_column, axis=1)
        self.y = self.df[self.target_column]

        # Preprocess the data
        # Convert non-numeric columns to numeric
        for col in self.X.columns:
            if self.X[col].dtype == 'object':
                self.X[col] = LabelEncoder().fit_transform(self.X[col])

        # Replaces inf/-inf with NaN in features
        self.X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # adds missing values with median
        imputer = SimpleImputer(strategy='median')
        self.X_imputed = imputer.fit_transform(self.X)

    def test_data_loading(self):
        # Checks if the DataFrame is not empty
        self.assertIsNotNone(self.df)
        # Checks if the DataFrame has the correct shape
        self.assertEqual(self.df.shape, (len(self.df), len(self.df.columns)))

    def test_feature_importance(self):
        # Ensures that the  RandomForestClassifier can be initialised and trained
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(self.X_imputed, self.y)
        # Ensure that the feature importances are sorted
        importances = rf_classifier.feature_importances_
        self.assertEqual(len(importances), len(self.X.columns))
        # Ensure the top features are printed correctly
        n_top_features = 10  # Set the number of top features to print
        top_features = [self.X.columns[i] for i in np.argsort(importances)[::-1][:n_top_features]]
        self.assertEqual(len(top_features), n_top_features)

if __name__ == '__main__':
    unittest.main()
