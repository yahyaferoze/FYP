import unittest
import pandas as pd
import numpy as np

class TestMain(unittest.TestCase):
    def setUp(self):
        # Loads a small sample dataset for testing
        self.df = pd.DataFrame({
            'Feature1': [1, 2, np.nan, 4, 5],
            'Feature2': ['A', 'B', 'C', np.nan, 'E'],
            'Label': [0, 1, 0, 1, 0]
        })

    def test_data_loading(self):
        self.assertIsNotNone(self.df)  # Check if DataFrame is not empty
        self.assertEqual(self.df.shape, (5, 3))  # Check if DataFrame has the correct shape

    def test_target_column_adjustment(self):
        target_column = ' Label'
        self.assertIn(target_column.strip(), self.df.columns)  # Check if target column exists in DataFrame

    def test_feature_target_split(self):
        X = self.df.drop('Label', axis=1)
        y = self.df['Label']
        self.assertEqual(X.shape[1], 2)  # Check if X has the correct number of features
        self.assertEqual(y.shape[0], 5)  # Check if y has the correct number of labels

if __name__ == '__main__':
    unittest.main()
