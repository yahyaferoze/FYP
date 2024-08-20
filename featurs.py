import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Load the dataset
df = pd.read_csv('~/Desktop/Friday-WorkingHours-Morning.csv', low_memory=False)

# Split into features (X) and the target variable (y)
target_column = ' Label'
X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert non-numeric columns to numeric
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])

# Replace inf/-inf with NaN in features
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Convert back to DataFrame with appropriate column names
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed_df, y, test_size=0.2, random_state=42)


X_train_dropped = X_train.copy()  # Ensure it is a DataFrame

# Drop specified columns from the training set
columns_to_exclude = ['Flow ID', 'Source IP', 'Destination IP', 'Destination Port', 'Timestamp', 'External IP', 'Total Length of Bwd Packets', 'Total Length of Fwd Packets']
X_train_dropped.drop(columns=columns_to_exclude, errors='ignore', inplace=True)

# Re-impute any missing values with median
X_train_dropped_imputed = imputer.fit_transform(X_train_dropped)

# Converts back to a DataFrame after imputation
X_train_dropped_imputed_df = pd.DataFrame(X_train_dropped_imputed, columns=X_train_dropped.columns)

# Initialise and fit the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_dropped_imputed_df, y_train)

# Gett the feature importances and sorts them in order
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# Selects  and prints the top features
n_top_features = 30
print("Feature ranking:")
for f in range(n_top_features):
    print(f"{f + 1}. feature {X_train_dropped_imputed_df.columns[indices[f]]} ({importances[indices[f]]})")
