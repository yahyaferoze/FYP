import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# Load the dataset
df = pd.read_csv('~/Desktop/Friday-WorkingHours-Morning.csv', low_memory=False)

# Correct the column names by stripping spaces
df.columns = df.columns.str.strip()


target_column = 'Label'

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(df[target_column])

# Specify the features to use based on importance scores or your prior feature selection

features = [
    'Destination Port', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'Bwd Packets/s', 'Bwd IAT Min', 'Flow IAT Mean', 'Fwd Packets/s',
    'Bwd Packet Length Mean', 'Bwd IAT Max', 'Avg Bwd Segment Size',
    'Flow Duration', 'Packet Length Mean', 'Subflow Bwd Bytes',
    'Fwd IAT Max', 'Flow Packets/s', 'Bwd IAT Std',
    'Flow IAT Max', 'Bwd IAT Total'
]

# Preprocess features: replace inf/-inf with NaN and impute missing values
X = df[features].replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Initialise models
knn = KNeighborsClassifier()
ada_boost = AdaBoostClassifier(algorithm='SAMME')
decision_tree = DecisionTreeClassifier()

# Define the models
models = {
    'KNN': KNeighborsClassifier(),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
    'DecisionTree': DecisionTreeClassifier()
}

# Perform time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
results = {}

for name, model in models.items():
    model_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'time_taken': []}
    for train_index, test_index in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        start_time = time.time()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        end_time = time.time()

        # Gather performance metrics
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0, output_dict=True)

        model_results['accuracy'].append(accuracy)
        model_results['precision'].append(report['1']['precision'])
        model_results['recall'].append(report['1']['recall'])
        model_results['f1-score'].append(report['1']['f1-score'])
        model_results['time_taken'].append(end_time - start_time)

    results[name] = model_results
    print(f"{name} Model Fold Results Across {tscv.n_splits} Folds:")
    print(f"Average Accuracy: {np.mean(model_results['accuracy']):.4f}")
    print(f"Average Precision: {np.mean(model_results['precision']):.4f}")
    print(f"Average Recall: {np.mean(model_results['recall']):.4f}")
    print(f"Average F1-Score: {np.mean(model_results['f1-score']):.4f}")
    print(f"Average Time Taken: {np.mean(model_results['time_taken']):.2f} seconds\n")
