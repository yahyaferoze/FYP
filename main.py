import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time

# Load the dataset
df = pd.read_csv('~/Desktop/Friday-WorkingHours-Morning.csv', low_memory=False)

# Adjust the target column name to match exactly
target_column = ' Label'

# Split DataFrame into features (X) and the target variable (y)
X = df.drop(target_column, axis=1)
y = df[target_column]












