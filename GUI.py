import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from datetime import datetime


class DataAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis GUI")
        self.df = None
        self.X_scaled = None
        self.y = None

        # Create GUI components
        self.load_button = tk.Button(root, text="Load Data", command=self.load_data)
        self.load_button.pack()

        self.preprocess_button = tk.Button(root, text="Preprocess Data", command=self.preprocess_data)
        self.preprocess_button.pack()

        # Button for K-Nearest Neighbors model
        self.run_knn_button = tk.Button(root, text="Run KNN Model", command=lambda: self.run_model('KNN'))
        self.run_knn_button.pack()

        # Button for AdaBoost model
        self.run_adaboost_button = tk.Button(root, text="Run AdaBoost Model", command=lambda: self.run_model('AdaBoost'))
        self.run_adaboost_button.pack()

        # Button for Decision Tree model
        self.run_dt_button = tk.Button(root, text="Run Decision Tree Model", command=lambda: self.run_model('DecisionTree'))
        self.run_dt_button.pack()

        # Text box for logging
        self.log_text = scrolledtext.ScrolledText(root, height=10)
        self.log_text.pack()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                messagebox.showinfo("Information", "Data loaded successfully")
                self.log("Data loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
        else:
            self.log("Data loading canceled by the user.")

    def preprocess_data(self):
        if self.df is not None:
            try:
                X = self.df.iloc[:, :-1]
                self.y = self.df.iloc[:, -1]
                X.replace([np.inf, -np.inf], np.nan, inplace=True)
                imputer = SimpleImputer(strategy='median')
                X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                scaler = StandardScaler()
                self.X_scaled = scaler.fit_transform(X_imputed)
                messagebox.showinfo("Information", "Data preprocessed successfully")
                self.log("Data preprocessed successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to preprocess data: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Load data before preprocessing.")

    def run_model(self, model_name):
        if self.X_scaled is not None and self.y is not None:
            models = {
                'KNN': KNeighborsClassifier(),
                'AdaBoost': AdaBoostClassifier(),
                'DecisionTree': DecisionTreeClassifier()
            }
            try:
                model = models[model_name]
                X_train, X_test, y_train, y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                report = classification_report(y_test, predictions)
                self.log(f"{model_name} Model Classification Report:\n{report}")
                messagebox.showinfo(f"{model_name} Model Results", f"{model_name} model executed successfully. Check the log for the classification report.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to run {model_name} model: {str(e)}")
        else:
            messagebox.showwarning("Warning", f"Load and preprocess data before running the {model_name} model.")

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp}: {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        with open("model_run_log.txt", "a") as log_file:
            log_file.write(log_entry)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()
