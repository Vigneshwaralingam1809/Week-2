# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from google.colab import files
import joblib

# Upload the dataset
uploaded = files.upload()

# Load the dataset
def load_data(file_path):
    """Loads the dataset from the given file path."""
    data = pd.read_csv(file_path)
    return data

# Preprocess data and split into features and target
def preprocess_data(data):
    """Handles missing values, feature scaling, and encoding."""
    data = data.fillna(data.mean())  # Fill missing values with mean
    
    X = data.drop(columns=['generated_power_kw'])  # Features
    y = data['generated_power_kw']  # Target variable
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler

# Train multiple models
def train_models(X_train, y_train):
    """Trains Linear Regression, Random Forest, and Gradient Boosting."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

# Evaluate the models
def evaluate_models(models, X_test, y_test):
    """Evaluates models and returns performance metrics."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "MSE": mse, "MAE": mae, "RMSE": rmse, "R2 Score": r2
        }
        
        # Visualization
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_pred, alpha=0.6, label="Predicted vs Actual")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        plt.title(f"{name} - Actual vs Predicted")
        plt.xlabel("Actual Power Output (kW)")
        plt.ylabel("Predicted Power Output (kW)")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return results

# Save the best model
def save_best_model(models, results):
    """Saves the best model based on highest R2 score."""
    best_model_name = max(results, key=lambda x: results[x]["R2 Score"])
    best_model = models[best_model_name]
    
    joblib.dump(best_model, "best_model.pkl")
    print(f"Best model '{best_model_name}' saved successfully!")
    return best_model_name

# Main function
if __name__ == "__main__":
    # File path to the uploaded dataset
    file_path = "03f4d1c1a55947025601 (1).csv"  # Automatically get uploaded file name
    
    # Step 1: Load the data
    data = load_data(file_path)
    
    # Step 2: Preprocess the data
    (X_train, X_test, y_train, y_test), scaler = preprocess_data(data)
    
    # Step 3: Train the models
    models = train_models(X_train, y_train)
    
    # Step 4: Evaluate the models
    results = evaluate_models(models, X_test, y_test)
    
    # Print results
    for model_name, metrics in results.items():
        print(f"\n{model_name} Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Step 5: Save the best model
    best_model_name = save_best_model(models, results)
