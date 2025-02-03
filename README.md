

# **Solar Power Output Prediction using Linear Regression**  

## **Overview**  
This project predicts **solar power generation (kW)** using historical data and **Linear Regression**. The model helps in forecasting energy output, aiding **efficient power management** and **optimization of solar resources**.  

## **Features**  
✅ **Data Preprocessing**: Cleans and prepares the dataset  
✅ **Train-Test Split**: Splits data into training and testing sets  
✅ **Linear Regression Model**: Trains a model for prediction  
✅ **Evaluation Metrics**: Computes **MSE (Mean Squared Error)** and **R² Score**  
✅ **Visualization**: Graphs **actual vs. predicted** power output  

## **Improvements from Week 1**  
🔹 **Added Dynamic File Upload Handling** (No need to rename files manually)  
🔹 **Data Cleaning**: Handled missing values and outliers  
🔹 **Feature Scaling**: Standardized numerical columns for better model accuracy  
🔹 **Enhanced Visualization**: Improved scatter plot with trend lines  

## **Installation & Usage**  

### **1. Clone Repository**  
```bash
git clone https://github.com/vigneshwaralingam1809/solar-power-prediction.git
cd solar-power-prediction
```

### **2. Install Dependencies**  
```bash
pip install pandas numpy scikit-learn matplotlib
```

### **3. Run the Script in Google Colab**  
Upload the dataset and execute the Python script:  
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import files
import io

# Upload and Load Data
uploaded = files.upload()
file_path = list(uploaded.keys())[0]  
data = pd.read_csv(io.BytesIO(uploaded[file_path]))

# Data Preprocessing
X = data.drop(columns=['generated_power_kw'])
y = data['generated_power_kw']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualization
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
plt.title('Actual vs Predicted Solar Power Output')
plt.xlabel('Actual Power Output (kW)')
plt.ylabel('Predicted Power Output (kW)')
plt.legend()
plt.grid(True)
plt.show()
```

## **Dataset**  
The dataset contains:  
- **Weather conditions** (temperature, humidity, sunlight hours)  
- **Solar panel characteristics**  
- **Power output readings (kW)**  

## **Contributors**  
- **[Your Name]** - Project Implementation  

## **License**  
This project is open-source and available under the **MIT License**.
