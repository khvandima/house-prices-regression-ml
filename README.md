# 🏠 House Prices — End-to-End Regression (Kaggle)

An end-to-end machine learning project on the classic **Kaggle House Prices** dataset.

This repository demonstrates a complete ML workflow, including:

- Exploratory Data Analysis (EDA)
- Feature engineering
- Classical ML models (Linear Regression, Tree-based models, XGBoost/LightGBM)
- Simple ANN (MLP) for tabular data
- Saving trained models for deployment
- Clean, modular project structure

This project is part of my **ML/MLOps portfolio**.

---

# 🚀 Project Goals

- Build a clear, reproducible machine learning pipeline  
- Perform deep EDA and understand relationships in the data  
- Compare multiple regression models  
- Train a simple neural network (ANN)  
- Prepare artifacts for future deployment (FastAPI + Docker)  

---

# 📊 Workflow Overview

### 1️⃣ Exploratory Data Analysis (EDA)
- Missing values  
- Numerical & categorical feature exploration  
- Outlier analysis  
- Correlations and feature importance  
- Target distribution insights  

### 2️⃣ Feature Engineering
- Handling missing values  
- One-hot encoding of categorical variables  
- Scaling numerical variables  
- Data splitting  
- Feature selection  

### 3️⃣ Modeling
- Linear Regression (baseline)  
- RandomForest / GradientBoosting  
- XGBoost / LightGBM (optional)  
- Simple ANN (Multi-Layer Perceptron)  
- Cross-validation + error metrics (RMSE, MAE, R²)  

### 4️⃣ Saving the Model
- Export final model with `joblib`  
- Save feature transformers  
- Prepare `/models/` folder for deployment  

---

# 🛠 Tech Stack

- **Python 3.10+**
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- feature-engine  
- tensorflow / keras (for ANN)  
- xgboost
- lightgbmё

---

# 📁 Project Structure (planned)

```text
house-prices-regression-ml/
 ├─ notebooks/
 │   └─ house_prices_regression.ipynb
 ├─ requirements.txt
 └─ README.md
```

---

▶️ How to Run (planned)
```
# 1. Clone the repository
git clone https://github.com/khvandima/house-prices-regression-ml.git
cd house-prices-regression-ml

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open Jupyter notebook
jupyter notebook notebooks/house_prices_regression.ipynb
```

---

📌 Status

🚧 Work in progress
Notebooks, scripts and models will be added step by step.

