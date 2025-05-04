# ML-PROJECT-PREMIUM-PREDICTION
ML HEALTH INSURANCE PREMIUM PREDICTION PROJECT
URL : https://ml-insurance-premium-predictor.streamlit.app/
************************************************************************************************************************************************************
************************************************************************************************************************************************************
1. Data Preprocessing
   
Common Steps
- Load Data: Read Excel files and standardize column names.
- Handle Missing Values: Drop rows with missing data.
- Sanitize Values: Convert negative values (e.g., number_of_dependants) to absolute.
- Outlier Removal: Remove entries with unrealistic ages (>100) and extreme income values (above 99.9th percentile).
- Categorical Cleanup: Standardize categorical entries (e.g., unify smoking status labels).
- Feature Engineering:
  - Medical History: Split and score diseases to compute a normalized risk score.
  - Income & Plan Encoding: Map income levels and insurance plans to ordinal values.
  - One-hot Encoding: Convert nominal categorical variables into dummy/indicator variables.

Young Segment Special Notes

- Uses a Linear Regression model.
- Focuses on genetic risk features.

Rest Segment Special Notes

- Uses an XGBoost Regressor with hyperparameter tuning (RandomizedSearchCV).
- Sets genetical_risk to 0 for this group.

************************************************************************************************************************************************************
2. Feature Scaling

- MinMaxScaler is applied to key numerical columns:
  - age
  - number_of_dependants
  - income_level
  - income_lakhs
  - insurance_plan
  - genetical_risk

- The scaler and the list of scaled columns are saved for future inference.

************************************************************************************************************************************************************
3. Model Training

Young Segment

- Model: Linear Regression (sklearn.linear_model.LinearRegression)
- Train/Test Split: 70% train, 30% test
- Feature Selection: Drops income_level due to multicollinearity (checked using VIF).
- Model Export: Saved as model_young.joblib

Rest Segment

- Model: XGBoost Regressor (xgboost.XGBRegressor)
- Hyperparameter Tuning: RandomizedSearchCV on n_estimators, learning_rate, max_depth.
- Feature Selection: Drops income_level due to multicollinearity.
- Model Export: Saved as model_rest.joblib

************************************************************************************************************************************************************
4. Error Analysis

- Prediction: Generate predictions on the test set.
- Residual Calculation: Compute absolute and percentage errors.
- Extreme Error Analysis: Identify cases where prediction error exceeds 10%.
- Feature Distribution: (Optional, commented out) Visualize feature distributions for extreme error cases.

************************************************************************************************************************************************************
5. Model Export

- Trained models and scalers are exported using joblib for deployment or further use.

************************************************************************************************************************************************************
6. Requirements

All dependencies are listed in requirements.txt:

joblib==1.4.2
pandas==2.2.3
streamlit==1.45.0
numpy==2.2.0
scikit-learn==1.6.1
xgboost==3.0.0
************************************************************************************************************************************************************
7. Usage Instructions

  1. Install Dependencies

  pip install -r requirements.txt

  2. Prepare Data

  - Ensure premiums_young_with_gr.xlsx and premiums_rest.xlsx are in the working directory.

  3. Run Pipelines

  python 12try01-young_with_gr.py
  python 12try01-rest_with_gr.py

  4. Artifacts

- Trained models and scalers will be saved in the artifacts/ directory.

************************************************************************************************************************************************************
************************************************************************************************************************************************************
