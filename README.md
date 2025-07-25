# Car Price Predictor – Regression with Random Forest

This project predicts the **selling price of used cars** based on various features such as mileage, engine size, max power, and more. It uses a full machine learning pipeline with preprocessing, feature engineering, and model training using a Random Forest Regressor.

---

## Project Summary

### ✅ What I Did:

1. **Data Loading and Cleaning**
   - Loaded dataset: `Car details v3.csv`.
   - Dropped rows with missing target values (`selling_price`).
   - Removed the `name` column as it was not used for prediction.

2. **Feature Engineering**
   - Created a custom transformer to clean and convert string columns:
     - Removed units from `mileage`, `engine`, and `max_power`.
     - Converted them to numeric format using `pd.to_numeric`.

3. **Preprocessing Pipeline**
   - **Numerical columns**: Imputed missing values using the mean.
   - **Categorical columns**: One-hot encoded (dropping the first to avoid multicollinearity).
   - Combined all steps using `ColumnTransformer` inside a `Pipeline`.

4. **Model**
   - Used `RandomForestRegressor` with 100 trees and a fixed random seed for reproducibility.
   - Wrapped model and preprocessing into a single `Pipeline`.

5. **Training and Evaluation**
   - Split data into 80% training and 20% testing sets.
   - Trained the pipeline and evaluated it using:
     - **Mean Absolute Error (MAE)**
     - **Root Mean Squared Error (RMSE)**
   - Reported performance:
     - MAE: ~value (printed in console)
     - RMSE: ~value (printed in console)

6. **Model Saving**
   - Saved the entire pipeline (including preprocessing and model) as `car_price_pipeline.pkl` using `joblib`.

---

## Future Improvements

- Add additional features like car brand/model extracted from the `name` column.
- Use cross-validation to tune hyperparameters for the Random Forest model.
- Try different regression algorithms (e.g., XGBoost, Gradient Boosting).
- Deploy the model as a web app using Streamlit or Flask for real-time price predictions.
