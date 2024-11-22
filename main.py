import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the dataset."""
    df = df.dropna(subset=['SalePrice'])
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col].fillna('None', inplace=True)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def split_data(df, target='SalePrice'):
    """Split the dataset."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a model and calculate RMSE manually.
    """
    predictions = model.predict(X_test)
    
    # Calculate RMSE manually
    mse = np.mean((y_test - predictions) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate MAE
    mae = mean_absolute_error(y_test, predictions)
    
    save_plot_results(y_test, predictions, model_name)
    
    return rmse, mae, predictions

def save_plot_results(y_test, predictions, model_name="Model"):
    """Plot actual vs predicted."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"Actual vs Predicted: {model_name}")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    file_path = f'reports/{model_name}_actual_vs_predicted.png'
    if not os.path.exists('reports'):
        os.makedirs('reports')
    plt.savefig(file_path)
    plt.close()
    print(f"The scatter plot of {model_name} results has been successfully saved in the 'reports' folder")

# ========== Main Workflow ==========
if __name__ == "__main__":
    # Load and preprocess the dataset
    file_path = "data/house_prices.csv"
    data = load_data(file_path)
    data = preprocess_data(data.copy())
    X_train, X_test, y_train, y_test = split_data(data)
    
    # List of models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=10.0, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(max_depth=10),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, force_row_wise=True),
        "SVR": SVR(kernel='rbf', C=100)
    }

    # Train and evaluate traditional models
    results = {}
    model_instances = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        rmse, mae, predictions = evaluate_model(model, X_test, y_test, model_name=name)
        results[name] = rmse
        model_instances[name] = model
    
    # Neural Network
    print("\nTraining Neural Network...")
    input_dim = X_train.shape[1]
    nn = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    nn.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    nn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
    nn_predictions = nn.predict(X_test).flatten()
    # RMSE calculation for Neural Network
    nn_mse = np.mean((y_test - nn_predictions) ** 2)
    nn_rmse = np.sqrt(nn_mse)
    nn_mae = mean_absolute_error(y_test, nn_predictions)
    results["Neural Network"] = nn_rmse
    model_instances["Neural Network"] = nn
    save_plot_results(y_test, nn_predictions, model_name="Neural Network")
    
    # Summary of results
    best_model = min(results, key=results.get)
    print("\nSummary of Results:")
    print("===================================")
    for model_name, rmse in results.items():
        print(f"{model_name}: RMSE = {rmse:.4f}")
    print("\nBest Model (Lowest RMSE):")
    print(f"{best_model}: RMSE = {results[best_model]:.4f}")

    # Save best model information to a JSON file
    best_model_info = {
        "model_name": best_model,
        "rmse": results[best_model]
    }
    output_file = "reports/best_model.json"
    if not os.path.exists('reports'):
        os.makedirs('reports')
    with open(output_file, "w") as f:
        json.dump(best_model_info, f, indent=4)
    print(f"\nBest model information saved to {output_file}")

    # Save the best model
    best_model_instance = model_instances[best_model]
    model_file = "models/best_model.pkl"
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(best_model_instance, model_file)
    print(f"\nBest model saved to {model_file}")