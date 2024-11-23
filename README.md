# House Price Prediction (Regression)

This repository showcases a machine learning project designed to predict house prices using various regression algorithms and a neural network for performance comparison. The project explores different models to determine the best-performing one based on accuracy metrics. You can view the results and make your own predictions through this [interactive link](https://lucasjs-house-price-prediction.streamlit.app/).

---

## **Objective**

The objective of this project is to develop a robust and efficient machine learning pipeline to predict house prices based on various features from a real estate dataset. By leveraging a combination of traditional regression models and neural networks, the project aims to:

- Train multiple machine learning models and evaluate their performance using key metrics like RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
- Identify the best-performing model and save it for future predictions.
- Provide meaningful visualizations to compare actual and predicted prices, enabling better insights into model performance.
- Demonstrate a scalable and modular approach to building machine learning workflows that can be adapted to similar regression problems.

---

## **Features**

### **Models Used**
1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Decision Tree**
5. **Random Forest**
6. **Gradient Boosting**
7. **XGBoost**
8. **LightGBM**
9. **Support Vector Regression (SVR)**
10. **Neural Network (Keras/TensorFlow)**

---

## **Project Pipeline**

### 1. **Data Loading**
   - Data is loaded from a CSV file.
   - Ensure the file is located in the `data/house_prices.csv` directory.
   - The dataset used is publicly available from the Kaggle House Prices - Advanced Regression Techniques competition.
   
    👉 [Kaggle Dataset Link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)  

### 2. **Preprocessing**
   - Filling missing values.
   - Transforming categorical variables into dummies.

### 3. **Data Splitting**
   - Training set: 80%
   - Test set: 20%

### 4. **Model Training**
   - Each model is trained using the training set.
   - The neural network is configured with:
     - 128 neurons in the first layer.
     - 64 neurons in the second layer.
     - `ReLU` activation function.

### 5. **Evaluation**
   - **RMSE (Root Mean Squared Error)**: Manually calculated using `numpy`.
   - **MAE (Mean Absolute Error)**: Calculated with `scikit-learn`.
   - Comparison of actual and predicted values using scatter plots.

### 6. **Select the Best Model**
   - The model with the lowest RMSE is selected as the best-performing model.
   - Model details and metrics are saved to a JSON file for documentation.

### 7. **Build a Streamlit App**
   - A Streamlit application is developed to:
     - Display results.
     - Perform SHAP analysis for interpretability.
     - Dynamically make new predictions based on user inputs.

---

## **Code Structure**

- **`main.py`**: Main script containing the entire project pipeline.
- **`app.py`**: Streamlit application to visualize results, perform SHAP analysis, and make predictions dynamically.
- **`data/house_prices.csv`**: Dataset file (not included in this repository).
- **`requirements.txt`**: Project dependencies.

---

## **How to Run**

1. Clone the repository:
```bash
   git clone https://github.com/lucassoares-eng/house-price-prediction.git
```
2. Install the dependencies:
```bash
   pip install -r requirements.txt
```
   
3. Ensure the data file is located in the data/raw/ folder with the name house_prices.csv.
4. Run the main script:
```bash
   python main.py
```
5. Run the Streamlit app:
```bash
   streamlit run app.py
```

---

## **Dependencies**

- Python 3.8+
- Libraries:
    - pandas
    - numpy
    - scikit-learn
    - xgboost
    - lightgbm
    - tensorflow
    - matplotlib
    - seaborn
    - shap
    - streamlit

To install all dependencies, use:
```bash
   pip install -r requirements.txt
```

---

## **Contributions**

Contributions are welcome! Feel free to open issues and submit pull requests to improve the project.

---

## **License**

This project is licensed under the APACHE License. See the `LICENSE` file for more details.

---

## **Contact**

If you have questions or suggestions, feel free to reach out:

Email: lucasjs.eng@gmail.com

LinkedIn: https://www.linkedin.com/in/lucas-de-jesus-soares-33486b42/