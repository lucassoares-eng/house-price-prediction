import os
import pickle
import subprocess
import shap
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the dataset."""
    if 'SalePrice' in df.columns:
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

# Function to collect user input
def user_input_features():
    st.sidebar.header('Features for Prediction')

    # Initial value for categorical features preprocess columns
    MSZoning_FV = MSZoning_RH = MSZoning_RL = MSZoning_RM = Street_Pave = LotShape_IR2 = LotShape_IR3 = LotShape_Reg = Utilities_NoSeWa = LotConfig_CulDSac = LotConfig_FR2 = LotConfig_FR3 = LotConfig_Inside = LandSlope_Mod = LandSlope_Sev = RoofStyle_Gable = RoofStyle_Gambrel = RoofStyle_Hip = RoofStyle_Mansard = RoofStyle_Shed = ExterQual_Fa = ExterQual_Gd = ExterQual_TA = Foundation_CBlock = Foundation_PConc = Foundation_Slab = Foundation_Stone = Foundation_Wood = BsmtQual_Fa = BsmtQual_Gd = BsmtQual_None = BsmtQual_TA = Heating_GasA = Heating_GasW = Heating_Grav = Heating_OthW = Heating_Wall = HeatingQC_Fa = HeatingQC_Gd = HeatingQC_Po = HeatingQC_TA = CentralAir_Y = Electrical_FuseF = Electrical_FuseP = Electrical_Mix = Electrical_None = Electrical_SBrkr = KitchenQual_Fa = KitchenQual_Gd = KitchenQual_TA = FireplaceQu_Fa = FireplaceQu_Gd = FireplaceQu_None = FireplaceQu_Po = FireplaceQu_TA = GarageType_Attchd = GarageType_Basment = GarageType_BuiltIn = GarageType_CarPort = GarageType_Detchd = GarageType_None = GarageQual_Fa = GarageQual_Gd = GarageQual_None = GarageQual_Po = GarageQual_TA = PoolQC_Fa = PoolQC_Gd = PoolQC_None = Fence_GdWo = Fence_MnPrv = Fence_MnWw = Fence_None = MiscFeature_None = MiscFeature_Othr = MiscFeature_Shed = MiscFeature_TenC = False

    # Numerical features (for continuous data)
    LotFrontage = st.sidebar.number_input('LotFrontage', min_value=0, max_value=1000, value=80, help="Linear feet of street connected to property")
    LotArea = st.sidebar.number_input('LotArea', min_value=0, max_value=1000000, value=5000, help="Lot size in square feet")
    OverallQual = st.sidebar.selectbox('OverallQual', [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], help="Rates the overall material and finish of the house. 10	Very Excellent, 9 Excellent, 8 Very Good, 7 Good, 6 Above Average, 5 Average, 4 Below Average, 3 Fair, 2 Poor, 1 Very Poor")
    YearBuilt = st.sidebar.number_input('YearBuilt', min_value=1900, max_value=2024, value=2000, help="Original construction date")
    YearRemodAdd = st.sidebar.number_input('YearRemodAdd', min_value=1900, max_value=2024, value=2005, help="Remodel date (same as construction date if no remodeling or additions)")
    TotalBsmtSF = st.sidebar.number_input('TotalBsmtSF', min_value=0, max_value=5000, value=800, help="Total square feet of basement area")
    fstFlrSF = st.sidebar.number_input('1stFlrSF', min_value=0, max_value=5000, value=1000, help="First Floor square feet")
    sndFlrSF = st.sidebar.number_input('2ndFlrSF', min_value=0, max_value=5000, value=500, help="Second floor square feet")
    GrLivArea = st.sidebar.number_input('GrLivArea', min_value=0, max_value=5000, value=2000, help="Above grade (ground) living area square feet")
    BsmtFullBath = st.sidebar.number_input('BsmtFullBath', min_value=0, max_value=5, value=1, help="BsmtFullBath: Basement full bathrooms")
    BsmtHalfBath = st.sidebar.number_input('BsmtHalfBath', min_value=0, max_value=5, value=1, help="Basement half bathrooms")
    FullBath = st.sidebar.number_input('FullBath', min_value=0, max_value=5, value=2, help="Full bathrooms above grade")
    HalfBath = st.sidebar.number_input('HalfBath', min_value=0, max_value=5, value=1, help="Half baths above grade")
    Bedroom = st.sidebar.number_input('Bedroom', min_value=0, max_value=10, value=3, help="Bedrooms above grade (does NOT include basement bedrooms)")
    Kitchen = st.sidebar.number_input('Kitchen', min_value=0, max_value=5, value=1, help="Kitchens above grade")
    TotRmsAbvGrd = st.sidebar.number_input('TotRmsAbvGrd', min_value=0, max_value=20, value=8, help="Total rooms above grade (does not include bathrooms)")
    Fireplaces = st.sidebar.number_input('Fireplaces', min_value=0, max_value=5, value=0, help="Number of fireplaces")
    GarageCars = st.sidebar.number_input('GarageCars', min_value=0, max_value=5, value=1, help="Size of garage in car capacity")
    WoodDeckSF = st.sidebar.number_input('WoodDeckSF', min_value=0, max_value=1000, value=100, help="Wood deck area in square feet")
    OpenPorchSF = st.sidebar.number_input('OpenPorchSF', min_value=0, max_value=1000, value=50, help="Open porch area in square feet")
    EnclosedPorch = st.sidebar.number_input('EnclosedPorch', min_value=0, max_value=1000, value=20, help="Enclosed porch area in square feet")
    PoolArea = st.sidebar.number_input('PoolArea', min_value=0, max_value=1000, value=0, help="Pool area in square feet")

    # Categorical features (multiple choice or single choice)
    MSZoning = st.sidebar.selectbox('MSZoning', ['FV', 'RH', 'RL', 'RM'], help="Identifies the general zoning classification of the sale. A	Agriculture, C Commercial, FV Floating Village Residential, I Industrial, RH Residential High Density, RL Residential Low Density, RP Residential Low Density, RM Residential Medium Density")
    globals()[f'MSZoning_{MSZoning}'] = True
    Street = st.sidebar.selectbox('Street', ['Pave'],help="Type of road access to property. Grvl Gravel, Pave Paved")
    globals()[f'Street_{Street}'] = True
    LotShape = st.sidebar.selectbox('LotShape', ['Reg', 'IR2', 'IR3'], help="General shape of property. Reg Regular, IR1 Slightly irregular, IR2 Moderately Irregular, IR3 Irregular")
    globals()[f'LotShape_{LotShape}'] = True
    Utilities = st.sidebar.selectbox('Utilities', ['NoSeWa'], help="Type of utilities available. AllPub All public Utilities (E,G,W,& S), NoSewr Electricity Gas and Water (Septic Tank), NoSeWa Electricity and Gas Only, ELO Electricity only")
    globals()[f'Utilities_{Utilities}'] = True
    LotConfig = st.sidebar.selectbox('LotConfig', ['Inside', 'CulDSac', 'FR2', 'FR3'], help="Lot configuration. Inside Inside lot, Corner Corner lot, CulDSac Cul-de-sac, FR2 Frontage on 2 sides of property, FR3 Frontage on 3 sides of property")
    globals()[f'LotConfig_{LotConfig}'] = True
    LandSlope = st.sidebar.selectbox('LandSlope', ['Mod', 'Sev'], help="Slope of property. Gtl Gentle slope, Mod Moderate Slope, Sev Severe Slope")
    globals()[f'LandSlope_{LandSlope}'] = True
    RoofStyle = st.sidebar.selectbox('RoofStyle', ['Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'], help="Type of roof")
    globals()[f'RoofStyle_{RoofStyle}'] = True
    ExterQual = st.sidebar.selectbox('ExterQual', ['Gd', 'TA', 'Fa'], help="Evaluates the quality of the material on the exterior. Ex Excellent, Gd Good, TA Average/Typical, Fa Fair, Po Poor")
    globals()[f'ExterQual_{ExterQual}'] = True
    Foundation = st.sidebar.selectbox('Foundation', ['CBlock', 'PConc', 'Slab', 'Stone', 'Wood'], help="Type of foundation. BrkTil Brick & Tile, CBlock Cinder Block, PConc Poured Contrete, Slab, Stone, Wood")
    globals()[f'Foundation_{Foundation}'] = True
    BsmtQual = st.sidebar.selectbox('BsmtQual', ['Gd', 'TA', 'Fa', 'None'], help="Evaluates the height of the basement. Ex Excellent (100+ inches), Gd Good (90-99 inches), TA Typical (80-89 inches), Fa Fair (70-79 inches), Po Poor (<70 inches), None No Basement")
    globals()[f'BsmtQual_{BsmtQual}'] = True
    Heating = st.sidebar.selectbox('Heating', ['GasA', 'GasW', 'Grav', 'OthW', 'Wall'], help="Type of heating. Floor Floor Furnace, GasA Gas forced warm air furnace, GasW Gas hot water or steam heat, Grav Gravity furnace, OthW Hot water or steam heat other than gas, Wall Wall furnace")
    globals()[f'Heating_{Heating}'] = True
    HeatingQC = st.sidebar.selectbox('HeatingQC', ['Gd', 'TA', 'Fa', 'Po'], help="Heating quality and condition, Ex Excellent, Gd Good, TA Average/Typical, Fa Fair, Po Poor")
    globals()[f'Heating_{Heating}'] = True
    CentralAir = st.sidebar.selectbox('CentralAir', ['Y'], help="Central air conditioning, N No, Y Yes")
    globals()[f'CentralAir_{CentralAir}'] = True
    Electrical = st.sidebar.selectbox('Electrical', ['SBrkr', 'FuseF', 'FuseP', 'Mix'], help="Electrical system. SBrkr Standard Circuit Breakers & Romex, FuseA Fuse Box over 60 AMP and all Romex wiring (Average), FuseF 60 AMP Fuse Box and mostly Romex wiring (Fair), FuseP 60 AMP Fuse Box and mostly knob & tube wiring (poor), Mix Mixed")
    globals()[f'Electrical_{Electrical}'] = True
    KitchenQual = st.sidebar.selectbox('KitchenQual', ['Gd', 'TA', 'Fa'], help="Kitchen quality. Ex Excellent, Gd Good, TA Typical/Average, Fa Fair, Po Poor")
    globals()[f'KitchenQual_{KitchenQual}'] = True
    FireplaceQu = st.sidebar.selectbox('FireplaceQu', ['Gd', 'TA', 'Fa', 'Po', 'None'], help="Fireplace quality. Ex Excellent - Exceptional Masonry Fireplace, Gd Good - Masonry Fireplace in main level, TA Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement, Fa Fair - Prefabricated Fireplace in basement, Po Poor - Ben Franklin Stove, None No Fireplace")
    globals()[f'FireplaceQu_{FireplaceQu}'] = True
    GarageType = st.sidebar.selectbox('GarageType', ['Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'None'], help="Garage location. 2Types More than one type of garage, Attchd Attached to home, Basment Basement Garage, BuiltIn Built-In (Garage part of house - typically has room above garage), CarPort Car Port, Detchd Detached from home, None No Garage")
    globals()[f'GarageType_{GarageType}'] = True
    GarageQual = st.sidebar.selectbox('GarageQual', ['Gd', 'TA', 'Fa', 'Po', 'None'], help="Garage quality. Ex Excellent, Gd Good, TA Typical/Average, Fa Fair, Po Poor, None No Garage")
    globals()[f'GarageQual_{GarageQual}'] = True
    PoolQC = st.sidebar.selectbox('PoolQC', ['Gd', 'Fa', 'None'], help="Pool quality. Ex Excellent, Gd Good, TA Average/Typical, Fa Fair, None No Pool")
    globals()[f'PoolQC_{PoolQC}'] = True
    Fence = st.sidebar.selectbox('Fence', ['MnPrv', 'GdWo', 'MnWw', 'None'], help="Fence quality. GdPrv Good Privacy, MnPrv Minimum Privacy, GdWo Good Wood, MnWw Minimum Wood/Wire, None No Fence")
    globals()[f'Fence_{Fence}'] = True
    MiscFeature = st.sidebar.selectbox('MiscFeature', ['Othr', 'Shed', 'TenC', 'None'], help="Miscellaneous feature not covered in other categories. Elev Elevator, Gar2 2nd Garage (if not described in garage section), Othr Other, Shed Shed (over 100 SF), TenC Tennis Court, None")
    globals()[f'MiscFeature_{MiscFeature}'] = True

    # Organize the inputs into a DataFrame
    input_data = {
        'LotFrontage': LotFrontage,
        'LotArea': LotArea,
        'OverallQual': OverallQual,
        'YearBuilt': YearBuilt,
        'YearRemodAdd': YearRemodAdd,
        'TotalBsmtSF': TotalBsmtSF,
        '1stFlrSF': fstFlrSF,
        '2ndFlrSF': sndFlrSF,
        'GrLivArea': GrLivArea,
        'BsmtFullBath': BsmtFullBath,
        'BsmtHalfBath': BsmtHalfBath,
        'FullBath': FullBath,
        'HalfBath': HalfBath,
        'BedroomAbvGr': Bedroom,
        'Kitchen': Kitchen,
        'TotRmsAbvGrd': TotRmsAbvGrd,
        'Fireplaces': Fireplaces,
        'GarageCars': GarageCars,
        'WoodDeckSF': WoodDeckSF,
        'OpenPorchSF': OpenPorchSF,
        'EnclosedPorch': EnclosedPorch,
        'PoolArea': PoolArea,
        'MSZoning_FV': MSZoning_FV,
        'MSZoning_RH': MSZoning_RH,
        'MSZoning_RL': MSZoning_RL,
        'MSZoning_RM': MSZoning_RM,
        'Street_Pave': Street_Pave,
        'LotShape_IR2': LotShape_IR2,
        'LotShape_IR3': LotShape_IR3,
        'LotShape_Reg': LotShape_Reg,
        'Utilities_NoSeWa': Utilities_NoSeWa,
        'LotConfig_CulDSac': LotConfig_CulDSac,
        'LotConfig_FR2': LotConfig_FR2,
        'LotConfig_FR3': LotConfig_FR3,
        'LotConfig_Inside': LotConfig_Inside,
        'LandSlope_Mod': LandSlope_Mod,
        'LandSlope_Sev': LandSlope_Sev,
        'RoofStyle_Gable': RoofStyle_Gable,
        'RoofStyle_Gambrel': RoofStyle_Gambrel,
        'RoofStyle_Hip': RoofStyle_Hip,
        'RoofStyle_Mansard': RoofStyle_Mansard,
        'RoofStyle_Shed': RoofStyle_Shed,
        'ExterQual_Fa': ExterQual_Fa,
        'ExterQual_Gd': ExterQual_Gd,
        'ExterQual_TA': ExterQual_TA,
        'Foundation_CBlock': Foundation_CBlock,
        'Foundation_PConc': Foundation_PConc,
        'Foundation_Slab': Foundation_Slab,
        'Foundation_Stone': Foundation_Stone,
        'Foundation_Wood': Foundation_Wood,
        'BsmtQual_Fa': BsmtQual_Fa,
        'BsmtQual_Gd': BsmtQual_Gd,
        'BsmtQual_None': BsmtQual_None,
        'BsmtQual_TA': BsmtQual_TA,
        'Heating_GasA': Heating_GasA,
        'Heating_GasW': Heating_GasW,
        'Heating_Grav': Heating_Grav,
        'Heating_OthW': Heating_OthW,
        'Heating_Wall': Heating_Wall,
        'HeatingQC_Fa': HeatingQC_Fa,
        'HeatingQC_Gd': HeatingQC_Gd,
        'HeatingQC_Po': HeatingQC_Po,
        'HeatingQC_TA': HeatingQC_TA,
        'CentralAir_Y': CentralAir_Y,
        'Electrical_FuseF': Electrical_FuseF,
        'Electrical_FuseP': Electrical_FuseP,
        'Electrical_Mix': Electrical_Mix,
        'Electrical_None': Electrical_None,
        'Electrical_SBrkr': Electrical_SBrkr,
        'KitchenQual_Fa': KitchenQual_Fa,
        'KitchenQual_Gd': KitchenQual_Gd,
        'KitchenQual_TA': KitchenQual_TA,
        'FireplaceQu_Fa': FireplaceQu_Fa,
        'FireplaceQu_Gd': FireplaceQu_Gd,
        'FireplaceQu_None': FireplaceQu_None,
        'FireplaceQu_Po': FireplaceQu_Po,
        'FireplaceQu_TA': FireplaceQu_TA,
        'GarageType_Attchd': GarageType_Attchd,
        'GarageType_Basment': GarageType_Basment,
        'GarageType_BuiltIn': GarageType_BuiltIn,
        'GarageType_CarPort': GarageType_CarPort,
        'GarageType_Detchd': GarageType_Detchd,
        'GarageType_None': GarageType_None,
        'GarageQual_Fa': GarageQual_Fa,
        'GarageQual_Gd': GarageQual_Gd,
        'GarageQual_None': GarageQual_None,
        'GarageQual_Po': GarageQual_Po,
        'GarageQual_TA': GarageQual_TA,
        'PoolQC_Fa': PoolQC_Fa,
        'PoolQC_Gd': PoolQC_Gd,
        'PoolQC_None': PoolQC_None,
        'Fence_GdWo': Fence_GdWo,
        'Fence_MnPrv': Fence_MnPrv,
        'Fence_MnWw': Fence_MnWw,
        'Fence_None': Fence_None,
        'MiscFeature_None': MiscFeature_None,
        'MiscFeature_Othr': MiscFeature_Othr,
        'MiscFeature_Shed': MiscFeature_Shed,
        'MiscFeature_TenC': MiscFeature_TenC
    }
    
    features = pd.DataFrame(input_data, index=[0])
    
    return features

# ========== Main Workflow ==========
if __name__ == "__main__":
    try:
        model_file = "models/best_model.pkl"
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        # Open and read the JSON file
        json_file_path = "reports/best_model.json"
        with open(json_file_path, 'r') as file:
            best_model_data = json.load(file)
        # Retrieve the model_name
        model_name = best_model_data.get("model_name", None)
        print(f"Model {model_name} loaded successfully!")
    except FileNotFoundError:
        print("Model not found. Running `main.py` to train the model...")
        # Execute main.py
        subprocess.run(["python", "main.py"], check=True)
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        print(f"Model {model_name} loaded successfully after training!")

    # Load the dataset (ensure preprocessing matches training)
    data_path = "data/house_prices.csv"
    raw_data = load_data(data_path)

    # Preprocess the data (update this with your preprocessing function)
    data = preprocess_data(raw_data.copy())  # Ensure this function matches your training pipeline
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]
    # Train-Test Split
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Select a sample of background data
    background_data = shap.sample(X, 100)  # Use 100 samples for KernelExplainer

    if model_name in ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"]:
        # TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
    elif model_name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
        # LinearExplainer for linear models
        explainer = shap.LinearExplainer(model, background_data)
    elif model_name == "SVR":
        # KernelExplainer for black-box models like SVR
        explainer = shap.KernelExplainer(model.predict, background_data)
    elif model_name == "Neural Network":
        # DeepExplainer for deep learning models
        explainer = shap.DeepExplainer(model.predict, background_data)        

    # Compute SHAP values
    shap_values = explainer.shap_values(X[:10])  # Explain first 10 rows
    
    # Save SHAP plot
    shap_plot_path = "reports/shap_summary_plot.png"
    if not os.path.exists('reports'):
        os.makedirs('reports')
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X[:10], show=False)
    plt.savefig(shap_plot_path, format='png')
    plt.close()

    # Interactive Frontend with Streamlit
    st.title("House Price Prediction")
    st.write("### Model Performance Metrics")
    st.text(f"Model: {model.__class__.__name__}")
    
    # Predict using the loaded model
    predictions = model.predict(X_test)

    # Display predictions vs actual prices
    st.write("### Predictions vs Actual Prices")
    prediction_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    st.write(prediction_df.head())

    # Visualize SHAP explanations for the first 10 samples
    st.write("### SHAP Summary Plot")

    st.image(shap_plot_path, caption="SHAP Summary Plot", use_container_width=True)

    # Selector for the user to choose input data (example)
    st.subheader("Enter the features for prediction:")

    # Collect user input
    user_data = user_input_features()

    # Make the prediction
    prediction = model.predict(user_data)

    # Display the prediction
    st.subheader("Predicted House Price:")
    st.write(f"${prediction[0]:,.2f}")

    # Show SHAP graph
    st.subheader("SHAP Values for the Prediction:")
    # Generate SHAP values for the input data
    shap_values_for_input = explainer.shap_values(user_data)
    # Create the force plot as an HTML file
    shap_plot_html = shap.force_plot(
        explainer.expected_value[0], 
        shap_values_for_input[0], 
        user_data, 
        feature_names=user_data.columns
    )
    # Save the HTML plot to a temporary file
    from streamlit.components.v1 import html
    shap_html = f"<head>{shap.getjs()}</head><body>{shap_plot_html.html()}</body>"
    html(shap_html, height=120)

    st.markdown("---")
    st.markdown("### **Contributions**")
    st.markdown(
        """
        Contributions are welcome! Feel free to open issues and submit pull requests to improve the project.

        [Project link](https://github.com/lucassoares-eng/house-price-prediction)

        ```bash
        git clone https://github.com/lucassoares-eng/house-price-prediction.git
        ```

        This project is licensed under the APACHE License.
        """
    )

    st.markdown("### **Contact**")
    st.markdown(
        """
        If you have questions or suggestions, feel free to reach out:

        - Email: [lucasjs.eng@gmail.com](mailto:lucasjs.eng@gmail.com)
        - LinkedIn: [Lucas de Jesus Soares](https://www.linkedin.com/in/lucas-de-jesus-soares-33486b42/)
        """
    )