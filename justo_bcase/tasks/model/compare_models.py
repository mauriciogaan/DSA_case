'''
This Python script compares the RMSE obtained from training 9 predictive models 
with 80% of the data and evaluating on 10%. (With out setting hyperparams)

'''

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import yaml

# Define paths
data_path = "c:/Users/itame/OneDrive/Desktop/justo_bcase/intermediates/"
hand_path = "c:/Users/itame/OneDrive/Desktop/justo_bcase/tasks/model/config/"

# open YAML file with parameters for data consolidation
stream = open(hand_path + "cv_params.yaml", 'r')
params = yaml.load(stream, Loader=yaml.Loader)

# Load data
df = pd.read_csv(data_path + "cleaned_data.csv")

X = df.drop(['Profit'], axis=1)
y = df['Profit']

# Define categorical and numerical features
categorical_features = ['Ship_Mode', 'Segment', 'State', 'Category', 'Sub-Category']
numerical_features = ['Sales', 'Quantity', 'Discount']

# Create the preprocessor with transformations for numerical and categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Standardize numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-hot encode categorical features
    ])

# Split the data into training and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=params["test_size"], random_state=params["seed"])
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=params["seed"])

# Set the random state for reproducibility
np.random.seed(params["seed"])

models = [LinearRegression(), RandomForestRegressor(), DecisionTreeRegressor(),
          GradientBoostingRegressor(), ElasticNet(), SVR(), Lasso(), Ridge(), XGBRegressor()]

model_names = ['Linear Regression', 'Random Forest', 'Decision Tree',
               'Gradient Boosting', 'Elastic Net', 'SVC', 'Lasso', 'Ridge', 'XGBoost']

model_rmse = []
models_ = []

for model in models:
    print(model)
    start = time.perf_counter()
    # Create a pipeline that first preprocesses the data then fits the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    models_.append(pipeline)
    
    # Calculate RMSE
    mse = mean_squared_error(y_valid, pipeline.predict(X_valid))
    rmse = np.sqrt(mse)
    
    # Save values
    model_rmse.append(rmse)
    # print duration
    duration = (time.perf_counter() - start)/60
    print(f"Duration: {duration} minutes")
    
df_models = pd.DataFrame(model_rmse, index=model_names, columns=['RMSE'])

# Display the models sorted by their RMSE
df_models_sorted = df_models.sort_values('RMSE')
df_models_sorted.to_csv(data_path + "comparing_pred_models.csv")
print(df_models_sorted)
