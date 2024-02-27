'''
This Python script performs an GridSearchCV (or RandomizedSearchCV) to find
the best combination of hyperparms. You can set the parameters in the .yaml file 
named cv_params.yaml 

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Define paths
data_path = "c:/Users/itame/OneDrive/Desktop/justo_bcase/intermediates/"
outcomes_path = "c:/Users/itame/OneDrive/Desktop/justo_bcase/outcomes/model/"
hand_path = "c:/Users/itame/OneDrive/Desktop/justo_bcase/tasks/model/config/"

# open YAML file with parameters 
stream = open(hand_path + "cv_params.yaml", 'r')
params = yaml.load(stream, Loader=yaml.Loader)

#Load data
df = pd.read_csv(data_path + "cleaned_data.csv")

# Define features and target variable
X = df.drop(['Profit'], axis=1)  # Features
y = df['Profit']  # Target variable

# Define categorical and numerical features
categorical_features = ['Ship_Mode', 'Segment', 'State', 'Category', 'Sub-Category', 'Postal_Code']
numerical_features = ['Sales', 'Quantity', 'Discount']

# Create the preprocessor with transformations for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Standardize numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-hot encode categorical features
    ])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["test_size"], random_state = params['seed'])

# Define models to be evaluated
models = [XGBRegressor(),  GradientBoostingRegressor(), RandomForestRegressor()]
model_names = ['XGBoost', 'Gradient Boosting', 'Random Forest']

# Dictionary to map model names to their parameter spaces
param_grids = {
    'XGBoost': {
        'model__n_estimators': params["gb_n_estimators"],
        'model__learning_rate': params["gb_learning_rate"],
        'model__max_depth': params["gb_max_depth"]
    },
    'Gradient Boosting': {
        'model__n_estimators': params["gb_n_estimators"],
        'model__learning_rate': params["gb_learning_rate"],
        'model__max_depth': params["gb_max_depth"]
    },
    'Random Forest': {
        'model__n_estimators': params["rf_n_estimators"],
        'model__max_depth': params["rf_max_depth"],
        'model__min_samples_split': params["rf_min_samples_split"]
    }
    # Add other models and their parameters here as needed
}

# Set up Cross-Validation
cv = KFold(n_splits= params['n_splits'], shuffle=True, random_state=params['seed'])

results = []

for model, name in zip(models, model_names):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),  # Preprocessing step
                               ('model', model)])  # Model step
    
    # Select the correct parameter space
    param_grid = param_grids.get(name, {})
    
    # Use GridSearchCV if the model has a defined parameter space
    if param_grid:
        search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', cv=cv, verbose=2, n_jobs=-1)
        #search = RandomizedSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', n_iter=50, cv=cv, verbose=2, n_jobs=-1)
    else:
        # If not, just fit the model without hyperparameter search
        search = pipeline

    search.fit(X_train, y_train)
    
    if param_grid:
        print(f"Best Hyperparameters for {name}:", search.best_params_)
        best_score = np.sqrt(-search.best_score_)
        print(f"Best RMSE (Cross-Validation) for {name}:", best_score)
    else:
        predictions = search.predict(X_train)
        best_score = np.sqrt(mean_squared_error(y_train, predictions))
        print(f"RMSE (training) for {name}: {best_score}")

    predictions = search.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    test_mae = mean_absolute_error(y_test, predictions)
    test_r2 = r2_score(y_test, predictions)

    # Create a DataFrame comparing actual and predicted values
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    comparison_file_path = f'{outcomes_path}actual_vs_predicted_{name}.csv'  # Adjust path as needed
    comparison_df.to_csv(comparison_file_path, index=False)
    
    
    errors = comparison_df['Actual'] - comparison_df['Predicted']
    
    # Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(comparison_df['Actual'], comparison_df['Predicted'], alpha=0.5)
    plt.title('Actual vs. Predicted Values')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.plot([comparison_df['Actual'].min(), comparison_df['Actual'].max()], [comparison_df['Actual'].min(), comparison_df['Actual'].max()], 'k--', lw=2)
    plt.savefig(f'{outcomes_path}scatter_plot_{name}.png')
    plt.close()

    # Histogram of Errors
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, edgecolor='k', alpha=0.7)
    plt.title('Histogram of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.savefig(f'{outcomes_path}error_histogram_{name}.png')
    plt.close()

    # Density plot Actual vs. Predicted
    plt.figure(figsize=(10, 6))
    sns.kdeplot(comparison_df['Actual'], label='Actual', shade=True)
    sns.kdeplot(comparison_df['Predicted'], label='Predicted', shade=True)
    plt.title('Density Plot of Actual and Predicted Values')
    plt.xlabel('Value')
    plt.legend()
    plt.savefig(f'{outcomes_path}density_plot_{name}.png')
    plt.close()
    
    #Scatter of the Resids
    plt.figure(figsize=(10, 6))
    plt.scatter(comparison_df['Predicted'], errors, alpha=0.5)
    plt.title(f'Residuals vs. Predicted Values for {name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.savefig(f'{outcomes_path}residuals_scatter_{name}.png')
    plt.close()
    
    
    results.append({
        'Model': name,
        'Test RMSE': test_rmse,
        'Test MAE': test_mae,
        'Test R2': test_r2
    })

# Store and display results
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv(f'{outcomes_path}cv_params_models.csv', index=False)
