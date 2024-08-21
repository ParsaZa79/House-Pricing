import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Clean and convert the 'Area' column
    df['Area'] = df['Area'].replace({',': ''}, regex=True).astype(float)
    df['Area'] = df['Area'].apply(lambda x: x / 10000 if x > 1000 else x)
    
    # Convert boolean columns to numeric
    bool_columns = ['Parking', 'Warehouse', 'Elevator']
    for col in bool_columns:
        df[col] = df[col].map({'True': 1, 'False': 0})
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = numeric_columns.drop(bool_columns)  # Exclude boolean columns from imputation
    imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    # Separate features and target
    X = df.drop(['Price', 'Price(USD)', 'Address'], axis=1)
    y = df['Price(USD)']
    
    # Perform feature scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, X, y

# The rest of your code remains the same
# The rest of your code remains the same

# Exploratory Data Analysis (EDA)
def perform_eda(X, y):
    # Visualize the distribution of features
    X.hist(figsize=(15, 10))
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()
    
    # Analyze feature correlations
    correlation_matrix = X.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('feature_correlations.png')
    plt.close()
    
    # Identify and visualize outliers
    plt.figure(figsize=(15, 10))
    X.boxplot()
    plt.title('Boxplot of Features')
    plt.xticks(rotation=45)
    plt.savefig('feature_boxplots.png')
    plt.close()

# Model Building and Evaluation
def build_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Initialize and train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation MSE scores: {-cv_scores}")
    print(f"Average CV MSE: {-cv_scores.mean()}")
    
    return model

# Hyperparameter Tuning
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

# Model Interpretation
def interpret_model(model, X):
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    plt.close()

# Main function
def main():
    file_path = 'housePrice.csv'
    
    # Data Preprocessing
    X_train, X_test, y_train, y_test, X, y = load_and_preprocess_data(file_path)
    
    # Exploratory Data Analysis
    perform_eda(X, y)
    
    # Model Building and Evaluation
    model = build_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # Hyperparameter Tuning
    best_model = tune_hyperparameters(X_train, y_train)
    
    # Model Interpretation
    interpret_model(best_model, X)
    
    # Final Report
    print("\nFinal Report:")
    print("1. Data Preprocessing:")
    print("   - Loaded the dataset and handled missing values using mean imputation.")
    print("   - Performed feature scaling using StandardScaler.")
    print("   - Split the data into training (80%) and testing (20%) sets.")
    print("\n2. Exploratory Data Analysis:")
    print("   - Generated visualizations for feature distributions, correlations, and outliers.")
    print("   - Saved plots as 'feature_distributions.png', 'feature_correlations.png', and 'feature_boxplots.png'.")
    print("\n3. Model Building and Evaluation:")
    print("   - Used Random Forest Regressor as the main model.")
    print("   - Evaluated the model using MSE, MAE, and R-squared metrics.")
    print("   - Performed cross-validation to assess model stability.")
    print("\n4. Hyperparameter Tuning:")
    print("   - Used GridSearchCV to find the best hyperparameters for the Random Forest model.")
    print("\n5. Model Interpretation:")
    print("   - Analyzed feature importance and saved the plot as 'feature_importance.png'.")
    print("\n6. Potential Improvements:")
    print("   - Consider feature engineering to create new meaningful features.")
    print("   - Experiment with other algorithms like Gradient Boosting or Neural Networks.")
    print("   - Collect more data if possible to improve model performance.")
    print("   - Investigate the impact of outliers and consider robust scaling techniques.")

if __name__ == "__main__":
    main()