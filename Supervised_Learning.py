import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
import requests 

# The purpose of this code is to segregate the data between numerical and categorical data.
# Each division will have different ML functions associated witht he data type. 

# while api not implemented. 
def manual_assignment():
    
    df = pd.read_csv('data_1.csv')
    
    feature_1, feature_2 = 'ENGINE_SIZE', 'FUEL_CONSUMPTION'

    return df, feature_1, feature_2

# def api_data_fetch() :
    
#     url: str = 'API URL'
  
#     try:
        
#         response = requests.get(url)
        
#         if response.status_code == 200: 
            
#             data = response.json()
#             df1 = pd.DataFrame(data['dataset_1'])
#             df2 = pd.DataFrame(data['dataset_2'])
#             feature_1 = data['feature_1']
#             feature_2 = data['feature_2']
#             by = data['by']
            
#             return by,feature_1,feature_2,df
            
#         else: 
            
#             return None,None,None,None

#     except requests.exceptions.RequestException as e:
        
#         print(e)
        
#         return None,None,None,None

# Ensuring that the desired features (categories) are in the dataset.
def validate_categories(df: pd.DataFrame, feature_1: str, feature_2: str) -> bool:
    
    return {feature_1, feature_2}.issubset(df.columns)
 
# If the feature dont exist, a fuzz search recommends the closest alphabetical match to the desired feature.
def search_categories(df: pd.DataFrame, feature_1: str, feature_2: str) -> None:
    
    matches = pd.DataFrame([(cat, process.extract(cat, df.columns)[0][0]) 
                             for cat in [feature_1,feature_2] if not df.columns.__contains__(cat)], 
                             columns=['category','Match'])

    print([(f'Category: {matches.iloc[i,0]} should be changed to: {matches.iloc[i,1]}') for i in range(len(matches))])
                
# Ensuring that the features are either numerical or categorical
# Returns 1: if data is strictly numerical. 
# Returns 0: if data is categorical.               
def get_category_type(df: pd.DataFrame, feature_1: str, feature_2: str, logic: str) -> int: 
    
    types = [1 if typee == 'float64' or typee == 'int64' else 0 for typee in (df[[feature_1,feature_2]].dtypes)]
    
    return(eval(f'{types[0]} {logic} {types[1]}'))
    
# General statistics about data.     
def get_stats(df: pd.DataFrame, feature_1: str, feature_2: str) -> None:     

    print(f'General Statistics: \nCorrelation between {feature_1} and {feature_2}: {df[feature_1].corr(df[feature_2]).__round__(3)}', "\n",round(df[[feature_1,feature_2]].describe(),3),"\n\n",)

# Returns outliers in the dataset.
def get_outliers(df: pd.DataFrame, cat1: str):
        
    print('Outliers: ','\n', df.query(f' {cat1} < {df[cat1].quantile(0.25) - iqr(df[cat1] *  1.5)} or {cat1} >  {df[cat1].quantile(0.75) + iqr(df[cat1] *  1.5)}'))

def print_correlation(scores: list[int]) -> None:
    
    print(f'Cross Validation R^2 scores: {scores}')
    print(f'Average R^2 Score: {np.mean(scores).round(2)}.\nStd Dev R^2 Score: {np.std(scores).round(2)}.')
    
# Calculate the basic linear regression model (Non-regularized)
def lin_reg(df: pd.DataFrame, feature_1: str, feature_2: str) -> None:
    
    # Instantiate data & regression model
    X, y = df[f'{feature_1}'].values.reshape(-1,1), df[f'{feature_2}'].values
    
    # Instantiate regression model
    log_reg_model = LinearRegression()
    
    # Divide the X, y columns into training and testing data.
    # Testing data is 30% of the X,y columns. 
    # Establish a seed for consistent random re-selection.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    log_reg_model.fit(X_train, y_train)
    
    # Predict dependent y_test varible using the independent X_test variable. 
    y_predict = log_reg_model.predict(X_test)
    
    # Split data into 6(K = 6) parts (Folds) 
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    
    # Get regression coefficient for each fold with the linear reg model. 
    r_squared_scores = cross_val_score(log_reg_model,X,y,cv=kf).round(2)
    print_correlation(r_squared_scores)
    
    # Plot the regression line: 
    plot_reg_2D(feature_1, feature_2, X_test, y_test, y_predict, numeric=True)
    
# Calculates the Ridge Regression: Penalty for overfitting and underfitting
# Loss function = OLS + ( alpha * sum(coef(i)^2) )     
def lin_reg_ridge(df: pd.DataFrame, feature_1: str, feature_2: str) -> None:
    
    # Instantiate & split data
    X, y = df[f'{feature_1}'].values.reshape(-1,1), df[f'{feature_2}'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scores = []
    
    # Loops through different alphas in increments of 1
    for alpha in np.arange(0,1000,0.1):
        
        ridge = Ridge(alpha=alpha)
        
        ridge.fit(X_train, y_train)

        scores.append([ridge.score(X_test, y_test), alpha])

def lin_reg_lasso():
    
    raise NotImplementedError('Unfortunately, Lasso Regression has not been implemented yet.')
    
# Self explanatory    
def plot_reg_2D(X: str, y: str, X_test: list[int], y_test: list[int], y_pred: list[int], numeric: bool) -> None: 
    
    # Create a scatter of existing 
    plt.scatter(X_test, y_test)
    
    # Plot predicted y values on the test features
    plt.plot(X_test, y_pred, scalex=True, scaley=True, c='r')
    
    plt.title(f'{X} vs {y}')
    plt.xlabel(X)
    plt.ylabel(y)
    plt.show()    
    
def main(df: pd.DataFrame, feature_1: str, feature_2: str) -> None:
    
    if validate_categories(df, feature_1, feature_2):
   
        if (get_category_type(df, feature_1, feature_2, 'and')): 
            
            # get_stats(df, feature_1, feature_2)
            
            # get_outliers(df, feature_1)
            
            lin_reg(df, feature_1, feature_2)
            
            lin_reg_ridge(df, feature_1, feature_2)
        
        else: 
            
            if get_category_type(df, feature_1, feature_2, 'or'):
                
                raise NotImplementedError('Unfortunately, categorical vs numerical data processing has not been implemented yet')
            
            else:
    
                raise NotImplementedError('Unfortunately, categorical vs categorical data processing has not been implemented yet')
                
    else:
        
        search_categories(df, feature_1, feature_2)

if __name__ == "__main__":

    df,feature_1, feature_2 = manual_assignment()
    
    main(df, feature_1, feature_2)
    
    
