import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr
from fuzzywuzzy import process
import requests 


def manual_assignment():
    
    df = pd.read_csv('data_1.csv')
    
    category_1, category_2 = 'MAKE', 'FUEL_CONSUMPTION'

    return df, category_1, category_2

def api_data_fetch() :
    
    url: str = 'API URL'
  
    try:
        
        response = requests.get(url)
        
        if response.status_code == 200: 
            
            data = response.json()
            df1 = pd.DataFrame(data['dataset_1'])
            df2 = pd.DataFrame(data['dataset_2'])
            category_1 = data['category_1']
            category_2 = data['category_2']
            by = data['by']
            
            return by,category_1,category_2,df
            
        else: 
            
            return None,None,None,None

    except requests.exceptions.RequestException as e:
        
        print(e)
        
        return None,None,None,None

def validate_categories(df: pd.DataFrame, category_1: str, category_2: str) -> bool:

    return {category_1, category_2}.issubset(df.columns)
 
 
def search_categories(df: pd.DataFrame, category_1: str, category_2: str) -> None:

    matches = pd.DataFrame([(cat, process.extract(cat, df.columns)[0][0]) 
                             for cat in [category_1,category_2] if not df.columns.__contains__(cat)], 
                             columns=['category','Match'])

    print([(f'Category: {matches.iloc[i,0]} should be changed to: {matches.iloc[i,1]}') for i in range(len(matches))])
                
                
def get_category_type(df: pd.DataFrame, category_1: str, category_2: str, logic: str) -> int: 
    
    types = [1 if typee == 'float64' or typee == 'int64' else 0 for typee in (df[[category_1,category_2]].dtypes)]
    # print(eval(f'{types[0]} {logic} {types[1]}'))
    return(eval(f'{types[0]} {logic} {types[1]}'))
    
    
def get_stats(df: pd.DataFrame, cat1: str, cat2: str) -> None:     

    print(f'Correlation between {category_1} and {category_2}: {df[category_1].corr(df[category_2])}', "\n\n",df[[cat1,cat2]].describe(),"\n\n",)


def get_outliers(df: pd.DataFrame, cat1: str):
        
    print(df.query(f' {cat1} < {df[cat1].quantile(0.25) - iqr(df[cat1] *  1.5)} or {cat1} >  {df[cat1].quantile(0.75) + iqr(df[cat1] *  1.5)}'))

    
def plot_graphs(df: pd.DataFrame, var_ind: str, var_dep:  str, rotation: int, numeric: bool) -> None:
    
    df.plot(kind='scatter', x=var_ind, y=var_dep, title=f'{var_ind} vs {var_dep}', rot=rotation)
    
    if numeric: 
        
        weight,bias = regression(df, var_ind, var_dep)
                
        plt.plot(df[category_1], (df[category_1] * weight + bias), color='r')
        
    plt.show()

def gradient_descent(weight, bias, l_rate, series_x: pd.Series, series_y: pd.Series) -> list[float,float]: 
    
    w_deriv: float = 0.00
    b_deriv: float = 0.00
    
    for i in range(len(series_x)):

            w_deriv += (-2 * series_x[i] * (series_y[i] - (weight * series_x[i] + bias)) / len(series_x))
            
            b_deriv += (-2 * (series_y[i] - (weight * series_x[i] + bias))) / len(series_x)
    
    weight -= w_deriv * l_rate
    
    bias -= b_deriv * l_rate
        
    return weight,bias

def regression(df: pd.DataFrame, cat1: str, cat2: str) -> None:
    
    l_rate: int = 0.001
    epochs: int = 3000
    weight: float = 0.0
    bias: float = 0.0

    for i in range(epochs):
        
        weight, bias = gradient_descent(weight,bias,l_rate,df[cat1],df[cat2])
    
    print(f'Weight: {weight} \nBias: {bias}')
        
    return weight, bias
    
def main(df: pd.DataFrame, category_1: str, category_2: str) -> None:
    
    if validate_categories(df, category_1, category_2):
        
        if (get_category_type(df, category_1, category_2, 'and')): 
            
            get_stats(df, category_1,category_2)
            
            get_outliers(df, category_1)
            
            plot_graphs(df, category_1, category_2, 0, True)
            
            
        else: 
            
            plot_graphs(df, category_1, category_2, 45, False)
            
            if get_category_type(df, category_1, category_2, 'or'):
                
                raise NotImplementedError('Unfortunately, categorical vs numerical data processing has not been implemented yet')
            
            else:
    
                raise NotImplementedError('Unfortunately, categorical vs categorical data processing has not been implemented yet')
                
    else:
        
        search_categories(df, category_1, category_2)
        


if __name__ == "__main__":

    df,category_1, category_2 = manual_assignment()
    
    df = pd.read_csv('data_1.csv')
    
    category_1, category_2 = 'MAKE', 'FUEL_CONSUMPTION'

    print(df['FUEL_CONSUMPTION'].astype(str).dtype)
    
    main(df, category_1, category_2)
    
    