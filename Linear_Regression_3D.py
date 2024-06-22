import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import process
import statistics as st
import requests

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
            
            return by,category_1,category_2,df1,df2
            
        else: 
            
            return None,None,None,None,None

    except requests.exceptions.RequestException as e:
        
        print(e)
        
        return None,None,None,None,None

def validate_categories(cat1: str, cat2: str, by: str , *args: pd.DataFrame) -> bool:
    
    return {cat1,cat2,by}.issubset(pd.concat(args, ignore_index=True).columns)

def print_fuzz( cat1: str, cat2: str, by: str, df1: pd.DataFrame) -> None:

    cats: list[str] = [cat1,cat2,by]
        
    wrong_category: list[str] = [cats[i] for i in range(len(cats)) if cats[i] not in df1.columns]
    
    match_df: pd.DataFrame = pd.DataFrame(process.extract(wrong_category[0], df1.columns), columns=['word','p_match'])
    
    match_df = match_df.set_index('word')
    
    highest_match: pd.DataFrame = match_df[match_df['p_match'] == match_df['p_match'].max()]
    
    print(f'Unable to locate a category \nDid you mean {highest_match.index.values} instead of {wrong_category[0]}?')
    
 
def return_stats(cat_1, cat_2, data: pd.DataFrame, by: str) -> pd.DataFrame:
    
    print(data.groupby(by)[[cat_1,cat_2]].agg([np.mean,np.median, st.mode, min, max]).head(5) , "\n")
    
    return data.groupby(by)[[cat_1,cat_2]].agg([np.mean,np.median, st.mode, min, max]).head(10)

def gradient_descent(weight: float,bias: float, learning_rate: float, series_x: pd.Series, series_y: pd.Series) -> float:
    
    w_deriv = 0.00
    b_deriv = 0.00
    
    for i in range(len(series_x)):

            w_deriv += (-2 * series_x[i] * (series_y[i] - (weight * series_x[i] + bias)) / len(series_x))
            
            b_deriv += (-2 * (series_y[i] - (weight * series_x[i] + bias))) / len(series_x)
    
    weight -= w_deriv * learning_rate
    
    bias -= b_deriv * learning_rate
        
    return weight,bias
    
def error(weight: float, bias: float, x_series: pd.Series, y_actual: pd.Series) -> float:
    
    return (np.square(y_actual - (weight * x_series + bias))).mean()

def regression(data: pd.DataFrame, cat1, cat2):
    
    weight,bias,epochs,l_rate = 0.0, 0.0, 3000, 0.001
    
    weight_bias_error = []
    
    for i in range(epochs):
        
        weight, bias = gradient_descent(weight,bias,l_rate,data[cat1],data[cat2])
        
        curr_mse = error(weight,bias,data[cat1],data[cat2])
        
        weight_bias_error.append([weight,bias,curr_mse])
    
    mse_per_wb = pd.DataFrame(weight_bias_error, columns=['weight', 'bias', 'mse'])     

    print(f'Regression: Weight: {round(weight, 4)}, Bias: {round(bias, 4)}, MSE: {round(curr_mse,4)} \n')

    return weight, bias, curr_mse, mse_per_wb
    
def plot_gradient_3D(df: pd.DataFrame) -> None:
    
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    z = df.iloc[:,2]
    
    fig = plt.figure(figsize = (10,10))
    ax: plt = plt.axes(projection='3d')
    ax.grid()
    
    ax.scatter(x, y, z, c = 'g', s = 20)
    ax.set_title('( Weight & Bias ) vs MSE')
    
    ax.set_xlabel('Weight', labelpad=20)
    ax.set_ylabel('Bias', labelpad=20)
    ax.set_zlabel('MSE', labelpad=20)

    plt.show()

def plot_categories(df: pd.DataFrame, cat1, cat2, weight, bias) -> None:
            
    df.plot(x=cat1, y=cat2, kind='scatter', title=f'{cat1} vs {cat2}', xlabel=f'{cat1}', ylabel=f'{cat2}')
    
    plt.plot(df[cat1], (df[cat1] * weight + bias), color='r')
    
    plt.show()
    
def predict( weight: float, bias: float, df_pred: pd.DataFrame, cat1: str, cat2: str) -> None:
    
    prediction_mse = error(weight, bias, df_pred[cat1], df_pred[cat2])
    
    print(round(prediction_mse,4))
                    
def main() -> None:

    # by,cat1,cat2,df1,df2 = api_data_fetch()
    
    cat1,cat2,by = 'ENGINE SIZE','FUEL CONSUMPTION','MAKE'
    
    df1, df2= pd.read_csv("data_1.csv"), pd.read_csv("data_2.csv")
    
    # print(f'Available categories: \n{pd.concat([df1,df2], ignore_index=True, join='inner').columns}')

    if validate_categories(cat1,cat2,by,df1,df2):
        
        # Return Statistics 
        df_stats = return_stats(cat1,cat2,df1,by)
        
        # Calculate: weight, bias, mse_by_w_b 
        weight, bias, mse, mse_by_w_b = regression(df1,cat1,cat2)
        
        # Plot the regression line. 
        plot_categories(df1, cat1, cat2, weight, bias)
        
        # Plot the weight & bias vs MSE 
        plot_gradient_3D(mse_by_w_b)
        
        # Predict using predict_data
        predict(weight, bias, df2, cat1, cat2)
        
            
    else:
     
        print_fuzz(cat1,cat2,by,df1,df2)
            
        
        
        
        
        
if __name__ == "__main__":

    main()


