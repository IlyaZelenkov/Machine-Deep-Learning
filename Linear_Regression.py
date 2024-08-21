import math
import decimal
from decimal import Decimal


# Optimization: see if the 
class Regression_Model: 

    # Prediction Variables: Assigned for faster access. 
    # Learning rate = 0.01 by default. 
    # Steps = 10 by default. 
    weight= 1.0
    bias= 1.0
    learn_rate= 0.00001
    num_steps = 100
    training_data = []

    # Constructor: 
    def __init__(self, data) -> None:
        self.training_data = data
        

    # Returns a y value corresponding to the models weight and bias parameters.
    # Behaves on a y*=mx+b linear regression function. 
    def linear_prediction(self, x):        
        
        return (self.weight * x + self.bias)
    
    # Calculates average squared cost of the function.
    # Behaves on a mean squared error (MSE) function.
    # MSE = ( (y - y')^2  / N )
    def cost_function(self, x, y_actual):

        error_cumulative = 0

        for i in range(len(x)):
            
            decimal.getcontext().prec = 100
            error_cumulative += ((( Decimal(y_actual[i]) - Decimal(self.linear_prediction(x[i])))**2))
            # error_cumulative += ((( Decimal(y_actual[i]) - (Decimal(self.linear_prediction(x[i])))**2)))

        return error_cumulative / len(x)

    # Calculates the minimum point of the derivative with respect to the models lowest cost function output. 
    # Behaves on the lowest MSE output.  
    def gradient_descent(self) -> None:

        for i in range(self.num_steps): 

            self.tune_weights()

    # Tunes the weight with respect to minimal cost outputs.
    # Behaves on minimal partial derivatives of the MSE values. 
    def tune_weights(self, x, y_actual):

        weight_derivative, bias_derivative = 0, 0    

        for i in range(len(x)):
            weight_derivative += (-2 * x[i] * (y_actual[i] - self.linear_prediction(x[i])))
            print(f'weight derivative: {weight_derivative}')
            bias_derivative += (-2 * (y_actual[i] - self.linear_prediction(x[i])))
            print(f'bias derivative: {bias_derivative}')
        
        self.weight -= (weight_derivative) * self.learn_rate
        self.bias -= (bias_derivative) * self.learn_rate
        
        print(self.bias, self.weight)
        return self.weight, self.bias        

    def train(self, data):
        cost_db = []

        for i in range(self.num_steps):
            self.weight, self.bias = self.tune_weights(data[0], data[1])   
            cost = self.cost_function(data[0], data[1])
            cost_db.append(cost)

            if i % 10 == 0: 
                print("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2f}".format(i, self.weight, self.bias, cost)) 
 

    def set_learning_rate(self, rate) -> None:

        self.learn_rate = rate

        print(f'Learning rate set to: {self.learn_rate}')

    def set_steps(self, steps) -> None: 

        self.num_steps = steps

        print(f'Step sequence set to: {self.num_steps}')


def main(training_data, prediction_data) -> None:

    model_1 = Regression_Model(training_data)

    model_1.train(prediction_data)


if __name__ == "__main__":

    training_data = [[
                    65.78,
                    71.52,
                    69.40,
                    68.22,
                    67.78,
                    68.69,
                    69.80,
                    70.02,
                    67.90,
                    66.78,
                  ],
                  [
                    112.99,
                    136.49,
                    153.03,
                    142.34,
                    144.30,
                    123.30,
                    141.50,
                    136.46,
                    112.37,
                    120.67,                      
                  ]]

    prediction_data = [[
                        66.49,
                        67.62,
                        68.30,
                        67.12,
                        68.28,
                        71.09,
                        66.46,
                        68.65,
                        71.23,
                        67.13,
                    ],
                    [
                        127.45,
                        114.14,
                        125.61,
                        122.46,
                        116.09,
                        140.00,
                        129.50,
                        142.97,
                        137.90,
                        124.05                      
                    ]]


    print(training_data[1])

    main(training_data, prediction_data)
