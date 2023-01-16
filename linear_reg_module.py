import numpy as np
class Linear_Regression():

  def __init__(self, learning_rate, no_of_iterations):
    
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations

  def fit(self, x, y):
    self.m, self.n = x.shape # number of rows & columns

    self.w = np.zeros(self.n)
    self.b = 0
    self.x = x
    self.y = y

    # implementing Gradient descent
    for i in range(self.no_of_iterations):
      self.update_weights()
  
  def update_weights(self, ):

    y_prediction = self.predict(self.x)

    dw = - (2*(self.x.T).dot(self.y - y_prediction)) / self.m

    db =  -(2 * np.sum(self.y - y_prediction)) / self.m
    
    # Updating the weight
    self.w = self.w - self.learning_rate*dw
    self.b = self.b - self.learning_rate*db


  def predict(self, x):

    return x.dot(self.w) + self.b
