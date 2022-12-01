# Objective Functions

Objective Functions -> Loss Functions 

+ Predict numeric values -> squared error -> differentiability -> easy to optimize 
+ Classification -> error rate -> non-differentiability -> difficult to optimize

Learning Process: 

+ During optimization, we think of the loss as a function of the model’s parameters, and treat the training dataset as a constant. 

+ We learn the best values of our model’s parameters by minimizing the loss incurred on a set consisting of some number of examples collected for training. 
+ Use test data to prevent overfitting. 

# Optimization Algorithms

Gradient descent. 

