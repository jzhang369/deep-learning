# Bookmark

[book](https://www.learnpytorch.io/)
[video](https://www.youtube.com/watch?v=Z_ikDlimN6A&list=RDCMUCr8O8l5cCX85Oem1d18EezQ&start_radio=1&rv=Z_ikDlimN6A&t=4121)

12/14/2022 - 7:49:00

# Objective Functions

Objective Functions -> Loss Functions 

+ Predict numeric values -> squared error -> differentiability -> easy to optimize 
+ Classification -> error rate -> non-differentiability -> difficult to optimize

Learning Process: 

+ During optimization, we think of the loss as a function of the model’s parameters, and treat the training dataset as a constant. 
+ We learn the best values of our model’s parameters by minimizing the loss incurred on a set consisting of some number of examples collected for training. 
+ Use test data to prevent overfitting. 

# Optimization Algorithms

During optimization, we think of the loss as a function of the model’s parameters, and treat the training dataset as a constant.

+ The objective is to find $\theta_i$ that will minimize the objective function $J$. 
+ In machine learning, $J$ here is for the loss function. 
  
Gradient descent is used to incrementally, computationally identify $\theta_i$. 

![](https://miro.medium.com/max/1400/1*GTRi-Y2doXbbrc4lGJYH-w.png)


# Tensor

A **tensor** is a a n-dimensional array. 



+ A single number is a scalar, which is a 0-dimensional tensor.
+ A vector is a 1-dimensional tensor. 
+ A matrix is a 2-dimensional tensor. 
+ A 3-dimensional matrix is a 3-dimensional tensor.  

Implementation with different libraries. 

+ NumPy: ndarray, CPU only
+ MXNet ndarray, CPU + GPU
+ PyTorch: tensor, CPU + GPU
+ TensorFlow: tensor, CPU + GPU

## Initializating Tensors

```python
shape=(4, 3, 2)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")
```

This program creates three 3D tensors, where each tensor is a 4*3*2 matrix. For interpretation, it is easier, at least for me, to understand the matrix from the lowest dimension (i.e., from 2, to 3, and to 4, meaning 2 columns, 3 rows, and 4 layers). 

It is worth noting that a vector is not considered as an $1\times m$ matrix in Pytorch. See the following snippet. And if you have ```a.T```, you will get a complain but ```b.T``` will be fine. In addition, if you slice a matrix to one slice (e.g., ```[:,1]```), it will give you a vector; but if you slice it into a submatrix (e.g., ```[:,1:3]```), it will give you a matrix. Weird :confused:

:zap: Remember to give a try on ```squeeze``` and ```unsqueeze```. It will remove and add a dimension. 

```python
import torch
a = torch.arange(6)
print(a)
print(a.ndim)
print(a.shape)

b = a.reshape(2,3).reshape(1, -1)
print(b)
print(b.ndim)
print(b.shape)

#output would be
#tensor([0, 1, 2, 3, 4, 5])
#1
#torch.Size([6])
#tensor([[0, 1, 2, 3, 4, 5]])
#2
#torch.Size([1, 6])
```

```dtype``` is the data type of a tensor. Here is a [list](https://pytorch.org/docs/stable/tensors.html) of torch data types. Data types are relevant to precision and computing. 

    `The first thing to create a tensor is to decide its type.`

    `Tensor datatypes is one of the 3 big things with PyTorch and deep learning.`

:question: Not sure what it means but hopefully get it addressed later. 

# Attributes of A Tensor

```python
tensor = torch.rand(3,4)
# or tensor = torch.rand((3,4))
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

# Operations on Tensors

Indexing and Slicing. 

```python
tensor = torch.rand(3, 2)
print(tensor)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}") # it will do automatically transpose. 
print(f"Last column: {tensor[..., -1]}") # it will do automatically transpose. ... is the same to :
tensor[:,1] = 0
print(tensor)

print(tensor.T) # matrix transpose
```

# Reproducbility

PyTorch has a concept of a **random seed**. This is the same to the seed used for a pseudo random number generator. Essentially if you give the same seed, the random number generator will generate the same sequence of random numbers. 

```python
seed = 42
torch.manual_seed(seed)
random_tensor_A = torch.rand(3,4)

torch.manual_seed(seed)
random_tensor_B = torch.rand(3,4)

print(random_tensor_A == random_tensor_B)
# you will get True. 
```


# GPU with CoLab

Check [Section of Best Practices](https://pytorch.org/docs/stable/notes/cuda.html#best-practices)


```bash
!nvidia-smi
```
```python
import torch
# Check for GPU access with PyTorch
torch.cuda.is_available()

# 1. Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Count number of devices
torch.cuda.device_count

# 3. Put tensors and models on the GPU
tensor = torch.tensor([1,2,3])
# this tensor is not on GPU
print(tensor, tensor.device)
# move tensor to GPU if GPU is available
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)


# if tensor is on GPU, we cannot transfor it to NumPy
# tensor_on_device.numpy() will fail if it is on GPU.
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_on_cpu)
```




# Data Preprocessing

Import and load CSV data

```python
import pandas as pd
import torch

data = pd.read_csv(data_file)
print(data)

inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]

print(inputs)


#transfer the pandas data into pytorch tensor's formats
X, y = torch.tensor(inputs.values), torch.tensor(targets.values)
print(X)
print(y)


```


# PyTorch Workflow

```python
import torch
from torch import nn
# nn has all building blocks for neural networks in PyTorch. 

import matplotlib.pyplot as plt
#this is for data visualization.

```

## Data Preparing and Loading

ML has two parts:
+ Map data into a numerical representation. 
+ Build a model to learn patterns from numerical representation. 

Split data into training and test sets. 


## Build a Neural Network


PyTorch model building essentials

* torch.nn - all buildings for neural networks. 
* torch.nn.Parameter - what parameters should the model try and learn. Could be set through torch.nn layers. 
* torch.nn.Module - The base class for all neural network modules, if you subclass it, you should overwrite ```forward()```
* torch.optim - where the optimizers in PyTorch live. They will help with gradient descent. 
* ```def forward()``` - All nn.Module subclasses require you to overwrite ```forward()```, this method defines what happens in the forward computation. 


```python
from torch import nn
class LinearRegressionModel(nn.Module): 
    #almost everything in PyTorch inherhits from nn.Module, which is the base class for all neural networks. 
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=true, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=true, dtype=torch.float))

    def froward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

```


```python
# Create a random seed
torch.manual_seed(369)

# Initialize an instance of the model
model = LinearRegressionModel()

# List the internal parameters
list(model.parameters())
# or you can use
model.state_dict()
```


```python
# Make predictions without training the model so that you can see the initial paramter values do not work. 
# When the data runs through the model, it runs through the forward() function.

# Runs this directly. 
y_preds = model(X_test)
y_preds

# This is a better practice
# Using torch.inference_mode() to improve the performance. 
with torch.inference_mode():
    y_preds = model(X_test)
y_preds
```

**Now it is ready to train a model**

One way to measure how well your model/parameters perform is to use a loss function. 

* loss function == cost function == criterion function.
* A loss function measures how wrong your prediction is against the ground truth data. 
* Optimizer: Adjust the parameters so that the loss function will yield minimal values.  



```python
# Now to train the model.

# Setup a loss function
loss_fn = nn.L1Loss() # Using L1Loss and but there are other choices. 

# Setup an optimizer
# Inside the optimizer, you will often have to set two parameters. 
#  + params - the model parameters you want to optimize. 
#  + lr (learning rate)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01) # lr is for the learning rate. 

```

You will also need

* a training loop
* a testing loop

A couple of things:

+ 0. Loop through the data
+ 1. Forward pass to make predictions
+ 2. Calculate the loss
+ 3. Optimizer zero grad (:confused: what is this?)
+ 4. Loss backpropagation
+ 5. Step Optimizer (Gradient descent) to adjust the parameters. 

```python
epochs = 1

# 0. loop through the data
for epoch in range(epochs):
    # set the model to the training mode
    model.train() # turn on gradient tracking: train mode in PyTorch set all parameters that need to be gradients to be gradients

    # 1. forward pass
    y_pred = model(X_train)

    # 2. calculate the loss
    loss = loss_fn(y_pred, y_train)

    print(f"Loss: {loss}")

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Step the optimizer (to perform gradient descent)
    optimizer.step()


    print(model.state_dict()) # display the updated parameters for each iteration


    # Testing, this is not part for the trainning. 
    model.eval() # turn off gradient tracking, this is the opposite to model.train()
    with torch.inference_mode():
        # 1. Do the forward pass
        test_pred = model(X_test)
        # 2. Calculate the loss on the testing data
        test_loss = loss_fn(test_pred, y_test)

    print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

```

# Save and Load a Model

There are three main methods for saving and loading models in PyTorch. 

1. ```torch.save()``` - save a PyTorch object in Python's *pickle* format. 
2. ```torch.load()``` - load a saved PyTorch object. 
3. ```torch.nn.Module.load_state_dict()``` - load a model's saved state dictionary. 


```python
# Saving our PyTorch model
from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True, exist_ok = True) 

# 2. Create model save path
MODEL_NAME = "01_first.pth" # PyTorch usually has pth or pt as the file extension for its saved models.
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state_dict
torch.save(obj = model.state_dict(), f = MODEL_SAVE_PATH)
```

Loading a PyTorch model. 

Since for the previous example, we only saved state_dict rather than the entire model, we will need to create a new instance of our model class and load state_dict() into that. 

```python
# Instantiate a new instance of our model class
loaded_model = LinearRegressionModel()

loaded_model.state_dict() # this will display the default parameters. 

# Load the saved state_dict
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model.state_dict() # this will display the loaded parameters.
```