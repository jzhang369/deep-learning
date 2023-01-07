# Bookmark

[book](https://www.learnpytorch.io/)
[video](https://www.youtube.com/watch?v=Z_ikDlimN6A&list=RDCMUCr8O8l5cCX85Oem1d18EezQ&start_radio=1&rv=Z_ikDlimN6A&t=4121)

12/14/2022 - 20:00:00

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

# Activitation Functions

Activitation functions are used for two purposes when a nn is built. 

* Activation Functions for Hidden Layers: To make the model non-linear. 
* Activation Functions for Output Layers: To make the output suitable for binary classification and multi-class classification. 

## Non-linearity

This obvious since a linear model is weak on classifying data with complex patterns. So you can just add activitation functions in the hidden layer(s).

## binary/multi-class classification

This is interesting since you will typically use

+ ```sigmoid``` for binary classification. 
+ ```softmax``` for multi-class classification. 

But let us get into a little bit more details. 

The major purpose of an activation function in this context is **to model the loss**. This function is not necessary to classify one object after the model is trained. In other words, an activation function is a must for the loss function, for both binary classification and multi-class classification. If a loss function does not have the activiation function, you will need to put the activation function into your classification model. 

Well, I should say that you will still need the ```sigmoid``` function for binary classification since it will map the single output into $(0,1)$. But for multi-class classification, you can only pick up the dimension that gives you the largest value. 

That is why you have to be very careful when you use PyTorch to build a classification model. Specifically, **before you add an activation function to your model, you need to first check whether the loss function already has a built-in activation function**. 

### binary classification model

+ If the loss function does not have ```sigmoid``` integrated, you can add ```sigmoid``` as another layer in the model. Later, you can directly use the model output. 
+ If the loss function has the ```sigmoid``` integrated, you should not add ```sigmoid``` as another layer in the model. You train the model, but the model output should be processed by a ```sigmoid``` function that helps about the final classification. 

### multiclass classification model

+ If the loss function does not have ```softmax``` integrated, you can add ```softmax``` as another layer in the model. Later, you can directly use the model output. 
+ If the loss function has the ```softmax``` integrated, you should not add ```softmax``` as another layer in the model. You train the model, but the model output can be directly used to decide class (i.e., by picking up the dimension with the largest value). 



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


## Build a Neural Network for Linear Regression


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

## Train a Model

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

## Save and Load a Model

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


# Build a Model for Binary Classification

```python
# Create a Model
model = nn.Sequential(
    nn.Linear(in_features = 3, out_features = 100),
    nn.Linear(in_features = 100, out_features = 100),
    nn.ReLu(),
    nn.Linear(in_features = 100, out_features =3)
)

# Setup a loss function
loss_fn = nn.BCEWithLogitsLoss() # BCE is for binary crossentropy

# Setup an optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.001)

```


```python
# to generate some data
import sklearn
import pandas as pd
from sklearn.datasets import make_circles
n_samples = 1000
X, y = make_circles(n_samples, noise = 0.3, random_state = 42)

circles = pd.DataFrame("X1" : X[:, 0],
                        "X2" : X[:, 1], 
                        "label" : y)

# to visualize the data.
import matplotlib.pyplot as plt
plt.scatter(x=X[:, 0], y = X[:, 1], c=y, cmap = plt.cm.RdYlBu)

# to convert the data into tensors
# Now X is the numpy arrary with type float64
type(X)
X.dtype
# When we convert the data into torch censors, we also converet the type into float32. If we do not do this, there might be problems later.
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# split the data into training and testing data sets.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2,
                                                    random_state=42)
#20% samples for testing. 


# Build a model to classify
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))


model_0 = CircleModelV0().to(device)

# or you can do

model_0 = nn.Sequential(
                        nn.Linear(in_feautres = 2, out_features = 5),
                        nn.Linear(in_features = 5, out_features = 1)
                        ).to(device)


# make predictions
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))

# setup loss function and optimizer
# loss function: binary cross entropy or categorical cross entropy

# loss_fn = nn.BCELoss() #in this case, you will need to require inputs to have gone through the sigmoid activation function priori to input to BCELoss. i.e., you can put sigmoid as another layer in the model. 

loss_fn = nn.BCEWithLogitLoss() 
#This has the sigmoid function built-in. 
#This is why in the model, we will not need the layer of sigmoid. 
#But later you will need to convert the output into sigmoid for classification. 
#For example, you will need to convert the output using sigmoid as follows:
#
#with torch.inference_mode():
#    y_logits = model_0(X+test.to(device))
#y_pred_probs = torch.sigmoid(y_logits)
#y_pred_labels = torch.round(y_pred_probs) #y_pred_probs >= 0.5, y =1, else, y = 0.


optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.1)

```

```python
# Notice: This is a flawed model since it is completely linear!!
# Fit the model
torch.manual_seed(42)
epochs = 1000

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_3.train()
    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
    
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits; note here the nput is y_logits, not y_pred. BCEWithLogitsLoss() has the sigmoid built-in so its first input is y_logits rather than y_pred. y_pred is only used for calculating the accuracy (see accuracy_fn) and data display by the developer. 
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_3.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_3(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
      # 2. Calcuate loss and accuracy
      test_loss = loss_fn(test_logits, y_test)
      test_acc = accuracy_fn(y_true=y_test,
                             y_pred=test_pred)

    # Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
```


```python
# this is the model with non-linearity added. 
# Question: why relu is not added for the last layer?

# Build model with non-linear activation function
from torch import nn
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # <- add in ReLU activation function
        # Can also put sigmoid in the model 
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2().to(device)
print(model_3)
```
:question: why relu is not added for the last layer? Perhaps activation functions are used for hidden layers for non-linearity? 


# Build a Model for Multi-Class Classification

prepare the dataset. 

```python
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# set hyperparameters for data creation.
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples = 1000, 
                            n_features = NUM_FEATURES,
                            centers=NUM_CLASSES, 
                            cluster_std = 1.5, 
                            random_state=RANDOM_SEED)


# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
# Note: the data type!!!


# 3. Split data into training and testing
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, test_size = 0.2, random_seed = RANDOM_SEED)


# 4. Plot data
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)

```

build a multi-class classification model in PyTorch

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

class BlobModel(nn.Module):
    def __init_(self, input_features, output_features, hidden_units = 8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features = input_features, out_features = hidden_units), 
            #nn.ReLU() # Do you need it?
            nn.Linear(in_features = hidden_units, out_features = hidden_units),
            #nn.ReLU() # Do you need it?
            nn.Linear(in_features = hidden_units, out_features = output_features)
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)


model_4 = BlobModel(input_features = 2, output_features = 4, hidden_units=8).to(device)

# Create a loss function.
loss_fn = nn.CrossEntropyLoss()

# Create an optimizer for multi-class classification
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr = 0.1)


# Create the training and testing loop

# Fit the model
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
epochs = 100

# Put data to target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    ### Training
    model_4.train()

    # 1. Forward pass
    y_logits = model_4(X_blob_train) # model outputs raw logits 
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # go from logits -> prediction probabilities -> prediction labels
    # print(y_logits)
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_blob_train) 
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_4.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_4(X_blob_test)
      test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
      # 2. Calculate test loss and accuracy
      test_loss = loss_fn(test_logits, y_blob_test)
      test_acc = accuracy_fn(y_true=y_blob_test,
                             y_pred=test_pred)

    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%") 
```

# Build a Model for Computer Vision

(The CNN part was not noted here.)

## Inputs and Outputs Shapes

input: 
+ NHWC - [batch_size, height, width, colour_channels]
+ NCHW - [batch_size, colour_channels, height, width]

Some packages:

+ ```torchvision``` is the base domain library for computer vision in PyTorch. 
+ ```torchvision.datasets``` prepares data
+ ```torchvision.transforms``` functions for+  manipulating your vision data to be suitable for use with an ML model
+ ```torch.utils.data.Dataset``` base dataset class for PyTorch
+ ```torch.utils.data.DataLoader``` creats a python iterable over a dataset


Turn the data into iterables (batches)
+ training data should be turned into batches and shuffled.
+ testing data should be turned into batches but not shuffled. 


## Create a timer to time out the experiments

```python
from timeit import default_timer as timer
def print_train_time(start: float,
                        end: float,
                        device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

start_time = timer()
# Your code comes here
end_time = timer()
print_train_time(start=start_time, end=end_time, device="cpu")
```
## Using Progress Bar

```python
from tqdm.autom import tqdm
```

# Customer Datasets
