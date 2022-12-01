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