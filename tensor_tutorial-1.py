import torch
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"  # Cuda to run on GPU!

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True
)


print(my_tensor) 
print(my_tensor.dtype) 
print(my_tensor.device)
print(my_tensor.shape) 
print(my_tensor.requires_grad)


x = torch.empty(size=(3, 3))
x = torch.zeros((3, 3))  
x = torch.rand( (3, 3) )
x = torch.ones((3, 3)) 
x = torch.eye(5, 5)  
x = torch.arange( start=0, end=5, step=1) 
x = torch.linspace(start=0.1, end=1, steps=10) 
x = torch.empty(size=(1, 5)).normal_(   mean=0, std=1) 
x = torch.empty(size=(1, 5)).uniform_( 0, 1)
x = torch.diag(torch.ones(3))  # Diagonal matrix of shape 3x3

tensor = torch.arange(4)  
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())  # Converted to float16
print(tensor.float())
print(tensor.double())  # Converted to float64

np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_again = ( tensor.numpy())  

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])


z1 = torch.empty(3)
torch.add(x, y, out=z1)
z2 = torch.add(x, y) 
z = x + y 

z = x - y 

z = torch.true_divide(x, y)  # element wise division if of equal shape

# Inplace Operations
t = torch.zeros(3)

t.add_(x) 
t += x  

z = x.pow(2) 
z = x**2 

z = x > 0 
z = x < 0 

#  Matrix Multiplication --
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) 
x3 = x1.mm(x2)  

# -- Matrix Exponentiation --
matrix_exp = torch.rand(5, 5)
print(  matrix_exp.matrix_power(3)) 


z = x * y

z = torch.dot(x, y)  

# -- Batch Matrix Multiplication --
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) 

# broadcasting:
x1 = torch.rand((5, 5))
x2 = torch.ones((1, 5))
z = (   x1 - x2)  # Shape of z is 5x5
z = (  x1**x2) 


sum_x = torch.sum( x, dim=0 ) 
values, indices = torch.max(x, dim=0) 
values, indices = torch.min(x, dim=0) 
abs_x = torch.abs(x)  
z = torch.argmax(x, dim=0) 
z = torch.argmin(x, dim=0) 
mean_x = torch.mean(x.float(), dim=0) 
z = torch.eq(x, y)  # Element wise comparison
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)  
z = torch.any(x) 
z = torch.all( x) 


batch_size = 10
features = 25
x = torch.rand((batch_size, features))


print(x[0].shape)  
print(x[:, 0].shape)
print(x[2, 0:10].shape)  # shape: [10]
x[0, 0] = 100


x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])  

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols]) 


x = torch.arange(10)
print(x[(x < 2) | (x > 8)])  # will be [0, 1, 9]
print(x[x.remainder(2) == 0])  # will be [0, 2, 4, 6, 8]


print( torch.where(x > 5, x, x * 2)) 
x = torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique() 
print( x.ndimension())
x = torch.arange(10)
print(    x.numel() ) 


x = torch.arange(9)

x_3x3 = x.view(3, 3)

x_3x3 = x.reshape(3, 3)

x1 = torch.rand(2, 5)
x2 = torch.rand(2, 5)
print(torch.cat((x1, x2), dim=0).shape)  # Shape: 4x5
print(torch.cat((x1, x2), dim=1).shape)  # Shape 2x10


z = x1.view(-1)  

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(   batch, -1) 

z = x.permute(0, 2, 1)

z = torch.chunk(x, chunks=2, dim=1)
print(z[0].shape)
print(z[1].shape)


x = torch.arange(   10)
print(x.unsqueeze(0).shape)  # 1x10
print(x.unsqueeze(1).shape)  # 10x1


x = torch.arange(10).unsqueeze(0).unsqueeze(1)


z = x.squeeze(1)  # can also do .squeeze(0) both returns 1x10


