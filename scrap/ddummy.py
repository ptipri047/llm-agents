import numpy as np

a = np.random.rand(2,3,4)
print(a)

# 0,1,2
b = a[0,1:3,:]

print(b)

s = a.shape
print(type(s))
print(s)
print(s[2])

#or
dim1, dim2, dim3 = a.shape
print(dim2)

z= [[1,2,3],[3,4,5]]
print(f'{type(a)} - {type(z)}')
#s = z.shape
print(len(z))