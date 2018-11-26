import torch
print(1)

x = torch.Tensor(5)
print(x.size())

a = torch.Tensor(13, 13).zero_()

# 1 column 0 row
# start
# length

#fill all with 1
a. narrow (1 , 0, 13) . fill_ (1.0)
#fill 2nd, 6th and 11th column with 2
a. narrow (1 , 1, 1) . fill_ (2.0)
a. narrow (1 , 6, 1) . fill_ (2.0)
a. narrow (1 , 11, 1) . fill_ (2.0)
#fill 2nd, 6th and 11th row with 2
a. narrow (0 , 1, 1) . fill_ (2.0)
a. narrow (0 , 6, 1) . fill_ (2.0)
a. narrow (0 , 11, 1) . fill_ (2.0)

#fill 3 используя narrow
a. narrow (0 , 3, 2) . narrow (1 , 3, 2). fill_ (3.0)
a. narrow (0 , 8, 2) . narrow (1 , 8, 2). fill_ (3.0)
a. narrow (0 , 3, 2) . narrow (1 , 8, 2). fill_ (3.0)
a. narrow (0 , 8, 2) . narrow (1 , 3, 2). fill_ (3.0)


# fill 5 используя slicing
a[3:5, 3:5].fill_(5)
a[8:10, 8:10].fill_(5)
a[8:10, 3:5].fill_(5)
a[3:5, 8:10].fill_(5)

print(a)

#разложение на собственные вектора
# 2 Eigendecomposition
# Without using python loops, create a square matrix M (a 2d tensor) of dimension 20 × 20, filled with
# random Gaussian coefficients, and compute the eigenvalues of M−1 diag(1, . . . , 20) M.
# Hint: Use torch.arange , torch.diag , torch.mm , torch.inverse , torch.normal and torch.eig .

#create a square matrix M (a 2d tensor) of dimension 20 × 20, filled with random Gaussian coefficients
m = torch.Tensor(20, 20).normal_()
m1 = torch.inverse(m)

#M−1 diag(1, . . . , 20) M
d = torch.diag(torch.arange(1,21,1)).float()

#multiply
step1 = torch.mm(m1,d)
step2 = torch.mm(step1,m).eig()


#b. narrow (1 , 0, 20) . fill_ (1.0)
#torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))

#print(b)



# Generate two square matrices of dimension 5000 × 5000 filled with random Gaussian coefficients,
# compute their product, measure the time it takes, and estimate how many floating point products
# have been executed per second (should be in the billions or tens of billions).
# Hint: Use torch.normal , torch.mm , and time.perf counter .

import time

start = time.perf_counter()
z = torch.Tensor(5000, 5000).normal_()
s = torch.Tensor(5000, 5000).normal_()
p = torch.mm(z,s)
end = time.perf_counter()
print(end-start)


