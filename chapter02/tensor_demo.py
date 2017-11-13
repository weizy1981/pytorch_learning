import torch

# Pytorch中默认张量为FloatTensor
a = torch.Tensor([[2, 3], [4, 8], [7, 9]])
print('a is: {}'.format(a))
print('a size is {}'.format(a.size()))

# 定义LongTensor张量
b = torch.LongTensor([[2, 3], [4, 8], [7, 9]])
print('b is: {}'.format(b))

# 定义全为0的空张量
c = torch.zeros((3, 2))
print('c is: {}'.format(c))

# 定义随机分配的张量
d = torch.rand((3, 2))
print('d is: {}'.format(d))

# 修改张量的值
a[0, 1] = 100
print('new a is: {}'.format(a))

# 张量转化为numpy
b_np = b.numpy()
print('conver b to numpy is: \n{}'.format(b_np))

import numpy as np
# 从numpy转换成张量
e = np.array([[2, 3], [4, 7]])
e_tensor = torch.from_numpy(e)
print('from numpy e to tensor is: {}'.format(e_tensor))

# 改变张量的类型
e_tensor_float = e_tensor.float()
print('convert e_tensor to float tensor is: {}'.format(e_tensor_float))

if torch.cuda.is_available():
    # 将张量放到GPU上
    e_cuda = e_tensor_float.cuda()
    print(e_cuda)
else:
    # 将张量放到CPU上
    e_cpu = e_tensor_float.cup()
    print(e_cpu)