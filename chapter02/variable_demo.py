from torch.autograd import Variable
import torch

# 定义变量
x = Variable(torch.zeros(2, 2), requires_grad=True)
# 变量运算
y = x + 2
z = y * y + 2
out = z.mean()

# 自动求导
out.backward()
print(x.grad)