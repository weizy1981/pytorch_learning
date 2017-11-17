from torch import nn


class MyNetwork(nn.Module):
    def __init__(self, otherparams):
        super(MyNetwork, self).__init__()
        # 定义自己的神经网络层
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)

    def forward(self, x):
        # 定义向前传播
        out = self.conv(x)
        return out

# 定义损失函数
criterian = nn.CrossEntropyLoss(size_average=False)

#执行损失函数，并进行反向传播
loss = criterian(output, target)
loss.backward()

from torch import optim
myNetword = MyNetwork()
# 定义优化器
sgd = optim.SGD(myNetword.parameters(), lr=0.1, momentum=0.9)

# 优化器清零
sgd.zero_grad()

# 更新梯度值
sgd.step()