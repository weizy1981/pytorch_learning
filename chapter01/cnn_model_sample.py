import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

lr = 1e-3
batch_size = 100
epochs = 50

# 张量转换函数
trans_img = transforms.Compose([
    transforms.ToTensor()
])

trainset = MNIST('./data', train=True, transform=trans_img, download=True)
testset = MNIST('./data', train=False, transform=trans_img, download=True)

# 将输入数据集转换成张量
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1)

# 构建模型
class Lenet(nn.Module):

    def __init__(self):
        super(Lenet, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module(name='layer1', module=nn.Conv2d(1, 6, 3, stride=1, padding=1))
        self.conv.add_module(name='layer2', module=nn.MaxPool2d(2, 2))
        self.conv.add_module(name='layer3', module=nn.Conv2d(6, 16, 5, stride=1, padding=0))
        self.conv.add_module(name='layer4', module=nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential()
        self.fc.add_module(name='layer5', module=nn.Linear(5 * 16 * 5, 120))
        self.fc.add_module(name='layer6', module=nn.Linear(120, 84))
        self.fc.add_module(name='layer7', module=nn.Linear(84, 10))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

lenet = Lenet()
if torch.cuda.is_available():
    # 如果GPU
    lenet.cuda()
else:
    # 如果CPU
    lenet.cpu()


# 定义损失函数
criterion = nn.CrossEntropyLoss(size_average=False)
# 定义优化器（梯度下降）
optimizer = optim.SGD(lenet.parameters(), lr=lr)

# 训练模型
for i in range(epochs):
    running_loss = 0.0
    running_acc = 0.0
    for (img, label) in trainloader:
        if torch.cuda.is_available():
            # 如果使用GPU
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            # 如果使用CPU
            img = Variable(img).cpu()
            label = Variable(label).cpu()

        # 归零操作
        optimizer.zero_grad()
        
        output = lenet(img)
        loss = criterion(output, label)
        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        _, predict = torch.max(output, 1)
        correct_num = (predict == label).sum()
        running_acc += correct_num.data[0]

    running_loss /= len(trainset)
    running_acc /= len(trainset)
    print('[%d/%d] Loss: %.5f, Acc: %.2f' %(i + 1, epochs, running_loss, running_acc * 100) )

# 评估模型
lenet.eval()
testloss = 0.0
testacc = 0.1
for (img, label) in testloader:
    if torch.cuda.is_available():
        # 如果使用GPU
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        # 如果使用CPU
        img = Variable(img).cpu()
        label = Variable(label).cpu()

    output = lenet(img)
    loss = criterion(output, label)
    _, predict = torch.max(output, 1)
    correct_num = (predict == label).sum()
    testacc += correct_num.data[0]

testloss /= len(testset)
testacc /= len(testset)
print('Loss: %.5f, Acc: %.2f' %(testloss, testacc * 100) )