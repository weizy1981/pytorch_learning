from torch import nn
from torch.autograd import Variable
from torch import optim
import torch
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import OneHotEncoder


train_folder = 'data/train'
test_folder = 'data/test'
batch_size = 64
epochs = 200
lr = 0.01

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 5)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 像素为3x2304x3456
    train_dataset = ImageFolder(train_folder, transform=transform)
    test_dataset = ImageFolder(test_folder, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    model = GoogLeNet()

    # 定义损失函数
    criterion = nn.CrossEntropyLoss(size_average=False)
    # 定义优化器（梯度下降）
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 训练模型
    for i in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        for (img, label) in train_loader:
            img = Variable(img)
            label = Variable(label)

            # 归零操作
            optimizer.zero_grad()

            output = model(img)
            loss = criterion(output, label)
            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            _, predict = torch.max(output, 1)
            correct_num = (predict == label).sum()
            running_acc += correct_num.data[0]
            print(running_acc)

        running_loss /= len(train_dataset)
        running_acc /= len(train_dataset)
        print('[%d/%d] Loss: %.5f, Acc: %.2f' % (i + 1, epochs, running_loss, running_acc * 100))

    # 评估模型
    model.eval()
    testloss = 0.0
    testacc = 0.1
    for (img, label) in test_loader:
        # 如果使用CPU
        img = Variable(img)
        label = Variable(label)

        output = model(img)
        loss = criterion(output, label)
        _, predict = torch.max(output, 1)
        correct_num = (predict == label).sum()
        testacc += correct_num.data[0]

    testloss /= len(test_dataset)
    testacc /= len(test_dataset)
    print('Loss: %.5f, Acc: %.2f' % (testloss, testacc * 100))