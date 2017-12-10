from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

train_labels = 'train_labels.csv'
test_labels = 'test_labels.csv'
img_folder = 'train'
batch_size = 64
epochs = 200
lr = 0.01

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.relu(out, inplace=True)

class Inception(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs)

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, label_file, img_folder, transform=None, target_transform=None, loader=default_loader):
        imgs = []

        with open(label_file, 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split(',')
                img = img_folder + '/' + words[0]
                label = int(words[1])
                imgs.append((img, label))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':

    # 像素为3x2304x3456
    train_dataset = MyDataset(label_file=train_labels, img_folder=img_folder, transform=transforms.ToTensor())
    test_dataset = MyDataset(label_file=train_labels, img_folder=img_folder, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    model = Inception(in_channels=3, pool_features=5)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss(size_average=False)
    # 定义优化器（梯度下降）
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 训练模型
    for i in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        for (img, label) in train_loader:
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

            output = model(img)
            loss = criterion(output, label)
            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            _, predict = torch.max(output, 1)
            correct_num = (predict == label).sum()
            running_acc += correct_num.data[0]

        running_loss /= len(train_dataset)
        running_acc /= len(train_dataset)
        print('[%d/%d] Loss: %.5f, Acc: %.2f' % (i + 1, epochs, running_loss, running_acc * 100))

    # 评估模型
    model.eval()
    testloss = 0.0
    testacc = 0.1
    for (img, label) in test_loader:
        if torch.cuda.is_available():
            # 如果使用GPU
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            # 如果使用CPU
            img = Variable(img).cpu()
            label = Variable(label).cpu()

        output = model(img)
        loss = criterion(output, label)
        _, predict = torch.max(output, 1)
        correct_num = (predict == label).sum()
        testacc += correct_num.data[0]

    testloss /= len(test_dataset)
    testacc /= len(test_dataset)
    print('Loss: %.5f, Acc: %.2f' % (testloss, testacc * 100))