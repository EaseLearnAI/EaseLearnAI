# 内容:
# 开发日期 :2024/11/10
import torch
import swanlab
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# 设置随机种子，保证每次运行结果相同
torch.manual_seed(1)
# 超参数设置
EPOCH = 2  # 训练的轮数
BATCH_SIZE = 64  # 每个batch的样本数量
LR = 0.001  # 学习率
DOWNLOAD_MNIST = True  # 如果没有下载MNIST数据集，就下载


# 初始化 SwanLab 实验
swanlab.init(
    project="MNIST-classification",  # 设置项目名称
    config={
        "learning_rate": LR,  # 学习率
        "epochs": EPOCH,  # 训练轮数
        "batch_size": BATCH_SIZE,  # 批次大小
        "architecture": "CNN",  # 模型结构
        "dataset": "MNIST"  # 数据集
    }
)

# 下载 MNIST 手写数字数据集
train_data = torchvision.datasets.MNIST(
    root='./data/',  # 数据集保存路径
    train=True,  # 训练集
    transform=torchvision.transforms.ToTensor(),  # 将图片转换为 Tensor 格式
    download=DOWNLOAD_MNIST,  # 是否下载
)

# 加载训练数据集
train_loader = Data.DataLoader(
    dataset=train_data,  # 数据集
    batch_size=BATCH_SIZE,  # 每批次样本数
    shuffle=True  # 是否打乱数据
)

# 加载测试数据集
test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False,  # 测试集
    transform=torchvision.transforms.ToTensor(),  # 将图片转换为 Tensor 格式
)

# 测试数据处理，取出图片和标签
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255  # 测试前2000个数据，归一化
test_y = test_data.targets[:2000]  # 前2000个标签


# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层：输入为1通道，输出为16通道，卷积核大小为5x5，padding保持输入输出大小一致
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),  # (1, 28, 28) -> (16, 28, 28)
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2),  # 池化层，降低维度 (16, 28, 28) -> (16, 14, 14)
        )
        # 第二个卷积层：输入为16通道，输出为32通道
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # (16, 14, 14) -> (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 14, 14) -> (32, 7, 7)
        )
        # 全连接层，输入32*7*7的特征图，输出10个分类
        self.out = nn.Linear(32 * 7 * 7, 10)  # 10是MNIST的分类数量

    def forward(self, x):
        x = self.conv1(x)  # 经过第一个卷积层
        x = self.conv2(x)  # 经过第二个卷积层
        x = x.view(x.size(0), -1)  # 展平 (batch_size, 32*7*7)
        output = self.out(x)  # 全连接层输出
        return output


# 实例化 CNN 模型
cnn = CNN()
print(cnn)

# 定义损失函数和优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # Adam 优化器
loss_func = nn.CrossEntropyLoss()  # 损失函数为交叉熵

# 训练模型
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # 遍历每个batch
        output = cnn(b_x)  # 前向传播
        loss = loss_func(output, b_y)  # 计算损失
        optimizer.zero_grad()  # 清除上一步的梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 每50步打印一次结果
        if step % 50 == 0:
            test_output = cnn(test_x)  # 测试集的预测
            pred_y = torch.max(test_output, 1)[1].data.numpy()  # 获取预测结果
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            # print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.data.numpy():.4f}, Test Accuracy: {accuracy:.2f}')

            # 向swanlab上传训练指标
            swanlab.log({
                "epoch": epoch,
                "step": step,
                "loss": float(loss.data.numpy()),  # 将损失值转换为浮点数
                "test_accuracy": float(accuracy)  # 测试准确率转换为浮点数
            })

# 保存模型 吧
torch.save(cnn.state_dict(), 'mnist_cnn.pth')

# 加载并测试模型
cnn.load_state_dict(torch.load('mnist_cnn.pth'))

# 设置模型为评估模式
cnn.eval()

# 从测试集中取32个数据进行预测
test_output = cnn(test_x[:32])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(f'Prediction: {pred_y}')
print(f'Real: {test_y[:32].numpy()}')

# 获取前10个测试样本的预测结果
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()  # 获取每张图片的预测标签

# 绘制前10个测试样本的图像、预测结果和真实标签
fig, axes = plt.subplots(2, 5, figsize=(12, 5))  # 2行5列的图形布局
for i in range(10):
    ax = axes[i // 5, i % 5]  # 定位到2x5网格的每个子图
    ax.imshow(test_data.data[i].numpy(), cmap='gray')  # 显示灰度图像
    ax.set_title(f'预测: {pred_y[i]}\n真实: {test_y[i].item()}')  # 显示预测和真实标签
    ax.axis('off')  # 去除坐标轴

plt.tight_layout()  # 调整布局
plt.show()

# [可选] 完成训练，这在notebook环境中是必要的
swanlab.finish()