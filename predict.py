import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 屏蔽 TensorFlow 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置 Matplotlib 字体，避免中文乱码
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义模型
def model(x, h1, h2, out):
    x = tf.nn.relu(tf.matmul(x, h1))  # 第一层
    x = tf.nn.relu(tf.matmul(x, h2))  # 第二层
    x = tf.matmul(x, out)             # 输出层
    return x

# 加载 MNIST 数据集
(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
test_images = test_images.reshape(-1, 784).astype("float32") / 255.0  # 归一化
test_labels = test_labels  # 保留为整数形式（真实标签）

# 加载权重
h1 = tf.Variable(tf.random.normal([784, 128], stddev=0.01))
h2 = tf.Variable(tf.random.normal([128, 64], stddev=0.01))
out = tf.Variable(tf.random.normal([64, 10], stddev=0.01))

class CustomModel(tf.keras.Model):
    def __init__(self, h1, h2, out):
        super(CustomModel, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.out = out

    def call(self, x):
        return model(x, self.h1, self.h2, self.out)

custom_model = CustomModel(h1, h2, out)
dummy_input = tf.zeros((1, 784))  # 构建模型
custom_model(dummy_input)
custom_model.load_weights('./mnist_models/model.weights.h5')
print("模型权重已加载！")

# 定义图片展示函数
def display_compare(num):
    """
    展示图片并比较预测值与真实标签是否正确。
    """
    # 提取输入图片
    input_image = test_images[num].reshape(1, 784)  # 重构输入维度为 [1, 784]
    true_label = test_labels[num]  # 获取真实标签（整数形式）

    # 模型预测
    logits = custom_model(input_image)
    probabilities = tf.nn.softmax(logits).numpy()[0]  # 计算概率分布
    predicted_label = tf.argmax(probabilities).numpy()  # 获取预测值

    # 输出预测概率分布
    print(f"预测概率分布：{probabilities}")
    print(f"预测值: {predicted_label}, 标签: {true_label}")

    # 判断预测是否正确
    is_correct = "预测正确！" if predicted_label == true_label else "预测错误！"

    # 显示图片
    plt.imshow(input_image.reshape(28, 28), cmap='gray')  # 重构图片为 28x28
    plt.title(f"预测值: {predicted_label}, 标签: {true_label}, {is_correct}")  # 设置标题
    plt.axis('off')  # 不显示坐标轴
    plt.show()

# 主程序
while True:
    try:
        print("请输入想查看的图像编号：", end="")
        num = int(input())
        if num == -1:
            break
        elif 0 <= num < 10000:
            display_compare(num)
        else:
            print("请输入有效的编号范围！")
    except ValueError:
        print("请输入整数！")
