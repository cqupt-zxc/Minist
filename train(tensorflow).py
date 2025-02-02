import os
import numpy as np
import tensorflow as tf

# 屏蔽 TensorFlow 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 使用 He 初始化权重
def init_weights(shape):
    initializer = tf.keras.initializers.HeNormal()
    return tf.Variable(initializer(shape=shape))

# 初始化权重
h1 = init_weights([784, 128])  # 第一层权重
h2 = init_weights([128, 64])  # 第二层权重
out = init_weights([64, 10])  # 输出层权重

# 定义模型
def model(x, h1, h2, out):
    x = tf.nn.relu(tf.matmul(x, h1))  # 第一层
    x = tf.nn.relu(tf.matmul(x, h2))  # 第二层
    x = tf.matmul(x, out)             # 输出层
    return x

# 定义 Keras 包装模型
class CustomModel(tf.keras.Model):
    def __init__(self, h1, h2, out):
        super(CustomModel, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.out = out

    def call(self, x):
        return model(x, self.h1, self.h2, self.out)

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, 784).astype("float32") / 255.0  # 归一化
test_images = test_images.reshape(-1, 784).astype("float32") / 255.0
train_labels = tf.one_hot(train_labels, 10)
test_labels = tf.one_hot(test_labels, 10)

# 定义超参数
batch_size = 64
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.96, staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# 定义训练步骤
def train_step(batch_x, batch_y):
    with tf.GradientTape() as tape:
        logits = model(batch_x, h1, h2, out)  # 前向传播
        loss = loss_fn(batch_y, logits)      # 计算损失
    grads = tape.gradient(loss, [h1, h2, out])  # 计算梯度
    optimizer.apply_gradients(zip(grads, [h1, h2, out]))  # 更新权重
    acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(batch_y, logits))  # 计算准确率
    return loss.numpy(), acc.numpy()

# 创建 Keras 模型实例
custom_model = CustomModel(h1, h2, out)

# 开始训练
train_num = int(input("请输入需要训练的次数："))
for step in range(1, train_num + 1):
    indices = np.random.choice(len(train_images), batch_size, replace=False)
    batch_x, batch_y = tf.gather(train_images, indices), tf.gather(train_labels, indices)
    loss, accuracy = train_step(batch_x, batch_y)

    if step % 1000 == 0:
        print(f"第 {step} 次训练 损失值: {loss:.6f} 训练准确率: {accuracy * 100:.3f}%")

# 测试集整体准确率
logits = model(test_images, h1, h2, out)
predictions = tf.argmax(logits, axis=1)
labels = tf.argmax(test_labels, axis=1)
test_accuracy = tf.reduce_mean(tf.cast(predictions == labels, tf.float32)).numpy()
print(f"平均测试准确率：{test_accuracy * 100:.3f}%")

# 构建模型以保存权重
custom_model.build(input_shape=(None, 784))  # 显式构建模型
os.makedirs('./mnist_models', exist_ok=True)  # 确保保存路径存在
custom_model.save_weights('./mnist_models/model.weights.h5')  # 保存权重
print("模型已保存为 './mnist_models/model.weights.h5'")
