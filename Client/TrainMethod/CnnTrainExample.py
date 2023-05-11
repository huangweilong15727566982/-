import tensorflow as tf
from tensorflow.keras import layers

# 设置超参数
batch_size = 32
epochs = 10
num_classes = 10
# `batch_size` 表示每次训练中使用的样本数量，
# `epochs` 表示训练轮数，
# `num_classes` 表示分类的类别数目。
# 加载数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# 通过调用`tf.keras.datasets.cifar10`的`load_data()`方法，我们可以获取训练集和测试集中的图像和标签数据。
#10分类 


# 对图像进行归一化
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
#将图像像素值归一化到0和1之间，可以有助于提高训练和准确度。

# 将类别标签转换为独热编码  使用`tf.keras.utils.to_categorical`方法将标签转换为独热编码，以便在训练模型时进行处理。
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

# 定义模型
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),#输入层
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')#输出有num_classes个分类
])
#  这个模型是一个卷积神经网络 (Convolutional Neural Network, CNN)，用于图像分类。下面是这个模型的底层工作原理的分解：
# 1. 输入层
# 输入层是模型的第一个层，其作用是将输入的数据传递给模型。模型的输入是一个 (32, 32, 3) 的三维张量，表示输入的图像大小为 32x32 像素，颜色通道为 3。
# 在这个模型中，使用了一个卷积层和一个池化层。
# 2. 卷积层
# 卷积层是深度学习中的一种特殊的线性变换，可以对图像进行特征提取。在这个模型中，使用了两个卷积层，第一个卷积层有 32 个 3x3 的卷积核，
# 第二个卷积层有 64 个 3x3 的卷积核。卷积层的输出是一个张量，表示从输入图像中提取的特征图。
# 3. 池化层
# 池化层是一种降维的操作，可以减小特征图的大小并且进一步提取图像的关键特征。在这个模型中，使用了两个最大池化层，每个池化层的大小为 (2, 2)，步幅为 2。
# 这意味着特征图的大小被减半了。
# 4. 全连接层
# 全连接层是神经网络中的一种常见层次，可以对特征进行非线性组合和分类。在这个模型中，使用了两个全连接层，每个全连接层包含 64 个神经元，使用了 ReLU 作为激活函数。
# 5. 输出层
# 输出层是模型的最后一个层，其作用是将模型的输出进行分类。在这个模型中，输出层是一个具有 num_classes 个神经元的密集层，使用了 softmax 作为激活函数。
# 这意味着模型的输出可以用一个概率分布来表示，每个类别的概率都是 0 到 1 之间的数字，并且所有概率之和为 1。
# 综上，该模型通过卷积和池化层、全连接层以及 softmax 输出层，对输入图像进行特征提取和分类
# 模型的底层工作原理可以总结为从底层到高层依次完成图像特征提取和分类预测的过程。

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
print(train_images.shape,"  ",train_labels.shape)
print(train_labels[0:20])
model.summary()
#model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels))
#model.save('my_model.h5')
# [bot] 在TensorFlow中，可以使用`model.save()`方法将已经训练好的模型保存在本地文件中。保存模型的方法可以有两种，一种是保存整个模型，另一种是仅保存模型的权重。

# 1. 保存整个模型

# 保存整个模型包括模型的架构和权重，在之后使用该模型时，可以直接进行加载和使用。

# ```
# # 保存整个模型
# model.save('my_model.h5')
# ```

# 这里将已经训练好的模型保存在‘my_model.h5’文件中。它将整个模型保存在一个HDF5文件中，包括模型的架构、权重和模型的训练配置。

# 2. 仅保存模型的权重

# 如果只需要保存模型的权重，可以使用`model.save_weights()`方法。

# ```
# # 仅保存模型的权重
# model.save_weights('my_model_weights.h5')
# ```

# 这里将模型的权重保存在‘my_model_weights.h5’文件中，这种保存方式不包含模型的整体结构和训练配置，所以在加载时需要重新定义模型和加载权重。

# 使用`model.load_weights()`方法来加载模型权重。

# ```
# # 加载模型权重
# model.load_weights('my_model_weights.h5')
# ```

# 这样就可以加载权重并将其应用在模型中，继续对模型进行训练或进行预测。