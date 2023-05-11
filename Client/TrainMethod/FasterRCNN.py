import tensorflow as tf
import numpy as np
from PIL import Image
import os
import xml.etree.ElementTree as ET
import cv2
# 定义模型参数
class Config(object):
    # anchor 的大小和宽高比
    anchor_scales = [128, 256, 512]
    anchor_ratios = [0.5, 1, 2]
    # RPN 模块的超参数
    rpn_nms_thresh = 0.7
    rpn_fg_fraction = 0.5
    rpn_batch_size = 256
    rpn_bg_thresh_hi = 0.5
    rpn_bg_thresh_lo = 0.1
    rpn_pos_thresh = 0.7

    # Fast R-CNN 模块的超参数

    roi_nms_thresh = 0.3
    roi_fg_fraction = 0.25
    roi_batch_size = 128
    roi_bg_thresh_hi = 0.5
    roi_bg_thresh_lo = 0.1
    roi_pos_thresh = 0.5

class Dataset(object):#数据集类
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'JPEGImages')
        self.annotation_dir = os.path.join(data_dir, 'Annotations')
        self.image_list = os.listdir(self.image_dir)

    def get_image_info(self, image_idx):#获取一张图像的框框位置和分类号
        # 读取图像信息
        image_file = os.path.join(self.image_dir, self.image_list[image_idx])
        image = np.array(Image.open(image_file))
        # 读取标注信息
        annotation_file = os.path.join(self.annotation_dir, self.image_list[image_idx][:-4] + '.xml')
        tree = ET.parse(annotation_file)
        objs = tree.findall('object')
        bbox = [] #对应的框框位置 ((x1,y1),(x2,y2))
        label = [] #分类号
        for obj in objs:
            bbox.append([int(obj.find('bndbox').find('xmin').text),
                         int(obj.find('bndbox').find('ymin').text),
                         int(obj.find('bndbox').find('xmax').text),
                         int(obj.find('bndbox').find('ymax').text)])
            label.append(obj.find('name').text)
        return image, bbox, label  

class FasterRCNN(tf.keras.Model):#FasterRCNN类
    def __init__(self, config):
        super(FasterRCNN, self).__init__()
        self.config = config
        # 初始化 RPN 模块和 Fast R-CNN 模块
        self.rpn = RPN(config)
        self.fast_rcnn = FastRCNN(config)

    def call(self, inputs, training=False):
        # 输入为图像和对应的标注框和类别标签
        image, bbox, label = inputs
        # RPN 模块生成建议框
        proposals = self.rpn(image, training=training)
        # Fast R-CNN 模块使用建议框进行分类和回归
        outputs = self.fast_rcnn(image, proposals, bbox, label, training=training)
        return outputs

class RPN(tf.keras.Model): #框框预测类
    def __init__(self, config):
        super(RPN, self).__init__()
        self.config = config
        # 定义卷积层
        self.conv = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')
        # 定义分类分支和回归分支
        self.rpn_cls = tf.keras.layers.Conv2D(2 * len(config.anchor_ratios), (1, 1), activation='softmax')#分类
        self.rpn_reg = tf.keras.layers.Conv2D(4 * len(config.anchor_ratios), (1, 1))#回归

    def call(self, inputs, training=False):##返回建议框
        # 输入为图像
        x = self.conv(inputs)
        # 将卷积结果分别传入分类分支和回归分支
        cls_score = self.rpn_cls(x)
        bbox_pred = self.rpn_reg(x)
        # 将分类结果和回归结果reshape成适当的形状
        cls_score = tf.reshape(cls_score, [-1, 2])
        bbox_pred = tf.reshape(bbox_pred, [-1, 4])
        # 生成建议框
        proposals = self.generate_proposals(cls_score.numpy(), bbox_pred.numpy())
        return proposals

    def generate_proposals(self, cls_score, bbox_pred):
        anchors = self.generate_anchors()
        proposals = []
        for i, score in enumerate(cls_score):
            if score[1] > self.config.rpn_pos_thresh:
                proposals.append(self.apply_bbox_deltas(anchors[i], bbox_pred[i]))
        proposals = np.array(proposals)
        # 进行非极大值抑制
        keep_indices = self.non_max_supression(proposals)
        proposals = proposals[keep_indices]
        return proposals

    def generate_anchors(self):
        # 生成 anchor
        anchors = []
        for scale in self.config.anchor_scales:
            for ratio in self.config.anchor_ratios:
                w = scale * np.sqrt(ratio)
                h = scale / np.sqrt(ratio)
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return np.array(anchors)

    def apply_bbox_deltas(self, boxes, deltas):
        # 根据回归分支的输出调整建议框的坐标
        x_center = (boxes[:, 2] + boxes[:, 0]) / 2
        y_center = (boxes[:, 3] + boxes[:, 1]) / 2
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]

        x_center = x_center + deltas[:, 0] * width
        y_center = y_center + deltas[:, 1] * height
        width = width * tf.math.exp(deltas[:, 2])
        height = height * tf.math.exp(deltas[:, 3])

        boxes[:, 0] = x_center - width / 2
        boxes[:, 1] = y_center - height / 2
        boxes[:, 2] = x_center + width / 2
        boxes[:, 3] = y_center + height / 2
        return boxes

    def non_max_supression(self, proposals):
        # 非极大值抑制
        scores = np.ones(len(proposals))
        keep_indices = tf.image.non_max_suppression(proposals, scores, max_output_size=len(proposals),
                                                    iou_threshold=self.config.rpn_nms_thresh).numpy()
        return keep_indices

class RoI(tf.keras.layers.Layer):
    def __init__(self, config):
        super(RoI, self).__init__()
        self.config = config

    def call(self, inputs):
        # 输入为 feature map 和建议框
        feature_map, proposals = inputs
        rois = []
        for i, proposal in enumerate(proposals):
            x1, y1, x2, y2 = proposal
            x1 = int(max(0, np.round(x1)))
            y1 = int(max(0, np.round(y1)))
            x2 = int(min(feature_map.shape[2] - 1, np.round(x2)))
            y2 = int(min(feature_map.shape[1] - 1, np.round(y2)))
            roi = feature_map[:, y1:y2, x1:x2]
            roi = tf.image.resize(tf.expand_dims(roi, axis=0), [self.config.roi_size, self.config.roi_size])
            rois.append(tf.squeeze(roi, axis=0))
        rois = tf.stack(rois, axis=0)
        return rois

class FastRCNN(tf.keras.Model):
    def __init__(self, config):
        super(FastRCNN, self).__init__()
        self.config = config
        # 定义 RoI-Pooling 层
        self.roi_pooling = tf.keras.layers.TimeDistributed(RoI(config))
        # 定义分类分支和回归分支
        self.fc1 = tf.keras.layers.Dense(1024, activation='relu')
        self.fc2 = tf.keras.layers.Dense(1024, activation='relu')
        self.cls_score = tf.keras.layers.Dense(2, activation='softmax')#二维输出
        self.bbox_pred = tf.keras.layers.Dense(4)

    def call(self, inputs, proposals, bbox, label, training=False):
        # 输入为图像、建议框、标注框和标注标签
        feature_map = self.extract_features(inputs)
        rois = self.roi_pooling([feature_map, proposals])
        x = self.fc1(tf.reshape(rois, [self.config.roi_batch_size, -1]))
        x = self.fc2(x)
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        # 计算分类损失和回归损失
        cls_loss = self.compute_cls_loss(cls_score, label)
        bbox_loss = self.compute_bbox_loss(bbox_pred, bbox)
        # 使用总损失更新模型
        loss = cls_loss + bbox_loss
        self.add_loss(loss)
        return cls_score, bbox_pred

    def extract_features(self, inputs):
        # 提取 feature map
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=x.shape[1:])
        for layer in resnet50.layers:
            if layer.name == 'conv5_block3_out':
                break
            layer.trainable = False
        feature_map = resnet50(x)
        return feature_map

    def compute_cls_loss(self, cls_score, label):
        # 计算分类损失
        label = tf.cast(label, dtype=tf.int64)
        cls_target = tf.one_hot(label, depth=2)
        cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=cls_target, logits=cls_score)
        cls_loss = tf.reduce_mean(cls_loss)
        return cls_loss

    def compute_bbox_loss(self, bbox_pred, bbox):
        # 计算回归损失
        bbox = tf.cast(bbox, dtype=tf.float32)
        bbox_diff = bbox_pred - bbox
        bbox_loss = tf.abs(bbox_diff)
        bbox_loss = tf.where(bbox_diff < 1, 0.5 * bbox_loss ** 2, bbox_loss - 0.5)
        bbox_loss = tf.reduce_mean(bbox_loss)
        return bbox_loss

# if __name__ == '__main__':
#     # 定义模型和数据集
#     config = Config()
#     #dataset = Dataset('path/to/your/dataset')#
#     model = FasterRCNN(config)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#     # 训练模型
#     #dataset.image_list
#     #dataset.get_image_info(i)
#     for epoch in range(100):#迭代100轮
#         for i in range(len(dataset.image_list)):
#             image, bbox, label = dataset.get_image_info(i)# 第i张图片的信息 image 图片本身， 多个盒子，多个标签
#             with tf.GradientTape() as tape:
#                 cls_score, bbox_pred = model((image, bbox, label), training=True)
#             grads = tape.gradient(model.losses, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))
#             if i % 10 == 0:
#                 print('Epoch: {}, Image: {}, Loss: {}'.format(epoch, i, model.losses))

    # 预测模型
    # image, bbox, label = dataset.get_image_info(0)
    # #生成建议框
    # proposals = model.rpn(image[np.newaxis])

    # cls_score, bbox_pred = model(image[np.newaxis], proposals)
    # print('Class Scores: {}'.format(cls_score))
    # print('Bounding Box Predictions: {}'.format(bbox_pred))
# 这个函数测试 Faster R-CNN 模型，遍历数据集中的所有图像，对于每个图像，先前向传播得到预测结果，然后解码输出，
# 最后对预测结果进行非极大值抑制，最多保留100个预测框，并用红色框框出预测框。
def FRCNN_train(images,labels):
    #labels是字符串
    # 定义模型和数据集
    config = Config()
    #dataset = Dataset('path/to/your/dataset')#
    model = FasterRCNN(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
   # model.summary()
    for epoch in range(100):#迭代100轮
        for i in range(len(images)):
            string=labels[i]
            string=string.split()
            #groups_of_five = []
            bbox=[]
            label=[]
            print(string)
            for i in range(0, len(string), 5):  # 每隔五个元素进行一次循环
                print("i=",i)
                bbox.append([int(string[i]),
                             int(string[i+1]),
                             int(string[i+2]),
                             int(string[i+3])])
                label.append(string[i+4])
            bbox = tf.reshape(bbox, [-1, 4])
            print("bbox,label",bbox,"\n",label)
                # group = string[i:i+5]
                # groups_of_five.append(group)
            image=images[i]
            img_array=np.frombuffer(image,np.int32)
            img=cv2.imdecode(img_array,cv2.IMREAD_COLOR)
            image = tf.image.convert_image_dtype(img, dtype=tf.float32)
            image = np.array([image])
            print(image.shape)
            #第i张图片的信息 image 图片本身， 多个盒子，多个标签
            with tf.GradientTape() as tape:
                cls_score, bbox_pred = model((image, bbox, label), training=True)
                grads = tape.gradient(model.losses, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if i % 10 == 0:
                print('Epoch: {}, Image: {}, Loss: {}'.format(epoch, i, model.losses))
    model.save_weights(r"FastRCNN.h5")


