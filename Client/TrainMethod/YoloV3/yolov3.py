import cv2
import numpy as np
import matplotlib.pyplot as plt
def look_img(img):
    img_RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()

# 加载YOLOv3网络
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# 加载 COCO 数据集标签
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    print(classes)

# 随机生成颜色，用于标记检测到的目标框
colors = np.random.uniform(0, 255, size=(len(classes), 3))



# 读取视频流
cap = cv2.VideoCapture(0)

while True:
    # 从视频流中读取一帧图像
    ret, frame = cap.read()
    
    # 转换图像为blob格式
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(416, 416), swapRB=True, crop=False)
    print(blob.shape)
    net.setInput(blob)
    LayersNames=net.getLayerNames()
    print(LayersNames)
    # 获取YOLO输出层的名称
    a=net.getUnconnectedOutLayers()
    print("三个输出层的索引号：\n",a)
    # break
    output_layers = [LayersNames[i- 1] for i in net.getUnconnectedOutLayers()]
    print("三个输出层名称",output_layers)
    
    # 将blob输入到YOLO网络中，获取目标框和类别信息
    outputs = net.forward(output_layers)
    print(outputs[0].shape)
    print(len(outputs))
    print(outputs)
    # 解析YOLO输出，提取目标框信息和类别概率
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                print("confidence:",confidence)
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # 应用NMS算法，移除重叠的目标框
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    print("indices",indices)
    # 绘制检测到的目标框和类别名称
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness=2)
            cv2.putText(frame, classes[class_ids[i]], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
    
    # 显示结果
    cv2.imshow('YOLOv3 Real-time Detection', frame)
    
    # 按下q键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()