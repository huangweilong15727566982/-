import cv2
import DB
import numpy as np
#将图片数据转为二进制数据
img=cv2.imread(r"./images/2.jpg")
retval,buffer=cv2.imencode('.jpg',img)
binary_data=buffer.tobytes()
img_array=np.frombuffer(binary_data,np.uint8)
img1=cv2.imdecode(img_array,cv2.IMREAD_COLOR)
cv2.imshow('img',img1)
# print(binary_data)
#存入收据库
#DB.addPicture(binary_data,2)
DB.showPic()
#cv2.imshow('test',img)
cv2.waitKey(0)