#人脸检测 找到人脸，不关心是谁
#人脸识别 校验，这个人是谁，分辨，验证
#火车站刷脸进展:检测人脸,身份证(人脸)
# 检测到的人脸和身份证的人脸进行对比,一样，放行
#和人工检票、类似的
#必对，特征相似，允许存在误差
#Data_annotation
import cv2
#from ...DataBase import DB
import keyboard
import sys
cap=cv2.VideoCapture(0)
face_detector=cv2.CascadeClassifier('E:\\huang\\haarcascade_frontalface_alt.xml')
filename=1
flag_write=False
# def on_key_press(event):
#     if event.name == 'w':  # 当按下'w'键时触发事件
#         flag_write=True
#     elif event.name=='q': exit()
#     elif event.name=='r': flag_write=False
# keyboard.on_press(on_key_press)
# keyboard.wait()
while True:
    flag,frame=cap.read()
    cv2.imshow("img2",frame)
    if flag==False:
        break
    gray=cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray) #人脸检测
    key=cv2.waitKey(1)
    if len(faces)!=0:#按下w才开始录入信息并且检测到人脸才进行数据录入
        for x,y,w,h in faces:
            face=gray[y:y+h,x:x+w]#获取人脸保存，50张
            face=cv2.resize(face,dsize=(64,64))#调整人脸的尺寸,统一
            retval,buffer=cv2.imencode('.jpg',face)
            binary_data=buffer.tobytes()
            if key==ord('w'): 
                print(faces)
                #DB.addPicture(binary_data,3)
                filename+=1#自增
            cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),
                        color=[0,0,255],thickness=2)
           #cv2.imshow("img",frame)
            cv2.waitKey(1)
            if key==ord('q'): sys.exit(1)
    cv2.imshow("img",frame)
    cv2.waitKey(1)
    # if filename>50:
    #     break
    # key=cv2.waitKey(1)
    # if key==ord('w'):
    #     flag_write=True
cv2.destroyAllWindows()
cap.release()
