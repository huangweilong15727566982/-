import cv2
cap = cv2.VideoCapture(r'http://192.168.189.22:8080/video')
while True:
    ret, frame = cap.read()

    # 在这里添加您的代码，例如使用cv2.imshow()显示图像
    cv2.imshow("img",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()