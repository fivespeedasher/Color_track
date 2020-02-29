import cv2
import numpy as np

#红色的HSV范围
preset_Lower = np.array([156, 43, 46])
preset_Upper = np.array([179, 255, 255])
green = (0,255,0)
def cutout(self):
    # 膨胀处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))#马赛克矩阵
    scr = cv2.dilate(self, kernel, iterations=1)
    # display(png2)

    # 掩膜处理
    hsv = cv2.cvtColor(scr, cv2.COLOR_BGR2HSV)  # 转hsv
    mask = cv2.inRange(hsv, preset_Lower, preset_Upper) #制作一个0与1的mask
    dst = cv2.bitwise_and(self, hsv, mask=mask)  # 与操作即保留感兴趣区
    return mask,dst

def curve(self):
    # gray = cv2.cvtColor(self, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(self, 100, 255, cv2.THRESH_BINARY)  # 二值化
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours,hierarchy

#通过手机APP“IP摄像头”打开的话，在里面读取rtsp即可
# cam = cv2.VideoCapture('rtsp://admin:admin@10.40.142.125:8554/live')


cam = cv2.VideoCapture(0)
# wide = cv2.CAP_PROP_FRAME_WIDTH #可是这并非像素点
while(cv2.waitKey(1)==-1):#敲下键盘关闭摄像头
    process, frame = cam.read()
    mask, dst = cutout(frame)

    wide = mask.shape[0]  # shape包含的数组[宽，高，3]
    high = mask.shape[1]
    longer = max(wide, high) #哪边是长取决于用户设置的分辨率
    if wide>=high:
        posture = 0
    else:posture = 1
    contours, hierarchy =  curve(mask)
    cv2.drawContours(frame, contours, -1, color=green, thickness=3)
    cv2.imshow('Video', frame)
    cv2.imshow("dst",dst)
    try:
        alist = np.mean(contours[1], axis=1) #axis=1代表每一列的平均值即返回[mean(宽)，mean(高)]
        # a = alist[1][0]#用于宽变化
        a = alist[1][0]

        if (a> longer/2):
            c=1
        elif (a<longer/2):
            c = -1
        else:c = 0
    except:
        a =longer/2
        c = 0
    print(a)
    # print(alist)
    with open ('control.txt','w') as f:
        f.write(str(c))
cam.release()
cv2.destroyAllWindows()
f.close()