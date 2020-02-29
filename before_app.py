import cv2
import numpy as np
import os

# def display(self):
#     # create a window
#     cv2.namedWindow("Image")
#     cv2.imshow("Image", self)
#     cv2.waitKey(0)
#     # cv2.destroyWindow("name")
#     cv2.destroyAllWindows()

# #获取摄像头图片
# cam = cv2.VideoCapture(0)
# while(cv2.waitKey(0)==-1):
#     ret,frame = cam.read()
green = (0,255,0)

# 每帧的镜头捕获
img = "test_3.jpg"
file = r"D:\I'M\pycharm\android_1\new"
save_file = os.path.join(file,img)
png = cv2.imread(img, 1)

#选中区域的色域
preset_Lower = np.array([156, 43, 46])
preset_Upper = np.array([179, 255, 255])

#膨胀处理
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
png2 = cv2.dilate(png, kernel, iterations=1)
# display(png2)

#掩膜处理
hsv = cv2.cvtColor(png2, cv2.COLOR_BGR2HSV) #转hsv
mask = cv2.inRange(hsv,preset_Lower,preset_Upper)
dst = cv2.bitwise_and(png2, hsv, mask=mask) #与操作即保留感兴趣区

#轮廓
gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(mask,100,255,cv2.THRESH_BINARY)#二值化
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
cv2.drawContours(png, contours, -1, color=green, thickness=3)
# print(contours)
a = np.mean(contours[1],axis=1) #axis=1是行（即宽）的平均值
print(a)
# wide = png.shape[1]
# print(wide)
cv2.imshow("png",png)
# cv2.imshow("dst", dst)
# cv2.imshow("gray",gray)
cv2.waitKey(0)