import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

src = cv2.imread('Test_data\\high\\20220315180341_wbf2_ch1_B69046_805_D22B3K0640.jpg') # 입력영상 밝은경우
# src = cv2.imread('candies2.png') # 입력영상 어두운경우
src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

if src is None:
    print('Image load failed!')
    sys.exit()

def on_trackbar(pos):
    hmin = cv2.getTrackbarPos('H_min', 'dst') # dst영상의 H_min를 받아오는 함수
    hmax = cv2.getTrackbarPos('H_max', 'dst') # H : 5 ~ 20 범위임
    Smin = cv2.getTrackbarPos('S_min', 'dst') # dst영상의 채도를 받아오는 함수
    Smax = cv2.getTrackbarPos('S_max', 'dst') # S : 200 ~ 255 범위임
    Vmin = cv2.getTrackbarPos('V_min', 'dst') # dst영상의 진하기를 받아오는 함수
    Vmax = cv2.getTrackbarPos('V_max', 'dst') # V : 150 ~ 255 범위임
    
    dst = cv2.inRange(src_hsv, (hmin, Smin, Vmin), (hmax, Smax, Vmax))
    cv2.imshow('dst', dst)



cv2.namedWindow('src', cv2.WINDOW_NORMAL)
cv2.imshow('src', src)
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)


cv2.createTrackbar('H_min', 'dst', 0, 179, on_trackbar)
cv2.createTrackbar('H_max', 'dst', 0, 179, on_trackbar)
cv2.createTrackbar('S_min', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('S_max', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('V_min', 'dst', 0, 255, on_trackbar)
cv2.createTrackbar('V_max', 'dst', 0, 255, on_trackbar)
on_trackbar(0)


# high_mask = cv2.inRange(src_hsv, (5, 200, 150), (20, 255, 255)) # left_camera : 디스케일러 high인 경우 수치
# high_mask = cv2.inRange(src_hsv, (5, 140, 135), (20, 255, 255)) # right_camera : 디스케일러 high인 경우 수치
# low_mask = cv2.inRange(src_hsv, (0, 140, 20), (30, 255, 255)) # 디스케일러 low인 경우 수치
# mid_mask = cv2.inRange(src_hsv, (0, 140, 90), (30, 255, 255)) # 디스케일러 low인 경우 수치
# mask = cv2.inRange(src_hsv, (0, 50, 150), (179, 255, 255)) # scale만 잡는 경우
bloom_mask = cv2.inRange(src_hsv, (0, 0, 152), (179, 255, 255)) # 디스케일러 low인 경우 수치
scale_mask1 = cv2.inRange(src_hsv, (0, 0, 2), (179, 255, 152)) # 디스케일러 low인 경우 수치
mask_bloom_inverse = ~bloom_mask
scale_mask = cv2.copyTo(scale_mask1, mask_bloom_inverse)


bloom_filter = cv2.copyTo(src_hsv, bloom_mask)
scale_filter = cv2.copyTo(src_hsv, scale_mask)

bloom_filter_bgr = cv2.cvtColor(bloom_filter, cv2.COLOR_HSV2BGR)
scale_filter_bgr = cv2.cvtColor(scale_filter, cv2.COLOR_HSV2BGR)

bloom_filter_bgr = bloom_filter_bgr[388:559, :]     # 2차개발 ch1,ch2통합
scale_filter_bgr = scale_filter_bgr[388:559, :]     # 2차개발 ch1,ch2통합

# bloom_filter_bgr = bloom_filter_bgr[388:559, :]     # 2차개발 ch1,ch2통합
# filter_bgr_2 = bloom_filter_bgr[450:540, 430:]     # 오른쪽카메라
# filter_bgr_3 = bloom_filter_bgr[450:540, 570:1780]  # 오른쪽카메라 중간이외 다 자르기 
# filter_bgr = filter_bgr[430:537, 402:1440] # 왼쪽카메라 가열로1기라인만 잘라냄
# filter_bgr = filter_bgr[430:595, 402:1480]   # 왼쪽카메라 가열로2기라인도 포함
bloom_mask = bloom_mask[388:559, :]
mask_bloom_inverse  = mask_bloom_inverse[388:559, :]
scale_mask1 = scale_mask1[388:559, :]
scale_mask = scale_mask[388:559, :]

# reverse = np.mean()



cv2.namedWindow('bloom_filter', cv2.WINDOW_NORMAL)
cv2.imshow('bloom_filter', bloom_filter_bgr)
cv2.namedWindow('scale_filter', cv2.WINDOW_NORMAL)
cv2.imshow('scale_filter', scale_filter_bgr)
cv2.namedWindow('mask_bloom_inverse', cv2.WINDOW_NORMAL)
cv2.imshow('mask_bloom_inverse', mask_bloom_inverse)
cv2.namedWindow('bloom_mask', cv2.WINDOW_NORMAL)
cv2.imshow('bloom_mask', bloom_mask)
cv2.namedWindow('scale_mask1', cv2.WINDOW_NORMAL)
cv2.imshow('scale_mask1', scale_mask1)
cv2.namedWindow('scale_mask', cv2.WINDOW_NORMAL)
cv2.imshow('scale_mask', scale_mask)


plt.imshow(src)
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()