import sys
from Small_Rolling_descaler_def_ver2 import *

#
# CATEGORY_DIRS = {1: 'high', 2:'midhigh', 3: 'mid', 4: 'midlow', 5: 'low'}
#

# 상수 정의
clipLimit=0.5           # cv2.createCLAHE 변수
tileGridSize = (8,8)    # cv2.createCLAHE 변수

# 모델 불러오기
MODEL_DIR = 'D:\\python_prj\\sbm\\models\\xtree_model_gray+blue+edge_smallRolling_20220601.joblib'
model = joblib.load(MODEL_DIR)

# 이미지 경로 받기
image_path = sys.argv[1]

# 이미지 전처리
histogram = data_preprocess_Small_Rolling_once(image_path, clipLimit, tileGridSize, target=False)

# 모델 처리
preds = model.predict(histogram)

# 결과
result = -1
for i in range(len(preds)):
    result = preds[i]

# 결과 출력    
print(result)
