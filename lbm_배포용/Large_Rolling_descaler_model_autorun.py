import sys
from Large_Rolling_descaler_def import *

#
# CATEGORY_DIRS = {1: 'high', 2:'midhigh', 3: 'mid', 4: 'midlow', 5: 'low'}
#

# 상수 정의
clipLimit=1.05          # cv2.createCLAHE 변수
tileGridSize = (8,8)    # cv2.createCLAHE 변수

# 모델 불러오기
MODEL_DIR = 'D:\\python_prj\\lbm\\models\\xtree_model_dataset_tune_largeRolling_20220529.joblib' 
model = joblib.load(MODEL_DIR)

# 이미지 경로 받기
image_path = sys.argv[1]

# 이미지 전처리
image = image_preprocess_value_large_Rolling(image_path, clipLimit=clipLimit, tileGridSize=tileGridSize)

# 모델 처리
hists = []
hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
hists.append(hist)
histogram = pd.DataFrame.from_records(hists) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
preds = model.predict(hists)

# 결과
result = -1
for i in range(len(preds)):
    result = preds[i]

# 결과 출력    
print(result)

