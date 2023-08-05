from large_Rolling_descaler_def import *

# CATEGORY_DIRS = {1: 'high', 2:'midhigh', 3: 'mid', 4: 'midlow', 5: 'low'}
ROOT_DIR = '.\\test_images' # 디스케일러 사진경로
# SAVE_DIR = '.\result'
clipLimit=1.05 # cv2.createCLAHE 변수
tileGridSize = (8,8) # cv2.createCLAHE 변수
MODEL_DIR = 'models\\xtree_model_dataset_tune_largeRolling_20220529.joblib' # 머신러닝모델 경로

model = joblib.load(MODEL_DIR)
dataset = image_loader(ROOT_DIR, target=False)
paths = dataset.path.values
filenames = dataset.filename.values
# PONs = dataset.PON.values
# IDs = dataset.ID.values
# HEATs = dataset.HEAT.values
# cameras = dataset.camera.values
# WBF = dataset.WBF.values
# Time = dataset.Time.values


hists = []
for idx, image in enumerate(paths):
    image = image_preprocess_value_large_Rolling(image, clipLimit=clipLimit, tileGridSize=tileGridSize)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
    hists.append(hist)

histogram = pd.DataFrame.from_records(hists) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
preds = model.predict(hists)

label = pd.concat([dataset, histogram, pd.DataFrame({'predict': preds})], axis=1)
label.to_csv('model_predict_large_Rolling.csv')



# print(label.head(5))
# for i in range(len(preds)):
#     class_ = preds[i]
#     print(class_)