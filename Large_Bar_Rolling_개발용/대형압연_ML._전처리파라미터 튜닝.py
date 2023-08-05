import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import glob
import random
import os
from pathlib import Path
import re
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate


# 데이터프레임(메타데이터) -> 이미지 로드 -> 이미지 전처리 -> 데이터프레임(빈도수, 타겟) -> 모델 적용 -> 결과 예측 -> 재학습 관련

CATEGORIES = {'high': 1, 'midhigh': 2, 'mid': 3, 'midlow': 4, 'low': 5}
COLORS = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525']
EPOCHS = 50
BATCH_SIZE = 32
AUTO = tf.data.experimental.AUTOTUNE # batch처리전에 미리 이전 batch를 준비한다.

# is_file() 이름을 str으로 바꿔라
# targets : high, midhigh, mid, midlow, low로 바꾸어라
def image_loader(): 
    base = Path('images').glob('**/*')
    all_files = [str(x) for x in base if x.is_file()]
    targets = [x.split('\\')[1] for x in all_files]
    dataset = pd.DataFrame({'path': all_files, 'target': targets})
    return dataset

def image_loader_new(): 
    base = Path('Train_data').glob('**/*')
    all_files = [str(x) for x in base if x.is_file()]
    targets = [x.split('\\')[1] for x in all_files]
    steel_grade = [x.split('\\')[2] for x in all_files]
    dataset = pd.DataFrame({'path': all_files, 'target': targets, 'steel_grade': steel_grade})
    return dataset


def image_preprocess(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = image[900:1200, 300:2100]
    clahe = cv2.createCLAHE(clipLimit=2.27, tileGridSize=(8, 8)) # clipLimit만 
    image = clahe.apply(image)
    return image

# 1.4115
def image_preprocess_gray(image, clipLimit=2.27, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize=tileGridSize) # clipLimit=0.5, tileGridSize=(8,32)
    image                   = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image                   = image[403:630, :].astype(np.uint8).copy()
    image                   = clahe.apply(image)
    gray                    = image[:, 266:1776].astype(np.uint8).copy()

    return gray

def image_preprocess_value_large_Rolling(image, clipLimit=1.05, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize=tileGridSize) # clipLimit=0.5, tileGridSize=(8,32)
    image                   = cv2.imread(image, cv2.IMREAD_COLOR)
    image                   = image[388:559, :].astype(np.uint8).copy()
# 1단계: 배경제거코드 적용 : 색상검출방식으로 블룸검출사진 및 스케일검출사진 분리
    image_ycrcb                 = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # YCrCb색상으로 변경
# 2단계: 히스토그램 평활화 적용 : contrast 상향
# 2-1) 블룸검출사진 히스토그램 평활화
    planes_cla_bloom        = cv2.split(image_ycrcb)
    planes_cla_bloom[0]     = clahe.apply(planes_cla_bloom[0]) # Y:밝기 히스토그램 평활화 : 컬러이미지
    bloom_ycrcb_aeq         = cv2.merge(planes_cla_bloom)
    bloom_bgr_aeq           = cv2.cvtColor(bloom_ycrcb_aeq, cv2.COLOR_YCrCb2BGR) #  컬러이미지 BGR로 변경
    bloom_hsv_aeq           = cv2.cvtColor(bloom_bgr_aeq, cv2.COLOR_BGR2HSV) #  컬러이미지 HSV로 변경하여 완성!
    planes_aeq_bloom        = cv2.split(bloom_hsv_aeq)

    value                   = planes_aeq_bloom[2] # Value채도
    return value

def pca(data, targets):
    pca = PCA(n_components=2)
    result = pca.fit_transform(data)

    plt.figure(figsize=(12, 12))
    plt.xlim(result[:, 0].min(), result[:, 0].max())
    plt.ylim(result[:, 1].min(), result[:, 1].max())

    for i in range(len(data)):
        plt.text(result[i, 0], result[i, 1], i,
                 color=COLORS[targets[i]], fontdict={'weight': 'bold', 'size': 9})
    plt.show()


def tsne(data, targets):
    model = TSNE(n_components=2)
    result = model.fit_transform(data)

    plt.figure(figsize=(12, 12))
    plt.xlim(result[:, 0].min(), result[:, 0].max() + 1)
    plt.ylim(result[:, 1].min(), result[:, 1].max() + 1)

    for i in range(len(data)):
        plt.text(result[i, 0], result[i, 1], i,
                 color=COLORS[targets[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.show()


def plot_hist(history):
    plt.plot(history.history['accuracy'], label='accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='val_accuracy', color='red')
    plt.legend()
    plt.show()

# 모델 구성하는 background 
def create_model():
    model = Sequential([
        Dense(128, input_dim=256, activation='relu'),
        Dropout(rate=0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(5, activation='sigmoid'),
    ])

    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy']) # metrics: 평가를 정확도로 하겠다.
    # print(model.summary())

    return model


def main(clipLimit, tileGridSize):
    images = image_loader_new()
    # print(images.shape)

    paths = images.path.values # path라는 컬럼의 값을 np.array로 만든다. images['path'].values와 같다.

    hists = []
    for idx, image in enumerate(paths):
        image = image_preprocess_value_large_Rolling(image, clipLimit, tileGridSize)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
        hists.append(hist)

    targets = [CATEGORIES[target] for target in images.target.values] # high=1, midhigh=2, mid=3, midlow=4, low=5로 대치한다 

    dataset = pd.DataFrame.from_records(hists) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset = pd.concat([dataset, pd.DataFrame(targets, columns=['target'])], axis=1) # target도 y변수로 취합한다.
    # print(dataset)


    train_dataset, validation_dataset = train_test_split(dataset, test_size=0.2, stratify=dataset['target'], random_state=111) # stratify : 1,2,3,4,5등급을 균등하게 나누어주는 인수
    # X_train, X_val, y_train, y_val = train_test_split(hists, targets, test_size=0.2, stratify=targets, random_state=111)

    # print(train_dataset)
    # print(validation_dataset)
#------------------------------------------------------------------------------------------------------------------
    X_train, y_train = train_dataset.iloc[:, :-1], train_dataset.iloc[:, -1].values
    X_val, y_val = validation_dataset.iloc[:, :-1], validation_dataset.iloc[:, -1].values

    # from sklearn.tree import DecisionTreeClassifier
    # model = DecisionTreeClassifier()
    # model.fit(X_train, y_train)
    # preds = model.predict(X_val)
    # print(preds)
    # print(model.feature_importances_)

    from sklearn.metrics import accuracy_score
    # print(accuracy_score(y_val, preds))

    from sklearn.ensemble import RandomForestClassifier
    
    
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    preds = rf.predict(X_val)
    scores = cross_val_score(rf, dataset, dataset.target , scoring='accuracy', cv=5, n_jobs=-1)
    print('교차 검증별 정확도:', np.round(scores,4))
    print('평균 검증 정확도:', np.round(np.mean(scores),4))
    # ===============================================================================
    # param_grid = {
    #     'max_depth': [2, 4, 6, 8],
    #     'min_samples_split': [2, 3, 4]
    # }
    # model = RandomForestClassifier(n_estimators=1000)
    # gscv = GridSearchCV(model, param_grid=param_grid, verbose=True)
    # gscv.fit(X_train, y_train)
    # preds = gscv.predict(X_val)
    # ===============================================================================

    # print(preds)
    print('tileGridSize: ', tileGridSize, 'clipLimit: ', clipLimit,'random forest accuracy: ', np.mean(scores))
    # save
    
    CL.append(clipLimit)
    tileGrid.append(tileGridSize)
    RandomForest.append(np.mean(scores))
    

    # joblib.dump(model, f'./models/rf_model_10_{datetime.today().strftime("%Y%m%d")}.joblib')


    import xgboost

    param_grid = {
        'max_depth': [2, 4, 6, 8],
        'min_samples_split': [2, 3, 4]
    }

    xgb_model = xgboost.XGBClassifier(n_estimators=1000)
    y_train = [v-1 for v in y_train]
    xgb_model.fit(X_train, y_train)

    preds = xgb_model.predict(X_val)
    y_val = [v-1 for v in y_val]

    # scores = cross_val_score(xgb_model, dataset, [v-1 for v in dataset.target], scoring='accuracy', cv=2, n_jobs=-1)
    # print('교차 검증별 정확도:', np.round(scores,4))
    # print('평균 검증 정확도:', np.round(np.mean(scores)),4)

    # print(preds)
    print('tileGridSize: ', tileGridSize, 'clipLimit: ', clipLimit, 'xgboost accuracy: ',  accuracy_score(y_val, preds))
    XGD.append( accuracy_score(y_val, preds))

    # joblib.dump(xgb_model, f'./models/xgb_model_10_{datetime.today().strftime("%Y%m%d")}.joblib')

#     from sklearn.ensemble import GradientBoostingClassifier # 트리기반모델의 경우, 분류시각화가 가능하다.
#     model = GradientBoostingClassifier()
#     model.fit(X_train, y_train)
#     preds = model.predict(X_val)
#     print(preds)
#     print(accuracy_score(y_val, preds))


# # 하이퍼파라메터 튜닝
# # CV Fold
# # 앙상블

# # 모델을 저장하고 불러오기
#     # from joblib import dump, load
#     # dump(model, 'filename.joblib') # filename.joblib
#     # model = load('filename.joblib')
#  # print(''.join([str(pred) for pred in preds]))

# #------------------------------------------------------------------------------------------------------------------
# # 딥러닝 코드
#     train_size = train_dataset.shape[0]
#     validation_size = validation_dataset.shape[0]
# # from_tensor_slices: 텐서플로우 기능 중 numpy가 아닌 tensor로 변환한다. → gpu를 사용할수 있다.
# # Target변수를 원-핫 인코딩으로 바꾼다. (1~5등급을 0, 1로 바꾸기)

#     train_dataset = tf.data.Dataset.from_tensor_slices(( 
#         train_dataset.iloc[:, :-1], 
#         pd.get_dummies(train_dataset.iloc[:, -1].values))
#     )
# # from_tensor_slices: 텐서플로우 기능 중 numpy가 아닌 tensor로 변환한다. → gpu를 사용할수 있다.
# # Target변수를 원-핫 인코딩으로 바꾼다. (1~5등급을 0, 1로 바꾸기)

#     validation_dataset = tf.data.Dataset.from_tensor_slices((
#         validation_dataset.iloc[:, :-1],
#         pd.get_dummies(validation_dataset.iloc[:, -1].values))
#     )

# # 1,2,3,4,5를 repeat(5) = 1234512345123451234512345 데이터를 5배로 늘린다.
# # shuffle(train_size) : buffer_size(특정간격)으로 train_size만큼 섞는다.
# # batch(BATCH_SIZE) : BATCH_SIZE=16만큼 묶는다.(1~16, 17~32, 33~48 이런식으로 묶는다)
# # prefetch(AUTO) : 시간단축할 때 쓰는 것.
# # 가져오기는 전처리와 훈련 스텝의 모델 실행을 오버랩합니다. 모델이 s스텝 훈련을 실행하는 동안 입력 파이프라인은 s+1스텝의 데이터를 읽습니다. 이렇게 하면 훈련을 하는 최대(합과 반대로) 스텝 시간과 데이터를 추출하는 데 걸리는 시간을 단축시킬 수 있습니다. 
#     train_dataset = train_dataset.repeat(5).shuffle(train_size).batch(BATCH_SIZE).prefetch(AUTO)
#     validation_dataset = validation_dataset.repeat(5).shuffle(validation_size).batch(BATCH_SIZE).prefetch(AUTO)

#     # for X, y in train_dataset.take(1):
#     #     print(X.numpy())
#     #     print(y.numpy())

#     model = create_model()
#     history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)
# # metrics로 정한 정확도, loss가 epoch마다 어떻게 변해왔는가. 그런 것들이 history에 저장됨

#     plot_hist(history)

# # preds = model.predict(X_val) 새로운 X_val이라는 hist는 고차원의 input이 필요함
# # ex) [[1,2 ,3 ], [4, 5, 6]] -> [1, 2] 2차원 input → 1차원 output 그래서 X_val도 2차원 input가 필요하니 np.expand_dims()이 필요함

if __name__ == '__main__':
    CL = []
    RandomForest = []
    XGD = []
    tileGrid = []
    clipLimits = np.arange(0, 10, 0.01) # 전처리 인자1
    tileGridSizes = [(8, 8)]

    # tileGridSizes = [(i, i) for i in range(1, 11)]
    # for grid in [(8,32), (4,16), (2,8), (1,4)]: # 전처리 인자2
    #     tileGridSizes.append(grid)

    print(tileGridSizes)
    for tileGridSize in tileGridSizes:
        for clipLimit in clipLimits:
            main(clipLimit=clipLimit, tileGridSize=tileGridSize)
    preprocess = pd.DataFrame({'tileGridSize': tileGrid, 'clipLimit': CL, 'RandomForest': RandomForest, 'xgboost': XGD})
    print(preprocess)
    preprocess.to_csv('Cliplimit최적화.csv')
    # np.expand_dims() : 차원 늘리는 함수
    # 딥러닝을 저장하고 불러오기