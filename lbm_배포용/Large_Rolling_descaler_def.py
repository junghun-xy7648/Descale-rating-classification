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
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from pycaret.classification import *
import xgboost
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from lightgbm import LGBMClassifier
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize, RobustScaler, Normalizer

CATEGORIES = {'high': 1, 'midhigh': 2, 'mid': 3, 'midlow': 4, 'low': 5}
COLORS = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525']

# 이미지 경로읽기
def image_loader(path, target=False): 
    base = Path(path).glob('**\\*')
    all_files = [str(x) for x in base if x.is_file()]
    image_filenames = [x.split('\\')[-1] for x in all_files]

    PON = [x.split('_')[-1] for x in image_filenames]
    PON = [x.split('.')[-2] for x in PON]
    ID = [x.split('_')[-2] for x in image_filenames]
    heat = [x.split('_')[-3] for x in image_filenames]
    ch = [x.split('_')[-4] for x in image_filenames]
    wbf = [x.split('_')[-5] for x in image_filenames]
    Time = [x.split('_')[-6] for x in image_filenames]

    dataset = pd.DataFrame({'path': all_files, 'filename': image_filenames, 'PON': PON, 'ID' : ID, 'HEAT' : heat,'camera' : ch, 'WBF' : wbf, 'Time' : Time})
    if target:
        targets = [x.split('\\')[-2] for x in all_files]
        dataset['target'] = targets
    return dataset

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
    #planes_cla_bloom_1      = clahe.apply(planes_cla_bloom[0])
    bloom_ycrcb_aeq         = cv2.merge([planes_cla_bloom[0], planes_cla_bloom[1], planes_cla_bloom[2]])
    bloom_bgr_aeq           = cv2.cvtColor(bloom_ycrcb_aeq, cv2.COLOR_YCrCb2BGR) #  컬러이미지 BGR로 변경
    bloom_hsv_aeq           = cv2.cvtColor(bloom_bgr_aeq, cv2.COLOR_BGR2HSV) #  컬러이미지 HSV로 변경하여 완성!
    planes_aeq_bloom        = cv2.split(bloom_hsv_aeq)

    value                   = planes_aeq_bloom[2] # Value채도
    return value

# 다양한 오차 측정 지표를 확인하기 위한 함수 정의

from sklearn.metrics import *

def get_loss(y_test, y_predict, i): 
    # explained_variance_score =  explained_variance_score(y_test, pred)
    MSE = mean_squared_error(y_test, y_predict)
    RMSE = np.sqrt(MSE)
    r2 = r2_score(y_test, y_predict)
    print('model: {}, MSE: {:.4f}, RMSE: {:.4f}, r2: {:.4f}'.format(Model_name[i], MSE, RMSE, r2))
    
# y_predict = xgb_model.predict(x_test)
# # get_clf_eval()를 이용해 사키릿런 래퍼 XGBoost로 만들어진 모델 예측 성능 평가
# get_loss(y_test, y_predict)

# 사이킷런의 정확도, 정밀도, 재현율, 오차행렬을 계산하는 API 호출
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 호출한 지표들을 한꺼번에 계산하는 함수 정의
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='micro')
    recall = recall_score(y_test, pred, average='micro')
    print('오차행렬')
    print(confusion)
    print('정확도 : {:.4f}\n정밀도 : {:.4f}\n재현율 : {:.4f}'.format(accuracy, precision, recall))


def data_preprocess(folder, clipLimit, tileGridSize):
    images = image_loader(folder)
    # print(images.shape)

    paths = images.path.values # path라는 컬럼의 값을 np.array로 만든다. images['path'].values와 같다.

    hists = []
    for idx, image in enumerate(paths):
        image = image_preprocess_gray(image, clipLimit, tileGridSize)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
        hists.append(hist)

    targets = [CATEGORIES[target] for target in images.target.values] # high=1, midhigh=2, mid=3, midlow=4, low=5로 대치한다 

    dataset = pd.DataFrame.from_records(hists) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset = pd.concat([dataset, pd.DataFrame(targets, columns=['target'])], axis=1) # target도 y변수로 취합한다.
    
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, stratify=dataset['target'], random_state=111) # stratify : 1,2,3,4,5등급을 균등하게 나누어주는 인수

    return dataset, train_dataset, test_dataset

def data_preprocess_large_Rolling(folder, clipLimit, tileGridSize):
    images = image_loader(folder)
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
    
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, stratify=dataset['target'], random_state=111) # stratify : 1,2,3,4,5등급을 균등하게 나누어주는 인수

    return dataset, train_dataset, test_dataset

def data_split(dataset):
    X = dataset.drop('target', axis=1)
    Y = dataset['target']

    return X, Y

def XG_preprocessing(y_train, y_test):
    # xgb boost 전처리
    y_train_xgb = [v-1 for v in y_train]
    y_val_xgb = [v-1 for v in y_test]

    return y_train_xgb, y_val_xgb

def robustScaler(train_dataset, validation_dataset):
    # 변형 객체 생성
    robust_scaler = RobustScaler()
    X_train_dataset = train_dataset.drop('target', axis=1)
    #컬럼받기
    Col_Lst = list(X_train_dataset.columns) 
    # Col_Lst들에 전부 스케일러 적용한후 새로운 데이터프레임에 받아줌
    # 새 데이터프레임에도 Col_Lst지정해주어야!!
    train_dataset_scaler = train_dataset.copy()
    validation_dataset_scaler = validation_dataset.copy()

    # 훈련데이터의 모수 분포 저장 및 스케일링
    train_dataset_scaler[Col_Lst] = robust_scaler.fit_transform(train_dataset[Col_Lst]) 
    # 테스트 데이터의 모수 분포 저장 및 스케일링
    validation_dataset_scaler[Col_Lst] = robust_scaler.transform(validation_dataset[Col_Lst])

    return train_dataset_scaler, validation_dataset_scaler


def normalScaler(train_dataset, validation_dataset):
    # 변형 객체 생성
    normal_scaler = Normalizer()
    X_train_dataset = train_dataset.drop('target', axis=1)
    #컬럼받기
    Col_Lst = list(X_train_dataset.columns) 
    # Col_Lst들에 전부 스케일러 적용한후 새로운 데이터프레임에 받아줌
    # 새 데이터프레임에도 Col_Lst지정해주어야!!
    train_dataset_scaler = train_dataset.copy()
    validation_dataset_scaler = validation_dataset.copy()

    # 훈련데이터의 모수 분포 저장 및 스케일링
    train_dataset_scaler[Col_Lst] = normal_scaler.fit_transform(train_dataset[Col_Lst]) 
    # 테스트 데이터의 모수 분포 저장 및 스케일링
    validation_dataset_scaler[Col_Lst] = normal_scaler.transform(validation_dataset[Col_Lst])

    return train_dataset_scaler, validation_dataset_scaler

def select_feature(dataset):
    Col_Lst = [92, 69, 71, 79, 90, 67, 98, 174, 'target']
    dataset([Col_Lst])
    return dataset