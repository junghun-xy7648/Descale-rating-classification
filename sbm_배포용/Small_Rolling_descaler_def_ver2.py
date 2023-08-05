import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

CATEGORY_DIRS = {1: 'high', 2:'midhigh', 3: 'mid', 4: 'midlow', 5: 'low'}
CATEGORIES = {'high': 1, 'midhigh': 2, 'mid': 3, 'midlow': 4, 'low': 5}

# 이미지 경로읽기
def image_loader(folder, target=False): 
    base = Path(folder).glob('**\\*')
    all_files = [str(x) for x in base if x.is_file()]
    image_filenames = [x.split('\\')[-1] for x in all_files]

    position = [x.split('_')[-1] for x in image_filenames]
    position = [x.split('.')[-2] for x in position]
    PONID = [x.split('_')[-2] for x in image_filenames]
    ID = [x.split('-')[-1] for x in PONID]
    PON = [x.split('-')[-2] for x in PONID]
    Time = [x.split('_')[-3] for x in image_filenames]

    dataset = pd.DataFrame({'path': all_files, 'filename': image_filenames, 'Position': position, 'ID' : ID, 'PON' : PON, 'Time' : Time})
    if target:
        targets = [x.split('\\')[-2] for x in all_files]
        dataset['target'] = targets
    return dataset

# 이미지 흑백
def image_preprocess_gray(image, clipLimit=2.27, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize=tileGridSize) # clipLimit=0.5, tileGridSize=(8,32)
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = image[403:630, :].astype(np.uint8).copy()
    image = clahe.apply(image)
    gray  = image[:, 266:1776].astype(np.uint8).copy()

    return gray

def image_preprocess_value_Small_Rolling_channel(image, clipLimit=1.05, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize=tileGridSize) # clipLimit=0.5, tileGridSize=(8,32)
    image                   = cv2.imread(image, cv2.IMREAD_COLOR)
    image                   = image[403:630, 266:1776].astype(np.uint8).copy()
# 1단계: 배경제거코드 적용 : 색상검출방식으로 블룸검출사진 및 스케일검출사진 분리
    image_ycrcb                 = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # YCrCb색상으로 변경
# 2단계: 히스토그램 평활화 적용 : contrast 상향
# 2-1) 블룸검출사진 히스토그램 평활화
    planes_cla_bloom        = cv2.split(image_ycrcb)
    planes_cla_bloom[0]     = clahe.apply(planes_cla_bloom[0]) # Y:밝기 히스토그램 평활화 : 컬러이미지
    #planes_cla_bloom_0     = clahe.apply(planes_cla_bloom[0]) # Y:밝기 히스토그램 평활화 : 컬러이미지       
    bloom_ycrcb_aeq         = cv2.merge([planes_cla_bloom[0], planes_cla_bloom[1], planes_cla_bloom[2]])
    bloom_bgr_aeq           = cv2.cvtColor(bloom_ycrcb_aeq, cv2.COLOR_YCrCb2BGR) #  컬러이미지 BGR로 변경
    bloom_bgr_aeq_split     = cv2.split(bloom_bgr_aeq)
    bloom_hsv_aeq           = cv2.cvtColor(bloom_bgr_aeq, cv2.COLOR_BGR2HSV) #  컬러이미지 HSV로 변경하여 완성!
    planes_aeq_bloom        = cv2.split(bloom_hsv_aeq)

    value                   = planes_aeq_bloom[2] # Value 명도
    gray                    = planes_cla_bloom_0 # gray 명암
    red                     = bloom_bgr_aeq_split[2] # red 빨강
    green                   = bloom_bgr_aeq_split[1] # green 녹색
    blue                    = bloom_bgr_aeq_split[0] # blue 파랑
    saturation              = planes_aeq_bloom[1] # saturation 채도
    hue                     = planes_aeq_bloom[0] # hue 색조
    Cr                      = planes_cla_bloom[1] # Cr
    Cb                      = planes_cla_bloom[2] # Cb
    edge                   = cv2.Canny(gray, 190, 220) # 엣지검출
    
    return value, gray, red, green, blue, saturation, hue, Cr, Cb, edge

# 전처리함수
def data_preprocess(folder, clipLimit, tileGridSize):
    images = image_loader(folder)
    # print(images.shape)
    paths = images.path.values # path라는 컬럼의 값을 np.array로 만든다. images['path'].values와 같다.
    filenames = images.filename.values
    image = image_preprocess_gray(paths, clipLimit, tileGridSize)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트

    return hist

def data_preprocess_Small_Rolling(folder, clipLimit, tileGridSize, target=False):
    images = image_loader(folder)
    # print(images.shape)

    paths = images.path.values # path라는 컬럼의 값을 np.array로 만든다. images['path'].values와 같다.

    hists_1 = []
    hists_2 = []
    hists_3 = []
    hists_4 = []
    hists_5 = []
    hists_6 = []
    hists_7 = []
    hists_8 = []
    hists_9 = []
    hists_10= []


    for idx, image in enumerate(paths):
        # print(image)
        # image = image_preprocess_value_Small_Rolling(image, clipLimit, tileGridSize)
        value, gray, red, green, blue, saturation, hue, Cr, Cb, edge  = image_preprocess_value_Small_Rolling_channel(image, clipLimit, tileGridSize)

        value = cv2.calcHist([value], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
        gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
        red = cv2.calcHist([red], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
        green = cv2.calcHist([green], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
        blue = cv2.calcHist([blue], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
        saturation = cv2.calcHist([saturation], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
        hue = cv2.calcHist([hue], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
        Cr = cv2.calcHist([Cr], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
        Cb = cv2.calcHist([Cb], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
        Edge = cv2.calcHist([edge], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트

        hists_1.append(value)
        hists_2.append(gray)
        hists_3.append(red)
        hists_4.append(green)
        hists_5.append(blue)
        hists_6.append(saturation)
        hists_7.append(hue)
        hists_8.append(Cr)
        hists_9.append(Cb)
        hists_10.append(Edge)
    
    # dataset = pd.DataFrame.from_records(hists) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_value = pd.DataFrame.from_records(hists_1) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_value.columns = list('value_{}'.format(n) for n in range(256))
    dataset_gray = pd.DataFrame.from_records(hists_2) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_gray.columns = list('gray_{}'.format(n) for n in range(256))
    dataset_red = pd.DataFrame.from_records(hists_3) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_red.columns = list('red_{}'.format(n) for n in range(256))
    dataset_green = pd.DataFrame.from_records(hists_4) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_green.columns = list('green_{}'.format(n) for n in range(256))
    dataset_blue = pd.DataFrame.from_records(hists_5) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_blue.columns = list('blue_{}'.format(n) for n in range(256))
    # dataset_blue = dataset_blue[['blue_19', 'blue_20', 'blue_21', 'blue_22', 'blue_23', 'blue_24']]
    dataset_saturation = pd.DataFrame.from_records(hists_6) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_saturation.columns = list('saturation_{}'.format(n) for n in range(256))
    dataset_hue = pd.DataFrame.from_records(hists_7) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_hue.columns = list('hue_{}'.format(n) for n in range(256))
    dataset_Cr = pd.DataFrame.from_records(hists_8) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_Cr.columns = list('Cr_{}'.format(n) for n in range(256))
    dataset_Cb = pd.DataFrame.from_records(hists_9) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_Cb.columns = list('Cb_{}'.format(n) for n in range(256))
    dataset_Edge = pd.DataFrame.from_records(hists_10) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_Edge.columns = list('edge_{}'.format(n) for n in range(256))
    dataset_Edge = dataset_Edge[['edge_255']]
    dataset_Edge['edge_255'] = dataset_Edge['edge_255']

    # dataset = pd.concat([dataset, pd.DataFrame(targets, columns=['target'])], axis=1) # target도 y변수로 취합한다.
    dataset = pd.concat([dataset_gray, dataset_blue, dataset_Edge], axis=1) # target도 y변수로 취합한다.

    return dataset


def data_preprocess_Small_Rolling_once(imgPath, clipLimit, tileGridSize, target=False):
    
    image = imgPath

    hists_1 = []
    hists_2 = []
    hists_3 = []
    hists_4 = []
    hists_5 = []
    hists_6 = []
    hists_7 = []
    hists_8 = []
    hists_9 = []
    hists_10= []
    
    value, gray, red, green, blue, saturation, hue, Cr, Cb, edge  = image_preprocess_value_Small_Rolling_channel(image, clipLimit, tileGridSize)
    
    value = cv2.calcHist([value], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
    gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
    red = cv2.calcHist([red], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
    green = cv2.calcHist([green], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
    blue = cv2.calcHist([blue], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
    saturation = cv2.calcHist([saturation], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
    hue = cv2.calcHist([hue], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
    Cr = cv2.calcHist([Cr], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
    Cb = cv2.calcHist([Cb], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트
    Edge = cv2.calcHist([edge], [0], None, [256], [0, 256]).flatten() # histogram이라는 list를 hists에 넣는다. 2차원 리스트

    hists_1.append(value)
    hists_2.append(gray)
    hists_3.append(red)
    hists_4.append(green)
    hists_5.append(blue)
    hists_6.append(saturation)
    hists_7.append(hue)
    hists_8.append(Cr)
    hists_9.append(Cb)
    hists_10.append(Edge)
    
    # dataset = pd.DataFrame.from_records(hists) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_value = pd.DataFrame.from_records(hists_1) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_value.columns = list('value_{}'.format(n) for n in range(256))
    dataset_gray = pd.DataFrame.from_records(hists_2) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_gray.columns = list('gray_{}'.format(n) for n in range(256))
    dataset_red = pd.DataFrame.from_records(hists_3) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_red.columns = list('red_{}'.format(n) for n in range(256))
    dataset_green = pd.DataFrame.from_records(hists_4) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_green.columns = list('green_{}'.format(n) for n in range(256))
    dataset_blue = pd.DataFrame.from_records(hists_5) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_blue.columns = list('blue_{}'.format(n) for n in range(256))
    # dataset_blue = dataset_blue[['blue_19', 'blue_20', 'blue_21', 'blue_22', 'blue_23', 'blue_24']]
    dataset_saturation = pd.DataFrame.from_records(hists_6) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_saturation.columns = list('saturation_{}'.format(n) for n in range(256))
    dataset_hue = pd.DataFrame.from_records(hists_7) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_hue.columns = list('hue_{}'.format(n) for n in range(256))
    dataset_Cr = pd.DataFrame.from_records(hists_8) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_Cr.columns = list('Cr_{}'.format(n) for n in range(256))
    dataset_Cb = pd.DataFrame.from_records(hists_9) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_Cb.columns = list('Cb_{}'.format(n) for n in range(256))
    dataset_Edge = pd.DataFrame.from_records(hists_10) # from_records: list를 dataframe으로 만든다. 0 ~ 255는 x변수로 나타남
    dataset_Edge.columns = list('edge_{}'.format(n) for n in range(256))
    dataset_Edge = dataset_Edge[['edge_255']]
    dataset_Edge['edge_255'] = dataset_Edge['edge_255']

    # dataset = pd.concat([dataset, pd.DataFrame(targets, columns=['target'])], axis=1) # target도 y변수로 취합한다.
    dataset = pd.concat([dataset_gray, dataset_blue, dataset_Edge], axis=1) # target도 y변수로 취합한다.

    return dataset




##################################################################################################################################################################################################################################################################

# 학습용 이미지 경로읽기
def image_loader_train(folder, target=False): 
    base = Path('{}'.format(folder)).glob('**/*')
    all_files = [str(x) for x in base if x.is_file()]
    image_filenames = [x.split('\\')[-1] for x in all_files]
    dataset = pd.DataFrame({'path': all_files, 'target': targets, 'filename': image_filenames})
    if target:
        targets = [x.split('/')[-2] for x in all_files]
        dataset['target'] = targets
    return dataset

# 학습용 이미지전처리
def data_preprocess_model_train(folder, clipLimit, tileGridSize):
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

# Target열 분리
def data_split(dataset):
    X = dataset.drop('target', axis=1)
    Y = dataset['target']

    return X, Y

# xgb boost 전처리
def XG_preprocessing(y_train, y_test):
    
    y_train_xgb = [v-1 for v in y_train]
    y_val_xgb = [v-1 for v in y_test]

    return y_train_xgb, y_val_xgb

# 정확도 평가 함수
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='micro')
    recall = recall_score(y_test, pred, average='micro')
    print('오차행렬')
    print(confusion)
    print('정확도 : {:.4f}\n정밀도 : {:.4f}\n재현율 : {:.4f}'.format(accuracy, precision, recall))