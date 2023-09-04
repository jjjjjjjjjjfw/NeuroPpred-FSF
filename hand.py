#import pickle
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import math
import warnings
import os
import sys
from Extract_feature import *
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.decomposition import PCA, TruncatedSVD as svd
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, matthews_corrcoef
import sys
print('Data Loading...')
import pandas as pd
# 读取CSV文件，假设第一列是序列，第二列是标签
data= pd.read_csv("/home/wenjian/IPPF-FE-main/train01.csv", header=None)
import numpy as np
# （1）获取第1列
Sequence = data.iloc[1:,0]
Sequence =np.array(Sequence)
label = data.iloc[1:,1]
label = np.array(label)
strs = Sequence
len_str = len(strs[0])
min_num_index = 0   # 最小值的下标
stack = [strs[0]]   # 利用栈来找出最短的字符串
for index, string in enumerate(strs):
    if len(string) < len_str:
        stack.pop()
        len_str = len(string)
        min_num_index = index # 知道最短字符对应的下标后，也可以自己找出最短字符
        stack.append(string)
print("最短字符串长度:", len_str)
print("最短字符串下标:", min_num_index)
print("最短字符串:", stack)
print("最短字符串:", strs[min_num_index])

print(Sequence)
features_crafted=Get_features(Sequence, 4)
print('feature_crafted:',len(features_crafted))
#print(features_crafted)
#print(np.isnan(features_crafted).any())
#features_crafted=np.nan_to_num(features_crafted)
#print(np.isnan(features_crafted).any())
with open('test_handfeature8038gai.pkl', 'wb') as f:
        pickle.dump(features_crafted, f)
