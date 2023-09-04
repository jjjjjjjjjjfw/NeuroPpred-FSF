import sys
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier  ,RandomForestClassifier, ExtraTreesClassifier
import pandas as pd
import joblib
def greater_than_half(lst):
    result = []
    for item in lst:
        if item >= 0.525:
            result.append(1)
        else:
            result.append(0)
    return result
# Load data
with open('/mnt/raid5/data4/jwen/wenjian/01/featuers_embedding_normalize.pkl','rb') as f:
        features_ensemble_train= pickle.load(f)
with open('/mnt/raid5/data4/jwen/wenjian/01/label_embedding.pkl', 'rb') as f:
            Label = pickle.load(f)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Label = le.fit_transform(Label)
with open('/mnt/raid5/data4/jwen/wenjian/01/train_handfeature8038gai.pkl','rb') as f:
        feature_hand= pickle.load(f)
print(features_ensemble_train.dtype,feature_hand.dtype)
#feature_hand= np.array(feature_hand).reshape(8038,1282)
#features_ensemble_train=np.array(features_ensemble_train)

print(features_ensemble_train.shape,feature_hand.shape)

features_ensemble = np.concatenate((features_ensemble_train, feature_hand), axis=1)

#features_ensemble  = features_ensemble.astype(np.float32)
print(features_ensemble.dtype)
print(features_ensemble.shape)
print(np.isnan(features_ensemble).any())
#sys.exit(0)
# Create KFold object
#kf = KFold(n_splits=5, shuffle=True, random_state=7)

# Define XGBoost model
model1 = XGBClassifier(max_depth=22, n_estimators=3700, random_state=21, eval_metric='mlogloss')

model1.fit(features_ensemble, Label)
#model1 = joblib.load('/mnt/raid5/data4/jwen/wenjian/RF/rfselectmodel.pkl')
thresholds =np.array(model1.feature_importances_)
thresholds.sort()
thresholds=abs(np.sort(-thresholds))
# Define thresholds to test
#thresholds = np.linspace(0,1 , num=1000)
# Loop over thresholds
results = []
print(len(thresholds))
print(thresholds[1500])

# Initialize lists to store results
thresh=1313
accuracies = []
aurocs = []
mccs = []
SPs = []
SNs = []
tps = []
tns = []
fps = []
fns = []
# Loop over folds

X_train=features_ensemble
y_train=Label

with open('/mnt/raid5/data4/jwen/wenjian/01/test-featuers_embedding_normalize.pkl', 'rb') as f:
    test_features_ensemble_train = pickle.load(f)
with open('/mnt/raid5/data4/jwen/wenjian/01/test-label_embedding (1).pkl', 'rb') as f:
    test_Label = pickle.load(f)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
test_Label = le.fit_transform(test_Label)
with open('/mnt/raid5/data4/jwen/wenjian/01/test_handfeature8038gai.pkl', 'rb') as f:
    test_feature_hand = pickle.load(f)
test_features= np.concatenate((test_features_ensemble_train, test_feature_hand), axis=1)
X_test=test_features
y_test=test_Label
print('测试集')
print(X_test.dtype)
print(X_test.shape)
print(np.isnan(X_test).any())
print(len(y_test))


# Train model with feature selection
selection = SelectFromModel(model1,threshold=thresholds[thresh], prefit=True)
select_X_train = selection.transform(X_train)
select_X_test = selection.transform(X_test)
model2 = GradientBoostingClassifier(max_features=9, learning_rate=0.05, n_estimators=1300, max_depth=7, random_state=10)
#select_X_train  = select_X_train.astype(np.float32)
#print(select_X_train.shape,select_X_train.dtype)
model2.fit(select_X_train, y_train)

# Predict on test set
#y_pred = model2.predict(select_X_test)
y_pred_prob = model2.predict_proba(select_X_test)[:, 1]
y_pred = greater_than_half(y_pred_prob)
conf_mat = confusion_matrix(y_test, y_pred)
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_pred_prob)
mcc = matthews_corrcoef(y_test, y_pred)
tn, fp, fn, tp = conf_mat.ravel()
# print("tn:", tn)
# print("fp:", fp)
# print("fn:", fn)
# print("tp:", tp)
SP = tn / (tn + fp)
SN = tp / (tp + fn)
print(accuracy,auroc,mcc)
# Append to lists
accuracies.append(accuracy)
aurocs.append(auroc)
mccs.append(mcc)
SPs.append(SP)
SNs.append(SN)
tps.append(tp)
tns.append(tn)
fps.append(fp)
fns.append(fn)
# Calculate mean results for this threshold
mean_accuracy = np.mean(accuracies)
mean_auroc = np.mean(aurocs)
mean_mcc = np.mean(mccs)
mean_sp=np.mean(SPs)
mean_sn=np.mean(SNs)
mean_tp=np.mean(tps)
mean_tn=np.mean(tns)
mean_fp=np.mean(fps)
mean_fn=np.mean(fns)
print("n=%d"%(select_X_train.shape[1]))
print('thresh:',thresholds[thresh])
print('accuracy:',mean_accuracy)
print('auroc:',mean_auroc)
print('mcc:',mean_mcc)
print('sp:',mean_sp)
print('sn:',mean_sn)
print('tp:',mean_tp)
print('tn:',mean_tn)
print('fp:',mean_fp)
print('fn:',mean_fn)

