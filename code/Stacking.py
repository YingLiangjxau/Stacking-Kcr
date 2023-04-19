import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn import metrics
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix

def load_feature_label():

    train_P= pd.read_csv(r'training1(label).csv')
    train_N = pd.read_csv(r'training0(label).csv')
    test_P = pd.read_csv(r'testing1(label).csv')
    test_N = pd.read_csv(r'testing0(label).csv')

    # label
    Y_train_P = train_P['label']
    Y_train_N = train_N['label']
    Y_test_P = test_P['label']
    Y_test_N = test_N['label']
    Y_train = np.concatenate((Y_train_P, Y_train_N), axis=0)
    Y_train = Y_train.reshape(-1)
    Y_test = np.concatenate((Y_test_P, Y_test_N), axis=0)
    Y_test = Y_test.reshape(-1)


    # feature_modal_1
    phyfeature_train = np.loadtxt('AE(out)-tra.txt')
    phyfeature_test = np.loadtxt('AE(out)-test.txt')

    # feature_modal_2
    seqfeature_train = np.loadtxt('EGAAC+Z-scale-tra.txt')
    seqfeature_test = np.loadtxt('EGAAC+Z-scale-test.txt')

    # feature_modal_3
    BLOSUM_train = np.loadtxt('BLOSUM62-tra.txt')
    BLOSUM_test = np.loadtxt('BLOSUM62-test.txt')

    return Y_train, Y_test, phyfeature_train, phyfeature_test, seqfeature_train, seqfeature_test, BLOSUM_train, BLOSUM_test

def get_stacking(clf, x_train, y_train, x_test, n_folds=5):

    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))

    kf = KFold(n_splits=n_folds)
    print(clf)
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]
        clf.fit(x_tra, y_tra)

        set_pre = clf.predict(x_tst)
        #set_acc = accuracy_score(y_tst, set_pre)
        set_pro = clf.predict_proba(x_tst)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_tst, set_pro)
        set_auc = auc(fpr, tpr)
        #print(test_index[0], 'set_acc:', set_acc, 'set_auc:', set_auc)
        print(test_index[0], 'set_auc:', set_auc)

        second_level_train_set[test_index] = clf.predict_proba(x_tst)[:, 1]
        test_nfolds_sets[:, i] = clf.predict_proba(x_test)[:, 1]

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)

    return second_level_train_set, second_level_test_set

def seqfeature_stackingmodel():
    train_sets_seqfeature = []
    test_sets_seqfeature = []
    for clf in [rf_model, lgbm_model, gdbc_model, xgb_model]:
        train_set_seqfeature, test_set_seqfeature = get_stacking(clf, seqfeature_train, Y_train, seqfeature_test)
        train_sets_seqfeature.append(train_set_seqfeature)
        test_sets_seqfeature.append(test_set_seqfeature)

    meta_train_seqfeature = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets_seqfeature], axis=1)
    meta_test_seqfeature = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets_seqfeature], axis=1)
    return meta_train_seqfeature, meta_test_seqfeature

def phyfeature_stackingmodel():
    train_sets_phyfeature = []
    test_sets_phyfeature = []
    for clf in [rf_model, lgbm_model, gdbc_model, xgb_model]:
        train_set_phyfeature, test_set_phyfeature = get_stacking(clf, phyfeature_train, Y_train, phyfeature_test)
        train_sets_phyfeature.append(train_set_phyfeature)
        test_sets_phyfeature.append(test_set_phyfeature)

    meta_train_phyfeature = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets_phyfeature], axis=1)
    meta_test_phyfeature = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets_phyfeature], axis=1)
    return meta_train_phyfeature, meta_test_phyfeature

# 添加
def BLOSUM_stackingmodel():
    train_sets_BLOSUM = []
    test_sets_BLOSUM = []
    for clf in [rf_model, lgbm_model, gdbc_model, xgb_model]:
        train_set_BLOSUM, test_set_BLOSUM = get_stacking(clf, BLOSUM_train, Y_train, BLOSUM_test)
        train_sets_BLOSUM.append(train_set_BLOSUM)
        test_sets_BLOSUM.append(test_set_BLOSUM)

    meta_train_BLOSUM = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets_BLOSUM], axis=1)
    meta_test_BLOSUM = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets_BLOSUM], axis=1)
    return meta_train_BLOSUM, meta_test_BLOSUM

# Y_train, seqfeature_train, phyfeature_train, BLOSUM_train这四个要同步打乱一下
def shuffle(Y_train, seqfeature_train, phyfeature_train, BLOSUM_train):

    shuffle_ix = np.random.permutation(np.arange(len(Y_train)))

    Y_train = Y_train[shuffle_ix]
    seqfeature_train = seqfeature_train[shuffle_ix]
    phyfeature_train = phyfeature_train[shuffle_ix]
    BLOSUM_train = BLOSUM_train[shuffle_ix]

    return Y_train, seqfeature_train, phyfeature_train, BLOSUM_train

Y_train, Y_test, phyfeature_train, phyfeature_test, seqfeature_train, seqfeature_test, BLOSUM_train, BLOSUM_test = load_feature_label()

Y_train, seqfeature_train, phyfeature_train, BLOSUM_train = shuffle(Y_train, seqfeature_train, phyfeature_train, BLOSUM_train)

rf_model = RandomForestClassifier(n_estimators=50, )
lgbm_model = LGBMClassifier()
gdbc_model = GradientBoostingClassifier()
xgb_model = XGBClassifier()
#tree_model = tree.DecisionTreeClassifier(criterion='gini')
#nb_model = GaussianNB()
#knn_model= KNeighborsClassifier(n_neighbors=6)


meta_train_seqfeature, meta_test_seqfeature= seqfeature_stackingmodel()
meta_train_phyfeature, meta_test_phyfeature = phyfeature_stackingmodel()
meta_train_BLOSUM, meta_test_BLOSUM = BLOSUM_stackingmodel()

meta_train = np.concatenate((meta_train_phyfeature, meta_train_seqfeature, meta_train_BLOSUM), axis=1)
meta_test = np.concatenate((meta_test_phyfeature,meta_test_seqfeature, meta_test_BLOSUM), axis=1)


print(meta_train.shape)
print(meta_test.shape)


# MLP作为次级分类器
dt_model = MLPClassifier(hidden_layer_sizes = (64,64), activation="relu", solver='adam', learning_rate_init = 0.01)
dt_model.fit(meta_train, Y_train)
df_predict = dt_model.predict(meta_test)
df_score = dt_model.predict_proba(meta_test)

acc = metrics.accuracy_score(Y_test, df_predict)
print('acc: '+str(acc))
f1 = metrics.f1_score(Y_test, df_predict,average='macro')
print('f1: '+str(f1))
fpr, tpr, thresholds = roc_curve(Y_test, df_score[:, 1])
auc = auc(fpr, tpr)
print('auc: '+str(auc))

precision, recall, thresholds = precision_recall_curve(Y_test, df_score[:, 1])


plt.figure(1)
plt.title('Precision/Recall Curve')# give plot a title
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')
plt.figure(1)
plt.plot(precision, recall)
plt.show()


plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
lw=2, label='ROC curve (area = %0.4f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.savefig('./result/ROC.pdf')
plt.show()

def calc(TN, FP, FN, TP):
    SN = TP / (TP + FN)  # recall
    SP = TN / (TN + FP)
    Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    # F1 = (2 * TP) / (2 * TP + FP + FN)
    fz = TP * TN - FP * FN
    fm = (TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)
    MCC = fz / pow(fm, 0.5)
    return SN, SP, ACC, MCC, Precision
tn, fp, fn, tp = confusion_matrix(Y_test, df_predict).ravel()
SN, SP, ACC, MCC, Precision = calc(tn, fp, fn, tp)
print('SN: '+str(SN))
print('SP: '+str(SP))
print('MCC: '+str(MCC))
print('ACC: '+str(ACC))
print('Precision: '+str(Precision))



