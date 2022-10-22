import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


'''def feature_select_pca(X, y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X)
    # X_test = sc.transform(X_test)

    pca = PCA(n_components=8)
    X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    # print(type(X_train))
    # data_scaled = pd.DataFrame(preprocessing.scale(X_train), columns = X_train.columns)
    # print(pd.DataFrame(pca.components_, columns=data_scaled.columns, index = ['PC-1','PC-2']))
    #
    return X_train

    # explained_variance = pca.explained_variance_ratio_
    # print(explained_variance)'''


def svm_poly_cv(X, y, stockName):
    model = svm.SVC(kernel='poly', degree=3)
    results = pd.DataFrame(
        cross_validate(model, X, y, cv=10, scoring=['accuracy', 'precision', 'recall', 'f1']))
    print(results)
    print(results.describe())

    results.plot.box(figsize=(10, 5), title="SVM Poly:" + stockName)
    plt.show()


def svm_rbf_cv(X, y, stockName):
    model = svm.SVC(kernel='rbf', degree=3)
    results = pd.DataFrame(
        cross_validate(model, X, y, cv=10, scoring=['accuracy', 'precision', 'recall', 'f1']))
    print(results)
    print(results.describe())

    results.plot.box(figsize=(10, 5), title="SVM RBF:" + stockName)
    plt.show()


aapl = pd.read_csv('./data/apple.csv')
amzn = pd.read_csv('./data/amazon.csv')

attribute1 = ['entity_pos_sent', 'entity_neg_sent', 'entity_net_sent']
attribute2 = ['entity_pos_sent',
 'entity_neg_sent',
 'entity_net_sent',
 'entity_pos_sent_minus_1',
 'entity_neg_sent_minus_1',
 'entity_net_sent_minus_1',
 'entity_pos_sent_minus_2',
 'entity_neg_sent_minus_2',
 'entity_net_sent_minus_2',
 'entity_pos_sent_minus_3',
 'entity_neg_sent_minus_3',
 'entity_net_sent_minus_3',
 'entity_pos_sent_minus_4',
 'entity_neg_sent_minus_4',
 'entity_net_sent_minus_4',
 'entity_pos_sent_minus_5',
 'entity_neg_sent_minus_5',
 'entity_net_sent_minus_5',
 'minus_1_pct_chng',
 'minus_2_pct_chng',
 'minus_3_pct_chng',
 'minus_4_pct_chng',
 'minus_5_pct_chng',
 'direction']

X = aapl[attribute1]
y = aapl['direction']
svm_poly_cv(X, y, 'aapl')
svm_rbf_cv(X, y, 'aapl')

X = amzn[attribute1]
y = amzn['direction']
svm_poly_cv(X, y, 'amzn')
svm_rbf_cv(X, y, 'amzn')

X = aapl[attribute2]
X = X.dropna()
y = X['direction']
X = X.drop('direction', axis=1)
svm_poly_cv(X, y, 'aapl')
svm_rbf_cv(X, y, 'aapl')

X = amzn[attribute2]
X = X.dropna()
y = X['direction']
X = X.drop('direction', axis=1)
svm_poly_cv(X, y, 'amzn')
svm_rbf_cv(X, y, 'amzn')
