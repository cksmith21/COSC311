from tkinter.tix import X_REGION
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

if __name__ == "__main__": 

    # read in data from txt file
    wifi_data = pd.read_csv("wifi_localization.txt", sep="\t", \
        names = ['ap0', 'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'type']) 
    print(wifi_data.info())
    
    # generate knn object

    knn = KNeighborsClassifier(n_neighbors = 3)
    dt = DecisionTreeClassifier(criterio='entropy', max_depth=5, min_samples_split=2)

    X = wifi_data[['ap0', 'ap1', 'ap2', 'ap3', 'ap4','ap5','ap6']].values
    Y = wifi_data['type'].values

    # fit to model 
    knn.fit(X, Y)
    dt.fit(X,Y)

    # self test accuracy score
    print('-'*30)
    print(f'Self test accuracy for KNN: {knn.score(X,Y)}.')
    print(f'Self test accuracy for DT: {dt.score(X,Y)}.')
    print('-'*30)

    # find the prediction for self test for KNN
    knn_predict = knn.predict(X)
    knn_report = classification_report(Y, knn_predict)

    print('-'*30)
    print("KNN CLassification Report: ")
    print(knn_report)
    print('-'*30)

    # for the prediction for self test for decision tree
   
    dt_pred = dt.predict(X)
    dt_report = classification_report(Y, dt_pred)

    print('-'*30)
    print("Decision Tree Classification Report")
    print(dt_report)
    print('-'*30)
    # independent test 

    X = wifi_data[['ap0', 'ap1', 'ap2', 'ap3', 'ap4','ap5','ap6']].values
    y = wifi_data['type'].values

    # split data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)

    # create objects

    knn_independent = KNeighborsClassifier(n_neighbors=5)
    dt_independent = DecisionTreeClassifier(max_depth=5)

    # train models

    knn_independent.fit(X_train, y_train)
    dt_independent.fit(X_train, y_train)

    # get predictions for the classification report

    y_knn_pred = knn_independent.predict(X_train)
    y_dt_pred = dt_independent.predict(X_train)

    knn_independent_report = classification_report(y_train, y_knn_pred)
    dt_independent_report = classification_report(y_train, y_dt_pred)

    print('-'*30)
    print("KNN Independent Classification Report")
    print(knn_independent_report) 
    print('-'*30)

    print('-'*30)
    print("DT Independent Classification Report")
    print(dt_independent_report)
    print('-'*30)

    # print out the scores

    print(knn_independent.score(X_train, y_train))
    print(dt_independent.score(X_train, y_train))

    # task three: independent test results for multiple values

    knn_independent_tests = KNeighborsClassifier(n_neighbors=3)

    cm = confusion_matrix(y_train, y_knn_pred)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix for KNN')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y)))
    plt.xticks(tick_marks, np.unique(y))
    plt.yticks(tick_marks, np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # 10% 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42, stratify=y)
    knn_independent_tests.fit(X_train, y_train)
    ten_percent = knn_independent_tests.score(X_train, y_train)

    # 20% 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    knn_independent_tests.fit(X_train, y_train)
    twenty_percent = knn_independent_tests.score(X_train, y_train)

    # 30% 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)
    knn_independent_tests.fit(X_train, y_train)
    thirty_percent = knn_independent_tests.score(X_train, y_train)

    # 40% 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42, stratify=y)
    knn_independent_tests.fit(X_train, y_train)
    fourty_percent = knn_independent_tests.score(X_train, y_train)

    # 50% 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=42, stratify=y)
    knn_independent_tests.fit(X_train, y_train)
    fifty_percent = knn_independent_tests.score(X_train, y_train)

    scores = {'Ten':ten_percent, 'Twenty':twenty_percent, 'Thirty':thirty_percent, 'Fourty':fourty_percent, 'Fifty':fifty_percent}
    categories = list(scores.keys())
    percent = list(scores.values())

    plt.bar(categories, percent)
    plt.show()