import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import seaborn as sns

if __name__ == "__main__": 

    labels = ['atr_'+str(i) for i in range(1,66)]
    labels.append('type')

    df = pd.read_csv('FoodTypeDataset.csv',names=labels)

    X = df[labels[0:65]].values
    y = df['type'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=40, stratify=y)
    knn_indepedent = KNeighborsClassifier(n_neighbors=20)
    dt_independent = DecisionTreeClassifier(max_depth=11, min_samples_split=2)
    
    dt_independent.fit(X_train, y_train)
    knn_indepedent.fit(X_train, y_train)
    y_dt_pred = dt_independent.predict(X_test)

    y_knn_pred = knn_indepedent.predict(X_test)
    dt_independent_report = classification_report(y_test, y_dt_pred)
    knn_independent_report = classification_report(y_test, y_knn_pred)

    print('-'*55)
    print("DT Independent Classification Report")
    print(dt_independent_report)
    print('-'*55)

    print("KNN Independent Classification Report")
    print(knn_independent_report)
    print('-'*55)

    cm = confusion_matrix(y_test, y_knn_pred)
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')   
