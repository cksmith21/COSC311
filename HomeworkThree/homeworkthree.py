import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

if __name__ == "__main__": 

    labels = ['atr_'+str(i) for i in range(1,66)]
    labels.append('type')

    df = pd.read_csv('FoodTypeDataset.csv',names=labels)

    X = df[labels[0:65]].values
    y = df['type'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    dt_independent = DecisionTreeClassifier(max_depth=11, criterion="entropy", min_samples_split=2)
    
    dt_independent.fit(X_train, y_train)
    y_dt_pred = dt_independent.predict(X_test)
    dt_independent_report = classification_report(y_test, y_dt_pred)

    print('-'*55)
    print("DT Independent Classification Report")
    print(dt_independent_report)
    print('-'*55)
