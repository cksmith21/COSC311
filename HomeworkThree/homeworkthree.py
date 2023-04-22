import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns

if __name__ == "__main__": 

    labels = [str(i) for i in range(1,66)]
    labels.append('type')

    df = pd.read_csv('FoodTypeDataset.csv',names=labels)

    X = df[labels[:-1]].values
    y = df['type'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    rfc_independent = RandomForestClassifier(n_estimators=75, max_depth=11, min_samples_split=2)

    rfc_independent.fit(X_train, y_train)
    y_rfc_pred = rfc_independent.predict(X_test)
    rfc_independent_report = classification_report(y_test, y_rfc_pred)

    print('-'*55)
  
    print("RFC Independent Classification Report")
    print(rfc_independent_report)
    print('-'*55)

    cm = confusion_matrix(y_test, y_rfc_pred)
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')   
    plt.title("Heatmap For Food Category Data")
    plt.show()