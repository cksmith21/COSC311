import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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

    knn = KNeighborsClassifier(n_neighbors = 5, algorithm="brute", weights="uniform")
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=9, min_samples_split=2)

    X = wifi_data[['ap0', 'ap1', 'ap2', 'ap3', 'ap4','ap5','ap6']].values
    Y = wifi_data['type'].values

    # fit to model 
    knn.fit(X, Y)
    dt.fit(X,Y)

    # self test accuracy score
    print('-'*55)
    print(f'Self test accuracy for KNN: {knn.score(X,Y)}.')
    print(f'Self test accuracy for DT: {dt.score(X,Y)}.')

    '''
        Self Test Accuracy with: 
            3 neighbors: 0.9915
            5 neighbors: 0.9915 ***
            7 neighbors: 0.9895
        Self Test Accuracy with: 
            max_depth 3: 0.971
            max_depth 5: 0.986
            max_depth 7: 0.99 ***

        I ran tests varying the number of neighbors for KNN and the max_depth for the DT in order to determine
        the best combination of parameters to acheive the highest self test accuracy. I also branched out 
        into other parameters, including class weight, algorithm, and min_sample_splits to see if they would change
        my results. I found that the best value for the min_sample_splits was 2. The other parameters did not seem
        to change the self test results. For KNN, I found that the brute algorithm and uniform weights worked the best. 

    '''

    # find the prediction for self test for KNN
    knn_predict = knn.predict(X)
    knn_report = classification_report(Y, knn_predict)

    print('-'*55)
    print("KNN CLassification Report: ")
    print(knn_report)
    print('-'*55)

    # for the prediction for self test for decision tree
   
    dt_pred = dt.predict(X)
    dt_report = classification_report(Y, dt_pred)

    print("Decision Tree Classification Report")
    print(dt_report)
    print('-'*55)

    # independent test -- TASK TWO

    X = wifi_data[['ap0', 'ap1', 'ap2', 'ap3', 'ap4','ap5','ap6']].values
    y = wifi_data['type'].values

    # split data - using 30% of data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)

    # create objects

    knn_independent = KNeighborsClassifier(n_neighbors = 5, algorithm="brute", weights="uniform")
    dt_independent = DecisionTreeClassifier(max_depth=9, criterion="entropy", min_samples_split=2)

    '''
        I used the same models for the self test and the independent test because I found that the combination
        of parameters was optimal for both scenarios. 
    
        For KNN, the average f1 score was 0.99 on the model, which is very good. The average recall and precision 
        were also .99, indicating that the model used was accurate in deciding the data. Fine tuning the model to hit
        1.00 accuracy was not possible with the given in the independent test, so it is feasible to use the self test model. 
        When running the testing data, the model had a score of .98, which is relatively high considering the scores in
        other areas. 

        For decision trees, the model was even more accurate. The f1, recall, and precision all had an average of 1.00,
        which is immensely accurate. This demonstrates that the model was chosen properly for the dataset - even with
        the independent tests - so it did not have to be tweaked. When running the testing data, the result was slightly 
        less accurate with a score of .97, however that value is not low enough to consider that the model may have been 
        overfitted. Instead, anyone reading those scores can assume that the model was well-fitted to the data.
    
    '''

    # train models

    knn_independent.fit(X_train, y_train)
    dt_independent.fit(X_train, y_train)

    # find the scores for the testing data and the training data

    print(f'KNN Independent Test training data: {knn_independent.score(X_train, y_train)}.')
    print(f'KNN Independent Test test data: {knn_independent.score(X_test, y_test)}.')
    print('-'*55)
    print(f'DT Independent Test training data: {dt_independent.score(X_train, y_train)}.')
    print(f'DT Independent Test test data: {dt_independent.score(X_test, y_test)}.')

    # get predictions for the classification report

    y_knn_pred = knn_independent.predict(X_train)
    y_dt_pred = dt_independent.predict(X_train)

    knn_independent_report = classification_report(y_train, y_knn_pred)
    dt_independent_report = classification_report(y_train, y_dt_pred)

    print('-'*55)
    print("KNN Independent Classification Report")
    print(knn_independent_report) 
    print('-'*55)
    print("DT Independent Classification Report")
    print(dt_independent_report)
    print('-'*55)

    # print out the scores

    print(knn_independent.score(X_test, y_test))
    print(dt_independent.score(X_test, y_test))

    # task three: independent test results for multiple values

    knn_independent_tests = KNeighborsClassifier(n_neighbors=5)

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
    ten_percent = knn_independent_tests.score(X_test, y_test)

    # 20% 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    knn_independent_tests.fit(X_train, y_train)
    twenty_percent = knn_independent_tests.score(X_test, y_test)

    # 30% 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)
    knn_independent_tests.fit(X_train, y_train)
    thirty_percent = knn_independent_tests.score(X_test, y_test)

    # 40% 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42, stratify=y)
    knn_independent_tests.fit(X_train, y_train)
    fourty_percent = knn_independent_tests.score(X_test, y_test)

    # 50% 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=42, stratify=y)
    knn_independent_tests.fit(X_train, y_train)
    fifty_percent = knn_independent_tests.score(X_test, y_test)

    scores = {'Ten':ten_percent, 'Twenty':twenty_percent, 'Thirty':thirty_percent, 'Fourty':fourty_percent, 'Fifty':fifty_percent}
    categories = list(scores.keys())
    percent = list(scores.values())

    plt.bar(categories, percent)
    plt.title("Percent Training Data vs. Accuracy")
    plt.xlabel("Percent Training Data")
    plt.ylabel("Accuracy")
    plt.show()