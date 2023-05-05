import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold 
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __name__ == "__main__": 

    '''
        TASK ONE: Conduct k-means clustering on the wi-fi data; output the center of each 
        cluster. 
            Use the last column as the ground truth (y_true) to match 
        each cluster with its label, calculate and output the clustering accuracy and show the 
        corresponding confusion matrix as a figure. 
            Calculate and output the clustering accuracy of each room. 
    
    '''

    # Create dataframe from the wifi data

    #wifi_data = pd.read_csv("wifi_localization.txt", sep="\t", header=None, names=['atr1', 'atr2', 'atr3', 'atr4', 'atr5', 'atr6', 'atr7', 'rooms'])
    wifi_data = pd.read_csv("wifi_localization.txt", sep='\t', header=None)
    X = wifi_data.iloc[:, :7].values
    y_true = wifi_data.iloc[:, -1:].values
   
    kmeans = KMeans(n_clusters=4, random_state=42).fit(X)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Get the predicted labels for each sample
    y_pred = (kmeans.labels_)+1

    print(*y_true)
    print(*y_pred)

    confusion = confusion_matrix(y_true, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion, display_labels = [1, 2, 3, 4])
    cm_display.plot()
    plt.show()
   
    '''
    # Get X and Y --> data and y_true (labels)

    data = wifi_data[['atr1', 'atr2', 'atr3', 'atr4', 'atr5', 'atr6', 'atr7']].values
    y_true = wifi_data['rooms'].values-1

    print(y_true)

    # K-means object, creating clusters

    kmeans = KMeans(n_clusters=4,random_state=42).fit(data)
    #y_pred = kmeans.fit_predict(data)
    #print(y_pred)

    # Getting labels and centers 

    labels = kmeans.labels_ 
    centers = kmeans.cluster_centers_
    print(f'The centers are {centers}.')

    # Creating confusion matrix

    accuracy = accuracy_score(y_true, labels)
    confusion = confusion_matrix(y_true, labels)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion, display_labels = [1, 2, 3, 4])
    cm_display.plot()
    plt.show()


    # Plotting confusion matrix

    fig, ax = plt.subplots()
    im = ax.imshow(confusion, cmap='Blues')
    tick_labels = np.unique(y_true)
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion matrix for Wi-Fi Signals')
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()

    # Calculating by room accuracy 

    by_room_accurary = [] 

    for i in np.unique(y_true): 
        mask = (y_true ==i)
        by_room_accurary.append(accuracy_score(y_true[mask], labels[mask]))

    print("The accuracy by room is: ", end='')
    print(*by_room_accurary)

    '''

    '''
        TASK TWO: Conduct a PCA analysis on the digits dataset and find out how many 
            principal components are need to keep at least 90% variance
        Assume m principal components are needed to keep at least 90% variance, 
            transform the dataset from 64 dimensions to m dimensions.
        Based on the above dimension-reduced dataset, build a classification model with 
            optimized parameters to do a cross-validatioon test (CVT) with fold = 10, show
            the CVT accuracy and corresponding confusion matrix in a figure. 

    '''
    # Load digits 

    digits = load_digits()

    # Create PCA

    pca = PCA()
    X_transformed = pca.fit_transform(digits.data)
    Y = digits.target

    # Compute the cumulative variance ratio 

    cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the number of components necessary

    n_components = np.argmax(cumulative_var_ratio >= 0.9) + 1

    # Tranform the dataset

    X_transformed_m = PCA(n_components=n_components).fit_transform(digits.data)
    print(f"Shape of the transformed dataset:{X_transformed_m.shape}.")

    # Split the data 

    X_train, X_test, y_train, y_test = train_test_split(X_transformed_m, digits.target, test_size=0.2, random_state=42)

    # Create and train the KNN 

    knn = KNeighborsClassifier(n_neighbors = 5, algorithm="brute", weights="uniform")
    knn.fit(X_train, y_train) 
    print("Train score after PCA", knn.score(X_train,y_train), "%")
    print("Test score after PCA", knn.score(X_test,y_test), "%")

    # Create KFold with 10 folds 

    folds=10
    kf= KFold(n_splits=folds, random_state=None)

    # Perform cross validation 

    result = cross_val_score(knn , X_transformed, digits.target, cv = kf)
    print(f"Average accuracy: {result.mean()}.")

    # Predict the Y

    y_knn_pred = knn.predict(X_train)

    # Plot the confusion matrix 

    cm = metrics.confusion_matrix(y_train, y_knn_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()

    '''
        TASK THREE: Assume the 'median_house_value' is related to the following 
            attributes: 'housing_median_age', 'total_rooms', 'total_bedrooms', 
            'population', and 'median_income', use correlatioon coefficient analysis
            to select three attributes that have higher correlation alues with the target
            variable.
        Randomly split all the samples (all samples include the three selected attributes
            and one target variable) split into two parts: 60% for training and 40% for 
            testing. 
        Use the training data to build a Multiple Linear Regression model and test it
            using the testing data. Show the performance of the regression model, including 
            MAE, MSE, RMSE. 
    '''

    # Creating dataframe and correlating the data

    housing_df = pd.read_csv('housing.csv')
    corrdat = housing_df.corr(numeric_only=True)

    # Find the top features

    feature = []
    value = []
    
    corr_matrix = housing_df[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'median_income', 'median_house_value']].corr()
    top_features = corr_matrix.nlargest(4, 'median_house_value')['median_house_value'].index[1:4]

    # Create the training and testing data based off the new dataframe

    X_train, X_test, y_train, y_test = train_test_split(housing_df[top_features], housing_df['median_house_value'], test_size=0.4, random_state=42)

    # Create the linear regression model 

    reg = LinearRegression().fit(X_train, y_train)

    # Find the predicted value

    y_pred = reg.predict(X_test)

    # Use the regression performance metrics and print them out

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")