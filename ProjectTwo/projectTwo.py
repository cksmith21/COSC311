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
            Use the last column as the ground truth (y_true or labels in this code) to match 
        each cluster with its label, calculate and output the clustering accuracy and show the 
        corresponding confusion matrix as a figure. 
            Calculate and output the clustering accuracy of each room. 
    
    '''

    # Create dataframe from the wifi data

    wifi_data = pd.read_csv("wifi_localization.txt", sep="\t", \
        names = ['ap0', 'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'type']) 

    # Get X and Y --> data and y_true (labels)

    data = wifi_data[['ap0', 'ap1', 'ap2', 'ap3', 'ap4','ap5','ap6']].values
    y_true = wifi_data['type'].values

    # K-means object, creating clusters

    kmeans = KMeans(n_clusters=4,random_state=0)
    clusters = kmeans.fit_predict(data)

    # Getting labels and centers 

    labels = kmeans.labels_ 
    centers = kmeans.cluster_centers_

    # Plotting the centers 

    fig, ax = plt.subplots()
    scatter = ax.scatter(data[:,0], data[:,1], c=labels, cmap='viridis')
    centers = ax.scatter(centers[:, 0], centers[:, 1], marker='*', c='red', s=300)
    legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Cluster")
    plt.show()

    # Creating confusion matrix

    y_pred = kmeans.fit_predict(data)
    accuracy = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)

    # Plotting confusion matrix

    fig, ax = plt.subplots()
    im = ax.imshow(confusion, cmap='Blues')
    ax.set_xticks(np.arange(len(np.unique(labels+1))))
    ax.set_yticks(np.arange(len(np.unique(labels+1))))
    ax.set_xticklabels(np.unique(labels+1))
    ax.set_yticklabels(np.unique(labels+1))
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion matrix for Wi-Fi Signals')
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()

    # Calculating by room accuracy 

    by_room_accurary = [] 

    for i in range(4): 
        mask = (labels ==i)
        by_room_accurary.append(accuracy_score(labels[mask], y_pred[mask]))

    print("The accuracy by room is: ", end='')
    print(*by_room_accurary)

    
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