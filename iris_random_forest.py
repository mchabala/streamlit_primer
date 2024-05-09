from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd

def iris_random_forest(data, n_estimators, criterion, test_size):
    """Function to train random forest model on iris dataset

    :param data: the iris dataframe to train model on
    :param n_estimators: the number of trees in the forest
    :param criterion: the function to measure the quality of the split 
    :param test_size: the size of the test dataset
    
    :return: model object, model accuracy, feature importances
    """
    
    # define features and ouput column
    feature_names = ['Petal_width', 'Petal_length', 'Sepal_width', 'Sepal_length']
    X=data[feature_names]  # Features
    y=data['Species_name']  # Labels

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # create model
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    
    # train the model on the training dataset
    clf.fit(X_train, y_train)
    
    # perform predictions on the test dataset
    y_pred = clf.predict(X_test)
    
    # finding feature importances
    feature_imp = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    # return model, accuracy, and feature importances
    accuracy = metrics.accuracy_score(y_test, y_pred)
    
    return clf, accuracy, feature_imp
    
    