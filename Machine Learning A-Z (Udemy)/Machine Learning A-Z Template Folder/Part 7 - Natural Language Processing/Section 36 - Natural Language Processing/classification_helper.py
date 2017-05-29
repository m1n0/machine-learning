def classify(classifier, X, y):
    """
    Conducts fit, predict and computes confusion matrix on
    a given classifier and provided data.
    """
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import confusion_matrix
    
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return cm
    

def calculate_performance_indicators(cm):
    """
    Calculates classification performance based on provided
    confusion matrix.
    """
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[1][0]
    FN = cm[0][1]
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
def print_output(name, performance):
    """
    Outputs stats for a given model
    """
    print("Type: {0}".format(name));
    print("Accuracy: {0:.2f}".format(performance["accuracy"]))
    print("Precision: {0:.2f}".format(performance["precision"]))
    print("Recall: {0:.2f}".format(performance["recall"]))
    print("F1 Score: {0:.2f}".format(performance["f1"]))
    print()
    

