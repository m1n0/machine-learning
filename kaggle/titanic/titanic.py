import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Importing the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Age categories.
def process_age(df, cut_points, label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["AgeCategory"] = pd.cut(df["Age"], cut_points, labels=label_names)

    return df

cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
label_names = ["Missing", "Infant", "Child", "Teenager", "YoungAdult", "Adult", "Senior"]

train = process_age(train, cut_points, label_names)
test = process_age(test, cut_points, label_names)

# Family size.
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

# Encode categorical values
def add_encoded_columns(df, column):
    dummies = pd.get_dummies(df[column], prefix = column)
    df = pd.concat([df, dummies], axis = 1)

    return df

categorical_features = ["AgeCategory", "Sex", "Embarked", "Pclass"]

for feature in categorical_features:
    train = add_encoded_columns(train, feature)
    test = add_encoded_columns(test, feature)

# Make sure there are no missing values
train['Fare'] = train['Fare'].fillna((train['Fare'].mean()))
test['Fare'] = test['Fare'].fillna((test['Fare'].mean()))

# Prepare train/test set.
columns = ["Fare", "AgeCategory_Child", "AgeCategory_Teenager", "AgeCategory_YoungAdult", "AgeCategory_Adult", "AgeCategory_Senior", "Sex_female", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S", "Pclass_1", "Pclass_2", "Pclass_3", "FamilySize"]
X_all = train[columns]
y_all = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = 0)

# Prepare classifiers.
classifiers = {
    "Logistic Regression": LogisticRegression(random_state = 0, solver="lbfgs", max_iter = 10000),
    "KNN": KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),
    "SVM": SVC(kernel = 'linear', random_state = 0),
    "Kernel SVM": SVC(kernel = 'rbf', random_state = 0),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
    "Random Forest": RandomForestClassifier(criterion = 'entropy', n_estimators = 100, random_state = 0),
    "Gradient Boost": GradientBoostingClassifier()
}

holdout = test
holdout_predictions = {}

# Fit, predict and output.
for type, classifier in classifiers.items():
    print(f"\n--- {type} ---")
    scores = cross_val_score(classifier, X_all, y_all, cv = 10)
    accuracy = np.mean(scores)
    min = np.min(scores)
    max = np.max(scores)

    print(f"\nAccuracy: {accuracy}\nMin: {min}\nMax: {max}\n")
    print(f"Fitting on all data, predicting test data...\n")

    classifier.fit(X_all, y_all)
    holdout_predictions[type] = classifier.predict(holdout[columns])

    holdout_ids = holdout["PassengerId"]
    submission_df = {"PassengerId": holdout_ids,
                     "Survived": holdout_predictions[type]}

    submission = pd.DataFrame(submission_df)
    submission.to_csv(f"titanic_{type}.csv", index=False)
