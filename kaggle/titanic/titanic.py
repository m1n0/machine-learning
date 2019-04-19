import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

import pickle

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

# Feature engineering
## Family size.
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

## Title
def titles_in_name(name: str, titles: list):
    for title in titles:
        if title in name:
            return title
    return np.nan

title_list=["Mrs", "Mr", "Master", "Miss", "Major", "Rev",
            "Dr", "Ms", "Mlle","Col", "Capt", "Mme", "Countess",
            "Don", "Jonkheer"]
train["Title"] = train["Name"].map(lambda x: titles_in_name(x, title_list))
test["Title"] = test["Name"].map(lambda x: titles_in_name(x, title_list))

def categorize_titles(person):
    title = person["Title"]

    if title in ["Don", "Major", "Capt", "Jonkheer", "Rev", "Col"]:
        return "Mr"
    elif title in ["Countess", "Mme"]:
        return "Mrs"
    elif title in ["Mlle", "Ms"]:
        return "Miss"
    elif title in ["Dr"]:
        if person["Sex"] == "Male":
            return "Mr"
        else:
            return "Mrs"
    else:
        return title

train["Title"] = train.apply(categorize_titles, axis=1)
test["Title"] = test.apply(categorize_titles, axis=1)

# Encode categorical values
def add_encoded_columns(df, column):
    dummies = pd.get_dummies(df[column], prefix = column)
    df = pd.concat([df, dummies], axis = 1)

    return df

categorical_features = ["AgeCategory", "Sex", "Embarked", "Pclass", "Title"]

for feature in categorical_features:
    train = add_encoded_columns(train, feature)
    test = add_encoded_columns(test, feature)

# Make sure there are no missing values
train["Fare"] = train["Fare"].fillna((train["Fare"].mean()))
test["Fare"] = test["Fare"].fillna((test["Fare"].mean()))

# Prepare train/test set.
columns = ["Fare", "AgeCategory_Infant", "AgeCategory_Child", "AgeCategory_Teenager", "AgeCategory_YoungAdult", "AgeCategory_Adult", "AgeCategory_Senior", "Sex_female", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S", "Pclass_1", "Pclass_2", "Pclass_3", "FamilySize", "Title_Mr", "Title_Mrs", "Title_Miss", "Title_Master"]
X_all = train[columns]
y_all = train["Survived"]

# Prepare classifiers.
classifiers = {
    "Logistic Regression": LogisticRegression(random_state = 0, solver="lbfgs", max_iter = 10000),
    "KNN": KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2),
    "SVM": SVC(kernel = "linear", random_state = 0),
    "Kernel SVM": SVC(kernel = "rbf", random_state = 0),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(criterion = "entropy", random_state = 0),
    "Random Forest": RandomForestClassifier(criterion = "entropy", n_estimators = 100, random_state = 0),
    "Gradient Boost": GradientBoostingClassifier()
}

holdout = test
holdout_predictions = {}

best_accuracy = 0
best_model = None

# Fit, predict and output.
for type, classifier in classifiers.items():
    print(f"\n--- {type} ---")
    scores = cross_val_score(classifier, X_all, y_all, cv = 10)
    accuracy = np.mean(scores)
    min = np.min(scores)
    max = np.max(scores)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = (type, classifier)

    print(f"\nAccuracy: {accuracy}\nMin: {min}\nMax: {max}\n")
    print(f"Fitting on all data, predicting test data...\n")

    classifier.fit(X_all, y_all)
    holdout_predictions[type] = classifier.predict(holdout[columns])

    holdout_ids = holdout["PassengerId"]
    submission_df = {"PassengerId": holdout_ids,
                     "Survived": holdout_predictions[type]}

    submission = pd.DataFrame(submission_df)
    submission.to_csv(f"predictions/titanic_{type}.csv", index=False)

print(f"\n...creating models and calculating predictions done.")

print(f"\n\nSaving best model for later use:")
print(f"\n{best_model[0]}")

pickle.dump(
    best_model[1],
    open(f"model/{best_model[0].lower().replace(' ', '_')}_classifier.model",
    "wb")
)
