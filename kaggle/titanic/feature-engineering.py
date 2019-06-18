import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Importing the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Encode Sex string to numeric
train["Sex"] = pd.factorize(train["Sex"])[0]
test["Sex"] = pd.factorize(test["Sex"])[0]

# Fill in the missing values
train["Fare"] = train["Fare"].fillna((train["Fare"].mean()))
test["Fare"] = test["Fare"].fillna((test["Fare"].mean()))
train["Age"] = train["Age"].fillna((train["Age"].mean()))
test["Age"] = test["Age"].fillna((test["Age"].mean()))

# Feature engineering

# Age categories - not used as it lowers performance.
# cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
# label_names = ["Missing", "Infant", "Child", "Teenager", "YoungAdult", "Adult", "Senior"]
#
# def process_age(df, cut_points, label_names):
#     df["Age"] = df["Age"].fillna(-0.5)
#     df["AgeCategory"] = pd.cut(df["Age"], cut_points, labels=label_names)
#
#     return df
#
# train = process_age(train, cut_points, label_names)
# test = process_age(test, cut_points, label_names)
# train["AgeCategory"] = pd.factorize(train["AgeCategory"])[0]                    
# test["AgeCategory"] = pd.factorize(test["AgeCategory"])[0]  


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
train["Title"] = pd.factorize(train["Title"])[0]
test["Title"] = pd.factorize(test["Title"])[0]

# Prepare train/test set.
columns = ["Fare", "Sex", "Pclass", "Age", "FamilySize", "Title"]
X_all = train[columns]
y_all = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = 0)

# Fit and predict on splitted train data.
classifier = LogisticRegression(random_state = 0, solver="lbfgs", max_iter = 100)
classifier.fit(X_train, y_train)
prediction = prediction = classifier.predict(X_test)
cm = confusion_matrix(y_test, prediction)

print(f"Confusion matrix:\n{cm}")
print(f"Accuracy: {(cm[1][1] + cm[0][0]) / len(y_test)}")

# Fit and predict on all data.
classifier.fit(X_all, y_all)
prediction = classifier.predict(test[columns])

# Prepare the submission data
submission_df = {"PassengerId": test["PassengerId"],"Survived": prediction}
submission = pd.DataFrame(submission_df)
submission.to_csv(f"predictions/titanic_feature_engineering.csv", index=False)
