import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Importing the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# One Hot Encoding
categorical_features = ["Sex", "Pclass"]
def add_encoded_columns(df, column):
    dummies = pd.get_dummies(df[column], prefix = column)
    df = pd.concat([df, dummies], axis = 1)

    return df

for feature in categorical_features:
    train = add_encoded_columns(train, feature)
    test = add_encoded_columns(test, feature)

# Fill in the missing values
train["Fare"] = train["Fare"].fillna((train["Fare"].mean()))
test["Fare"] = test["Fare"].fillna((test["Fare"].mean()))
train["Age"] = train["Age"].fillna((train["Age"].mean()))
test["Age"] = test["Age"].fillna((test["Age"].mean()))

# Prepare train/test set.
columns = ["Fare", "Sex_female", "Sex_male", "Pclass_1", "Pclass_2", "Pclass_3", "Age"]
X_all = train[columns]
y_all = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = 0)

# Fit and predict on splitted train data.
# classifier = GradientBoostingClassifier()
classifier = LogisticRegression(random_state = 0, solver="lbfgs", max_iter = 1000)
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
submission.to_csv(f"predictions/titanic_one-hot-encoding.csv", index=False)
