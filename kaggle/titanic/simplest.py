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

# Prepare train/test set.
columns = ["Fare", "Sex", "Pclass"]
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
submission.to_csv(f"predictions/titanic_simplest.csv", index=False)
