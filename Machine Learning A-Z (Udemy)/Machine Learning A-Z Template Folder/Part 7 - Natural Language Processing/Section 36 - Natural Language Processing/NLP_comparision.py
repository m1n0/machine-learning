# Natural Language Processing

# Importing the libraries
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
import classification_helper
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
#nltk.download('stopwords')

ps = PorterStemmer()
corpus = []

for i in range(0, dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

### Fitting different models
cms = []

# Logistic
classifier = LogisticRegression(random_state = 0)
cms.append({"type": "Logistic", "cm": classification_helper.classify(classifier, X, y)})

# KNN
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
cms.append({"type": "KNN", "cm": classification_helper.classify(classifier, X, y)})

# SVM
classifier = SVC(kernel = 'linear', random_state = 0)
cms.append({"type": "SVM", "cm": classification_helper.classify(classifier, X, y)})

# Kernel SVM RBF
classifier = SVC(kernel = 'rbf', random_state = 0)
cms.append({"type": "Kernel SVM RBF", "cm": classification_helper.classify(classifier, X, y)})

# Kernel SVM Sigmoid
classifier = SVC(kernel = 'sigmoid', random_state = 0)
cms.append({"type": "Kernel SVM Sigmoid", "cm": classification_helper.classify(classifier, X, y)})

# Kernel SVM Poly
classifier = SVC(kernel = 'poly', random_state = 0)
cms.append({"type": "Kernel SVM Poly", "cm": classification_helper.classify(classifier, X, y)})

# Naive Bayes
classifier = GaussianNB()
cms.append({"type": "Naive Bayes", "cm": classification_helper.classify(classifier, X, y)})

# Random forest
classifier = RandomForestClassifier(n_estimators = 100,
                                    criterion = 'entropy',
                                    random_state = 0)
cms.append({"type": "Random Forest", "cm": classification_helper.classify(classifier, X, y)})

# C5.0
classifier = DecisionTreeClassifier()
cms.append({"type": "C5.0", "cm": classification_helper.classify(classifier, X, y)})

results = []
### Compare the results
print("\n\nPrinting the results: \n")
for result in cms:
    performance = classification_helper.calculate_performance_indicators(result['cm'])
    results.append({"type": result["type"], "performance": performance})
    classification_helper.print_output(result["type"], performance)
    
    