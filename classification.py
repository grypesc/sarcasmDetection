import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

path = "./input/"
input = pd.read_csv(path + 'dataPostProcessed.csv', index_col=0)

from sklearn.model_selection import train_test_split
X = input['headline']#.append(df['headline'])
Y = input['is_sarcastic']#.append(df['is_sarcastic'])

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.01, random_state=42)
classifiers = [
    # Pipeline([('tfidf', TfidfVectorizer()), ('classifier', LinearSVC())]),
    Pipeline([('tfidf', TfidfVectorizer()), ('classifier', MLPClassifier(hidden_layer_sizes=(90,90), random_state=42, solver='lbfgs'))]),
    Pipeline([('tfidf', TfidfVectorizer()), ('classifier', MLPClassifier(hidden_layer_sizes=(20,20), random_state=42, solver='lbfgs'))]),
    Pipeline([('tfidf', TfidfVectorizer()), ('classifier', MLPClassifier(hidden_layer_sizes=(150,150, 150), random_state=42, solver='lbfgs'))]),
        Pipeline([('tfidf', TfidfVectorizer()), ('classifier', MLPClassifier(hidden_layer_sizes=(80,90, 100, 90, 80), random_state=42, solver='lbfgs'))]),
    # Pipeline([('tfidf', TfidfVectorizer()), ('classifier', RandomForestClassifier())]),
    # Pipeline([('tfidf', TfidfVectorizer()), ('classifier', KNeighborsClassifier(n_neighbors=2))]),
    ]

for classifier in classifiers:
    classifier.fit(xTrain, yTrain)
    print("###### " + classifier.named_steps.get("classifier").__class__.__name__ + " ############")
    print("\nTraining: " + str(classifier.score(xTrain, yTrain).round(3)))
    print("Test: " + str(classifier.score(xTest, yTest).round(3)) +"\n")
    prediction = classifier.predict(xTest)
    print(confusion_matrix(yTest, prediction))
    print(classification_report(yTest, prediction))
