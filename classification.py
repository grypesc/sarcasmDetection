import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

path = "./input/"
input = pd.read_csv(path + 'dataPostProcessed.csv', index_col=0)

from sklearn.model_selection import train_test_split
X = input['headline']#.append(df['headline'])
Y = input['is_sarcastic']#.append(df['is_sarcastic'])

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.1, random_state=42)
classifiers = [
    #SVC = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())]),
    Pipeline([('tfidf', TfidfVectorizer()), ('clf', MLPClassifier(hidden_layer_sizes=(300,200), random_state=42, warm_start=True, solver='adam'))]),
    #RFC = Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier())]),
    ]


for classifier in classifiers:
    classifier.fit(xTrain, yTrain)
    print("### " + classifier.__class__.__name__ + " ###")
    print("Classification accuracy: " + str(classifier.score(xTest, yTest)) +"\n")
    prediction = classifier.predict(xTest)
    print(confusion_matrix(yTest, prediction))
    print(classification_report(yTest, pred))
