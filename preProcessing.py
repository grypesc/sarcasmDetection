import pandas as pd
import string, re, nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def getPOS(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

path = "./input/"
input = pd.read_json(path + 'Sarcasm_Headlines_Dataset.json', lines=True)
input = input.drop(columns = ["article_link"])

lemmatizer = WordNetLemmatizer()
headlinesPP = [] # PP means post processed
for index, row in input.iterrows():
    row["headline"] = row["headline"].translate(str.maketrans('', '', string.punctuation)) #Removing punctuation characters
    row["headline"] = row["headline"].translate(str.maketrans('', '', string.digits)) #Removing digits
    tokenList = re.sub("[^\S]", " ",  row["headline"]).split() #Spliting to tokens, can be also tokenList = nltk.word_tokenize(sentence)
    for i in range(0, len(tokenList)): #lemmatization
        tokenList[i] = lemmatizer.lemmatize(tokenList[i], getPOS(tokenList[i]))
    headlinesPP.append(tokenList)

dataPP = pd.DataFrame(data = {'headline': headlinesPP, 'is_sarcastic': input["is_sarcastic"]} )
dataPP.to_csv(path+'dataPostProcessed.csv')
