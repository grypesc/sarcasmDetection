import pandas as pd

path = "./input/"
input = pd.read_json(path + 'Sarcasm_Headlines_Dataset.json', lines=True)
input = input.drop(columns = ["article_link"])

import string, re, nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag)

lemmatizer = WordNetLemmatizer()
headlinesPP = [] # PP means post processed
for index, row in input.iterrows():
    row["headline"] = row["headline"].translate(str.maketrans('', '', string.punctuation)) #Removing punctuation characters
    row["headline"] = row["headline"].translate(str.maketrans('', '', string.digits)) #Removing digits
    tokenList = re.sub("[^\S]", " ",  row["headline"]).split() #Spliting to tokens, can be also tokenList = nltk.word_tokenize(sentence)
    for token in tokenList: #lemmatization
        lemmatizer.lemmatize(token, get_wordnet_pos(token))
    headlinesPP.append(tokenList)

dataPP = pd.DataFrame(data = {'headline': headlinesPP, 'is_sarcastic': input["is_sarcastic"]} )
print (dataPP)
