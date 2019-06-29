from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
path = "./input/"
input = pd.read_csv(path + 'dataPostProcessed.csv', index_col=0)
input['headline'] = input['headline'].str.replace('\'', '').str.replace(',', '').str.replace('[', '').str.replace(']', '')


from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(background_color='gray', stopwords = STOPWORDS,
                max_words = 500, max_font_size = 100,
                random_state = 17, width=1920, height=1080)

plt.figure(figsize=(16, 12))
wordcloud.generate(str(input.loc[input['is_sarcastic'] == 1, 'headline']))
plt.imshow(wordcloud);
plt.title("The most frequent words from sarcastic headlines")

plt.figure(figsize=(16, 12))
wordcloud.generate(str(input.loc[input['is_sarcastic'] == 0, 'headline']))
plt.imshow(wordcloud);
plt.title("The most frequent words from not sarcastic headlines")

plt.figure(figsize=(16, 12))
input['is_sarcastic'].value_counts().plot(kind='bar')
plt.ylabel('amount')
plt.xlabel('category')

plt.figure(figsize=(16, 12))
input.loc[input['is_sarcastic'] == 1, 'headline'].str.len().hist(label='Sarcastic', bins = 100, alpha=.5)
input.loc[input['is_sarcastic'] == 0, 'headline'].str.len().hist(label='Not sarcastic', bins = 100, alpha=.5)
print(input.loc[input['is_sarcastic'] == 1, 'headline'].str.len())
plt.title("Length of Headlines")
plt.legend()


import nltk, re #Removing stop words from input
stopwords = nltk.corpus.stopwords.words('english')
headlinesPP = [] # PP means post processed
for index, row in input.iterrows():
    tokenList = re.sub("[^\S]", " ",  row["headline"]).split()
    tokenListCleansed = ''
    for token in tokenList:
        if (token not in stopwords):
            tokenListCleansed += token + ' '
    headlinesPP.append(tokenListCleansed)
dataPP = pd.DataFrame(data = {'headline': headlinesPP, 'is_sarcastic': input["is_sarcastic"]} )

nonSarcastic = dataPP.loc[input['is_sarcastic'] == 0 ]
sarcastic = dataPP.loc[input['is_sarcastic'] == 1]

plt.figure(figsize=(16, 12))
nonSarcasticBar = pd.Series(' '.join(nonSarcastic['headline']).lower().split()).value_counts()[:10].plot(kind='bar',
                            title="Most Frequent Words of not Sarcastic Comments")
nonSarcasticBar.set_xlabel("Word")
nonSarcasticBar.set_ylabel("Count")

plt.figure(figsize=(16, 12))
sarcasticBar = pd.Series(' '.join(sarcastic['headline']).lower().split()).value_counts()[:10].plot(kind='bar',
                            title="Most Frequent Words of Sarcastic Comments")
sarcasticBar.set_xlabel("Word")
sarcasticBar.set_ylabel("Count")
plt.show()
