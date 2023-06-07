import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

data1 = pd.read_csv('genres(1).csv')
data2 = pd.read_csv('movies_tmdb_popular.csv')

data1.head()
data2.head()
data1.shape
data2.shape

data1.isnull().sum()
data2.isnull().sum()
data2.dropna(inplace = True)
data2.isnull().sum()
data1['tmp'] = 1
data2['tmp'] = 1

data = pd.merge(data1, data2, on=['tmp'])
data = data.drop('tmp', axis=1)
data.head()
data.isnull().sum()
data.duplicated().sum()
data.head()

pop= data.sort_values('popularity', ascending=False)
#plt.figure(figsize=(12,4))
plt.barh(pop['title'].head(6),pop['popularity'].head(6),align = 'center' , color='blue')
plt.gca().invert_yaxis()
plt.xlabel("popularity")
plt.title("Popular Movies")

data['overview'].head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
data['overview'] = data['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(data['overview'])
tfidf_matrix.shape

indices = pd.Series(data.index, index=data['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices]

get_recommendations('The Avengers')
