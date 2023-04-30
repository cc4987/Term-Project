import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white', { 'axes.spines.right': False, 'axes.spines.top': False})
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords

#load data
df_meta=pd.read_csv('data/movies_metadata.csv', low_memory=False, encoding='UTF-8')
 
# remove invalid records
df_meta = df_meta.drop([19730, 29503, 35587])
# convert the id to type int and set id as index
df_meta = df_meta.set_index(df_meta['id'].str.strip().replace(',','').astype(int))
pd.set_option('display.max_colwidth', 20)
df_meta.head(2)
# load the movie credits
df_credits = pd.read_csv('data/credits.csv', encoding='UTF-8')
df_credits = df_credits.set_index('id')

# load the movie keywords
df_keywords=pd.read_csv('data/keywords.csv', low_memory=False, encoding='UTF-8') 
df_keywords = df_keywords.set_index('id')

# merge everything into a single dataframe 
df_k_c = df_keywords.merge(df_credits, left_index=True, right_on='id')
df = df_k_c.merge(df_meta[['release_date','genres','overview','title']], left_index=True, right_on='id')
df.head(3)

# create an empty DataFrame
df_movies = pd.DataFrame()

# extract the keywords
df_movies['keywords'] = df['keywords'].apply(lambda x: [i['name'] for i in eval(x)])
df_movies['keywords'] = df_movies['keywords'].apply(lambda x: ' '.join([i.replace(" ", "") for i in x]))

# extract the overview
df_movies['overview'] = df['overview'].fillna('')

# extract the release year 
df_movies['release_date'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

# extract the actors
df_movies['cast'] = df['cast'].apply(lambda x: [i['name'] for i in eval(x)])
df_movies['cast'] = df_movies['cast'].apply(lambda x: ' '.join([i.replace(" ", "") for i in x]))

# extract genres
df_movies['genres'] = df['genres'].apply(lambda x: [i['name'] for i in eval(x)])
df_movies['genres'] = df_movies['genres'].apply(lambda x: ' '.join([i.replace(" ", "") for i in x]))

# add the title
df_movies['title'] = df['title']

# merge fields into a tag field
df_movies['tags'] = df_movies['keywords'] + df_movies['cast']+' '+df_movies['genres']+' '+df_movies['release_date']

# drop records with empty tags and dublicates
df_movies.drop(df_movies[df_movies['tags']==''].index, inplace=True)
df_movies.drop_duplicates(inplace=True)

# add a fresh index to the dataframe, which we will later use when refering to items in a vector matrix
df_movies['new_id'] = range(0, len(df_movies))

# Reduce the data to relevant columns
df_movies = df_movies[['new_id', 'title', 'tags']]

# display the data
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.expand_frame_repr', False)
print(df_movies.shape)
df_movies.head(5)

# set a custom stop list from nltk
stop = list(stopwords.words('english'))

# create the tfid vectorizer, alternatively you can also use countVectorizer
# create TfidfVectorizer object with 'english' stopwords
tfidf = TfidfVectorizer(stop_words='english')

# fit and transform the data using the vectorizer
vectorized_data = tfidf.fit_transform(df_movies['tags'])
count_matrix = pd.DataFrame(vectorized_data.toarray(), index=df_movies['tags'].index.tolist())
print(count_matrix)
# helps with noise and model complexity
svd = TruncatedSVD(n_components=3000)
reduced_data = svd.fit_transform(count_matrix)
#end of external resources code - https://www.relataly.com/content-based-movie-recommender-using-python/4294/ 

#cosine function is used to calculate mathematical similarity values using movie descriptions as the vectors
    #similarity values work as follows:
    #-1 means the movies are entirely different
    #0 means they are independent
    #1 indicates an identical match

# compute the cosine similarity matrix
distance = pairwise_distances(reduced_data, metric='cosine')
similarity = 1 - distance
# generating content-based movie recs
def get_recs(title, n, cosine_sim=similarity):
    """gives recommendations for movies based on similarity scores"""
    # finds index of the movie to match the title
    movie_index = df_movies[df_movies.title==title].new_id.values[0]
    # finds pairwise similarity scores of movies with a particular movie, sorted by similarity score
    sim_scores_all = []
    for index, score in enumerate(cosine_sim[movie_index]):
        sim_scores_all.append((index, score))
    sim_scores_all.sort(key=lambda x: x[1], reverse=True)
    # checks if recommendations are limited
    if n > 0:
        sim_scores_all = sim_scores_all[1:n+1]
    #indices of top similar movies
    movie_indices = [i[0] for i in sim_scores_all]
    scores = [i[1] for i in sim_scores_all]
    #returns similar titles
    top_titles_df = df_movies.iloc[movie_indices][['title']]
    top_titles_df['sim_scores'] = scores
    top_titles_df['ranking'] = range(1, len(top_titles_df) + 1)

    return top_titles_df, sim_scores_all
# getting recs
movie_name = 'Spectre'
number_of_recs = 15
top_titles_df, _ = get_recs(movie_name, number_of_recs)
print(top_titles_df)

#visualization
sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top_titles_df, y='title', x='sim_scores', color='blue')
plt.xlim((0,1))
plt.xlabel('Similarity score')
plt.ylabel('Movie title')
plt.title(f'Here are {number_of_recs} similar recommendations to the movie {movie_name}')
plt.show(movie_name, top_titles_df)

# numpy.core._exceptions._ArrayMemoryError: Unable to allocate 83.1 GiB for an array with shape (45432, 245588) and data type float64