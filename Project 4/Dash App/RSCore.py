import pandas as pd
import numpy as np
import requests

#=================================================================
# Load Movies
# Define the URL for movie data
movie_url = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"
# Fetch the data from the URL
response_movie = requests.get(movie_url)

# Split the data into lines and then split each line using "::"
movie_lines = response_movie.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)
genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)

#-----------------------------------------------------------------
# Load ratings
rating_url = "https://liangfgithub.github.io/MovieData/ratings.dat?raw=true"
response_rating = requests.get(rating_url)
rating_lines = response_rating.text.split('\n')
rating_data = [line.split("::") for line in rating_lines if line]
ratings = pd.DataFrame(rating_data,columns=['user_id', 'movie_id', 'rating', 'Timestamp'])
ratings = ratings.astype('int32')

#==================================================================
# System 1
#group by movie_id, find number of rating users and average rating


grouped_ratings = ratings.groupby('movie_id').agg({'user_id': 'nunique', 'rating': 'mean'}).reset_index()
grouped_ratings.rename(columns={'user_id': 'num_users', 'rating': 'average_rating'},inplace = True)

movies['genre'] = movies['genres'].str.split('|')
exploded_genres_df = movies.explode('genre')
grouped_genre = pd.merge(grouped_ratings, exploded_genres_df, on='movie_id', how = 'left')
average_users_per_genre = grouped_genre.groupby('genre')['num_users'].mean().reset_index()
average_users_per_genre = average_users_per_genre.rename(columns={'num_users': 'average_num_users'})
grouped_genre = pd.merge(grouped_genre, average_users_per_genre, on='genre')
grouped_genre['lower_bound_users'] = grouped_genre['average_num_users'].astype(int)


def get_top_freq_movies(group):
    top_movies = group.nlargest(10, 'num_users')['title']
    return '|'.join(top_movies.astype(str))
def get_top_rated_movies(group):
    qualified_movies = group[group['num_users'] >= group['lower_bound_users']]
    return qualified_movies.nlargest(10, 'average_rating')['title']
top_freq_bygenre = grouped_genre.groupby('genre').apply(get_top_freq_movies).reset_index(name='top_movies')
top_rate_bygenre = grouped_genre.groupby('genre').apply(get_top_rated_movies).reset_index(name='top_movies')


#-----------------------------------------------------------------

def get_displayed_movies():
    return movies.head(100)

def get_popular_movies(genre: str):  # method = 'Freq'
    np.random.seed(42)
    method = np.random.choice(['Freq','Rate'])
    if method == 'Freq':
        movielists =  top_freq_bygenre[top_freq_bygenre['genre']==genre]['top_movies'].iloc[0].split('|')
    if method == 'Rate':
        movielists = top_rate_bygenre[top_rate_bygenre['genre']==genre]['top_movies'].iloc[0].split('|')
    movielists = pd.DataFrame(movielists,columns=['title'])
    return movielists.merge(movies,on='title',how = 'left')

#==================================================================
# System II

rating_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')

row_means = rating_matrix.apply(lambda row: row.mean(skipna=True), axis=1)

centered_rating_matrix = rating_matrix.sub(row_means, axis=0)
def compute_similarity(arr1, arr2):
    nan_mask = (~np.isnan(arr1))&(~np.isnan(arr2))
    arr1_m = arr1[nan_mask]
    arr2_m = arr2[nan_mask]
    n, = arr1_m.shape
    if n <= 2:
        return np.nan
    else:
        return 0.5+0.5*np.dot(arr1_m, arr2_m) / (np.linalg.norm(arr1_m) * np.linalg.norm(arr2_m))

#------------------------------------------------------------------
from tqdm import tqdm
# compute the similarity matrix
S = np.zeros([3706,3706])

crm = centered_rating_matrix.to_numpy()

# S = np.zeros([3706,3706])
# for i in tqdm(range(3706)):
#     for j in range(i,3706):
#         arr1 = crm[:,i]
#         arr2 = crm[:,j]
#         S[i][j] = S[j][i] = compute_similarity(arr1, arr2)    
        
# for i in range(3706):
#     S[i][i] = np.nan

from scipy.sparse import load_npz
filename = 'S_sparse_matrix.npz'
S = pd.DataFrame(load_npz(filename).toarray())
S.columns = rating_matrix.columns
S.index = rating_matrix.columns

#S = pd.read_csv('S_df.csv')
#S = S.drop('MovieID',axis = 1)


# Select top 30 ratings in each row and set the remaining to be nan
S_0 = np.nan_to_num(S, nan=0)
top_30_indices = np.argsort(S_0, axis=1)[:, -30:]
S_top = np.full_like(S_0, fill_value=np.nan, dtype=np.float64)

for i in range(S_0.shape[0]):
    S_top[i, top_30_indices[i]] = S_0[i, top_30_indices[i]]
S_top[S_top == 0] = np.nan

S_top_df = pd.DataFrame(S_top, index = centered_rating_matrix.columns, columns = centered_rating_matrix.columns)
#------------------------------------------------------------------


def get_recommended_movies(w):
    dic = w
    w = np.full(rating_matrix.shape[1], np.nan)
    for key, value in dic.items():
        w[key - 1] = value  # Adjusting for 0-based indexing in arrays

    w1 = np.nan_to_num(w, nan = 0)
    mask = ~np.isnan(w)
    w2 = mask.astype(int)
    S = np.nan_to_num(S_top_df.to_numpy(), nan = 0)
    rating = S@w1/(S@w2)
    rating[~np.isnan(w)] = np.nan
    df = pd.DataFrame(np.nan_to_num(rating, nan = 0), index = rating_matrix.columns)\
    .reset_index().rename(columns = {0:'rating'})
    df_candidate = df.nlargest(10,'rating').reset_index().drop('index',axis = 1)
    df_candidate = df_candidate[df_candidate['rating']>0]
    df_candidate = df_candidate.merge(movies, on='movie_id',how = 'left')[['movie_id','title','genres','rating']]
    l = len(df_candidate)
    if l < 10:
        return pd.concat([df_candidate, top10.iloc[:10-l,:]],axis=0)
    else:
        return df_candidate

#==================================================================
