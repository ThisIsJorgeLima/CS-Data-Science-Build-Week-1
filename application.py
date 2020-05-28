"""
import statements:
"""
# classics
import os
import pandas as pd

# sparse matrix
from scipy.sparse import csr_matrix

# algorithm
from sklearn.neighbors import NearestNeighbors

# string matching
from fuzzywuzzy import fuzz
from Levenshtein import *
from warnings import warn

# load and validate the data
data_path = '/Users/jorge/CS-Data-Science-Build-Week-1/data'
movies_path = '/Users/jorge/CS-Data-Science-Build-Week-1//data/movies.csv'
ratings_path = '/Users/jorge/CS-Data-Science-Build-Week-1//data/ratings.csv'
movies = pd.read_csv(
    os.path.join(data_path, movies_path),
    usecols=['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})
ratings = pd.read_csv(
    os.path.join(data_path, ratings_path),
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

# print(movies.shape)
# print(movies.head())
# print('+-------------------------------+')
# print(ratings.shape)
# print(ratings.head())
# print('+-------------------------------+')
# print(ratings.tail())

# making a pivot of our dataset into features
features = ratings.pivot(
    index='movieId',
    columns='userId',
    values='rating'
).fillna(0)
# Since very large matrices require a lot of memory, we want to scipy sparse and convert our dataframe of our features.
matrix_movie_features = csr_matrix(features.values)
# print('+-------------------------------+')
# print(movie_features.head())
# print('+-------------------------------+')
# now we will isolate films that have been rated 75 times.
# our rating frequency
# number of ratings each movie got.
movie_count = pd.DataFrame(ratings.groupby('movieId').size(), columns=['count'])
# print(movie_count.head())
#
# print('+-------------------------------+')

popularity_level = 75
popular_films = list(set(movie_count.query('count >= @popularity_level').index))
drop_films = ratings[ratings.movieId.isin(popular_films)]
# print('shape of  our  ratings data: ', ratings.shape)
# print('shape of our ratings data after dropping unpopular films: ', drop_films.shape)

# print('+-------------------------------+')
# get the number of ratings given by each user from  our data
user = pd.DataFrame(drop_films.groupby('userId').size(), columns=['count'])
# print(user.head())
#
# print('+-------------------------------+')
# filter data to come to an approximation of user likings.
ratings_level = 75
active_users = list(set(user.query('count >= @ratings_level').index))
drop_users = drop_films[drop_films.userId.isin(active_users)]
# print('shape of original ratings data: ', ratings.shape)
# print('shape of ratings data after dropping both unpopular movies and inactive users: ', drop_users.shape)

# print('+-------------------------------+')

# now we pivot and create our movie-user matrix
user_matrix = drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# here we are maping  our movie titles
movie_title_mapper = {
    movie: i for i, movie in
    enumerate(list(movies.set_index('movieId').loc[user_matrix.index].title))
}

# transform matrix to scipy sparse matrix
movie_user_matrix_sparse = csr_matrix(user_matrix.values)
# print(movie_user_matrix_sparse)
# print('+-------------------------------+')

# applying and defining our model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# fit
model_knn.fit(movie_user_matrix_sparse)


def fuzzy_matcher(mapper, favorite_movie, verbose=True):
    """

    We use fuzzy matcher to help get our ratio of movie title names that have been inputed to search through our database.

    By doing this it will return us the closest match via our fuzzy ratio, which will compare two strings and outputs our ratio.
    """
    match_tuple = []
    # geting our match
    for title, index in mapper.items():
        ratio = fuzz.ratio(title.lower(), favorite_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, index, ratio))
    # sorting
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Uh-Oh! Something went wrong on our end, please resubmit entry')
        return
    if verbose:
        print('Top ten similar matches: {0}\n'.format(
            [x[0] for x in match_tuple]))
    return match_tuple[0][1]


def recommendation(model_knn, data, mapper, favorite_movie, n_recommendations):
    """
    Now we can predict and return similar films based on our user's typed in movie title.

    Our model:
    ----------
    K-Nearest Neighbors Algorithm and sklearn model.
    data: movie-user matrix

    Our mapper: is our dictionary which maps through our movie title names in  our database.

    favorite_movie: str, name of user input movie
    n_recommendations: int, top n recommendations
    Return
    ------
    list of top n similar movie recommendations
    """
    # fit
    model_knn.fit(data)
    # get input movie index
    print('Film input:', favorite_movie)
    index = fuzzy_matcher(mapper, favorite_movie, verbose=True)

    print('Popular recommendations: ')
    print('.....\n')
    distances, indices = model_knn.kneighbors(data[index], n_neighbors=n_recommendations+1)

    raw_recommends = sorted(
        list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # reverse mapping and unflattening
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Here are more like {}:'.format(favorite_movie))
    for i, (index, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[index], dist))


type_will_predict_favorite_films = 'The Godfather: Part II'

recommendation(
    model_knn=model_knn,
    data=movie_user_matrix_sparse,
    favorite_movie=type_will_predict_favorite_films,
    mapper=movie_title_mapper,
    n_recommendations=10)
