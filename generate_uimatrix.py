import pandas as pd
import numpy as np


def load_data(file_path):
    data_set = file_path.split('/')[-2]
    if data_set == 'ml-100k':
        data = pd.read_csv(file_path + 'u.data', sep='\t', names=['uid', 'mid', 'rating', 'time'])
        rating = np.zeros((943, 1682), dtype=np.float32)
        for index, row in data.iterrows():
            row_id = int(row['uid']) - 1
            col_id = int(row['mid']) - 1
            rating[row_id][col_id] = row['rating']
        np.save('/home/jyfan/data/MoiveLens/ml-100k/rating.npy', rating)
        return rating

    elif data_set == 'ml-1m':
        ratings_df = pd.read_csv(file_path + 'ratings.dat',
                                 sep='::',
                                 names=['userId', 'movieId', 'rating', 'Time'])
        movies_df = pd.read_csv(file_path + 'movies.dat',
                                sep='::',
                                names=['movieId', 'name', 'type'])

    elif data_set == 'ml-latest-small':
        ratings_df = pd.read_csv(file_path + 'ratings.csv')
        movies_df = pd.read_csv(file_path + 'movies.csv')
    else:
        print("ERROR: data file NOT FOUND")
        return None

    user_num = ratings_df['userId'].drop_duplicates().size
    num_movies = movies_df.index.size
    movie_id_mapping = {}
    id_count = 1
    for i in movies_df['movieId']:
        movie_id_mapping[int(i)] = int(id_count)
        id_count += 1

    rating = np.zeros((user_num, num_movies), dtype=np.float32)
    for index, row in ratings_df.iterrows():
        row_id = int(row['userId']) - 1
        col_id = movie_id_mapping[int(row['movieId'])] - 1
        rating[row_id][col_id] = row['rating']
    np.save(file_path + 'rating.npy', rating)
    return rating
