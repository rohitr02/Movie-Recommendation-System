import csv
import math

from eval_metrics import get_average_precision_and_recall, get_f_measure, get_discounted_cumulative_gain, get_ideal_discounted_cumulative_gain, get_average_normalized_discounted_cumulative_gain

# Given a list of movie id's, map each id to a unique index and return the dictionary
def map_movie_id_to_index() -> dict:
    # Create a dictionary to map movie id's to indices
    movie_id_to_index = {}

    # Store original movie id's
    movie_ids = []

    # Read the movies.csv file
    with open('movies.csv', 'r', newline='') as movies_file:
        movies_read = csv.reader(movies_file, delimiter=',')
        # Iterate through the rows
        for row in movies_read:
            # Ignore the first row
            if row[0] == 'movieId':
                continue
            # Append the movie id to the list
            movie_ids.append(row[0])

    index = 0
    # Iterate through the movie id's
    for id in movie_ids:
        # If the movie id is not in the dictionary, add it
        if id not in movie_id_to_index.keys():
            movie_id_to_index[id] = index
            index += 1
    # Return the dictionary
    return movie_id_to_index

# Map genre to a unique index
def map_genre_to_index() -> dict:
    # Create a dictionary to map genres to indices
    genre_to_index = {}

    index = 0
    # Read the movies.csv file
    with open('movies.csv', 'r', newline='') as movies_file:
        movies_read = csv.reader(movies_file, delimiter=',')
        # Iterate through the rows
        for row in movies_read:
            # Ignore the first row
            if row[0] == 'movieId':
                continue
            # Iterate through the genres
            for genre in row[2].split('|'):
                # If the genre is not in the dictionary, add it
                if genre not in genre_to_index.keys():
                    genre_to_index[genre] = index
                    index += 1
    # Return the dictionary
    return genre_to_index

# Map all the different genres to the number of movies that have that genre
def get_num_of_movies_per_genre(genre_to_index) -> dict:
    # Create a dictionary to map genres to number of movies
    genre_to_number_of_movies = {}

    # Read the movies.csv file
    with open('movies.csv', 'r', newline='') as movies_file:
        movies_read = csv.reader(movies_file, delimiter=',')
        # Iterate through the rows
        for row in movies_read:
            # Ignore the first row
            if row[0] == 'movieId':
                continue
            # Iterate through the genres
            for genre in row[2].split('|'):
                # Get the index of the genre
                genre = genre_to_index[genre]
                # If the genre is not in the dictionary, add it
                if genre not in genre_to_number_of_movies.keys():
                    genre_to_number_of_movies[genre] = 1
                else:
                    # Increment the number of movies
                    genre_to_number_of_movies[genre] += 1
    # Return the dictionary
    return genre_to_number_of_movies

# Map movie index to a list of genres associated with the movie
def map_movie_index_to_genre_index(genre_to_index, movie_id_to_index) -> dict:
    # Create a dictionary to map movie index's to genres
    movie_index_to_genres = {}

    # Read the movies.csv file
    with open('movies.csv', 'r', newline='') as movies_file:
        movies_read = csv.reader(movies_file, delimiter=',')
        # Iterate through the rows
        for row in movies_read:
            # Ignore the first row
            if row[0] == 'movieId':
                continue
            movie_index = movie_id_to_index[row[0]]
            # Iterate through the genres
            for genre in row[2].split('|'):
                # Get the index of the genre
                genre = genre_to_index[genre]
                # If the movie id is not in the dictionary, add it
                if movie_index not in movie_index_to_genres:
                    movie_index_to_genres[movie_index] = [genre]
                else:
                    # Append the genre to the list
                    movie_index_to_genres[movie_index].append(genre)
    # Return the dictionary
    return movie_index_to_genres

# Calculate the term frequency of a movie
def get_term_frequency(genre_index, movie_index, movie_index_to_genre_index) -> float:
    # Get the list of genre indexes associated with the movie
    genres = movie_index_to_genre_index[movie_index]
    # Calculate the term frequency
    term_frequency = genres.count(genre_index) / len(genres)
    # Return the term frequency
    return term_frequency

# Calculate the inverse document frequency of a movie
def get_inverse_document_frequency(genre_index, genre_to_number_of_movies, total_num_of_movies) -> float:
    # Get the number of movies that have the genre
    num_of_movies_in_genre = genre_to_number_of_movies[genre_index]
    # Calculate the inverse document frequency
    inverse_document_frequency = math.log(total_num_of_movies / num_of_movies_in_genre)
    # Return the inverse document frequency
    return inverse_document_frequency

# Calculate the tf-idf of a movie given genre and movie-index
def get_tf_idf(genre_index, movie_index, movie_index_to_genre_index, genre_to_number_of_movies, total_num_of_movies) -> float:
    # Get the term frequency
    term_frequency = get_term_frequency(genre_index, movie_index, movie_index_to_genre_index)
    # Get the inverse document frequency
    inverse_document_frequency = get_inverse_document_frequency(genre_index, genre_to_number_of_movies, total_num_of_movies)
    # Calculate the tf-idf
    tf_idf = term_frequency * inverse_document_frequency
    # Return the tf-idf
    return tf_idf

# Get the tf-idf vector of a movie-id
def get_tf_idf_vector(movie_id, genre_to_index, movie_id_to_index, movie_index_to_genre_index, genre_to_number_of_movies, total_num_of_movies) -> list:
    # Create a list to store the tf-idf vector
    tf_idf_vector = []
    # Get the index of the movie
    movie_index = movie_id_to_index[movie_id]
    # Iterate through the genres
    for genre in genre_to_index:
        # Get the tf-idf of the movie given the genre
        tf_idf = get_tf_idf(genre_to_index[genre], movie_index, movie_index_to_genre_index, genre_to_number_of_movies, total_num_of_movies)
        # Append the tf-idf to the list
        tf_idf_vector.append(tf_idf)
    # Return the tf-idf vector
    return tf_idf_vector

# Calculate the cosine similarity of two tf-idf vectors
def get_cosine_similarity(tf_idf_vector_1, tf_idf_vector_2) -> float:
    # Calculate the dot product
    dot_product = 0
    for i in range(len(tf_idf_vector_1)):
        dot_product += tf_idf_vector_1[i] * tf_idf_vector_2[i]
    # Calculate the magnitude of the first vector
    magnitude_1 = 0
    for i in range(len(tf_idf_vector_1)):
        magnitude_1 += tf_idf_vector_1[i] ** 2
    magnitude_1 = math.sqrt(magnitude_1)
    # Calculate the magnitude of the second vector
    magnitude_2 = 0
    for i in range(len(tf_idf_vector_2)):
        magnitude_2 += tf_idf_vector_2[i] ** 2
    magnitude_2 = math.sqrt(magnitude_2)
    # Calculate the cosine similarity
    cosine_similarity = dot_product / (magnitude_1 * magnitude_2)
    # Return the cosine similarity
    return cosine_similarity

# Get map of user index to list of tuples of (movie id, rating) in the training.csv file
def map_train_user_id_to_movie_id_list() -> dict:
    # Create a dictionary to map user id's to movie id's
    user_id_to_movie_id_list = {}
    # Read the training.csv file
    with open('training.csv', 'r', newline='') as file:
        read = csv.reader(file, delimiter=',')
        # Iterate through the rows
        for row in read:
            user_id = row[0]
            movie_id = row[1]
            rating = row[2]
            # Ignore the first row
            if row[0] == 'userId':
                continue
            # If the user id is not in the dictionary, add it
            if user_id not in user_id_to_movie_id_list:
                user_id_to_movie_id_list[user_id] = [(movie_id, rating)]
            else:
                # Append the movie id to the list
                user_id_to_movie_id_list[user_id].append((movie_id, rating))
    # Return the dictionary
    return user_id_to_movie_id_list

# Build a user profile for a user using weighted mean of the user’s ratings and the TF-IDF vector
def build_user_profile(user_id, user_id_to_movie_id_list, movie_id_to_index, movie_index_to_genre_index, genre_to_index, genre_to_number_of_movies, total_num_of_movies) -> list:
    # Create a list to store the user profile
    user_profile = [0.0] * len(genre_to_index)
    # Get the list of tuples of (movie id, rating) for the user
    tuple_list = user_id_to_movie_id_list[user_id]
    # Store total ratings
    total_ratings = 0
    # Iterate through the movies
    for tuple in tuple_list:
        # Get the movie id
        movie_id = tuple[0]
        # Get the rating
        rating = float(tuple[1])
        # Add to total ratings
        total_ratings += rating
        # Get the tf-idf vector of the movie
        tf_idf_vector = get_tf_idf_vector(movie_id, genre_to_index, movie_id_to_index, movie_index_to_genre_index, genre_to_number_of_movies, total_num_of_movies)
        # Calculate the weighted mean
        weighted_vector = list(map(lambda x : x * rating, tf_idf_vector))
        # Add the weighted vector to the user profile
        for i in range(len(user_profile)):
            user_profile[i] += weighted_vector[i]
    # Calculate the mean by dividing each entry in the user profile by the total ratings
    for i in range(len(user_profile)):
        user_profile[i] /= total_ratings
    # Return the user profile
    return user_profile

# Map user_id to user profile for every user
def get_user_profiles(user_id_to_movie_id_list, movie_id_to_index, movie_index_to_genre_index, genre_to_index, genre_to_number_of_movies, total_num_of_movies) -> list:
    # Create a list to store the user profiles
    user_profiles = {}
    # Iterate through the user id's
    for user_id in user_id_to_movie_id_list:
        if user_id not in user_profiles:
            # Add the user profile to the dictionary
            user_profiles[user_id] = build_user_profile(user_id, user_id_to_movie_id_list, movie_id_to_index, movie_index_to_genre_index, genre_to_index, genre_to_number_of_movies, total_num_of_movies)
    # Return the list of user profiles
    return user_profiles

# Iterate through all the movies in movies.csv and return a dictionary mappying movie-id to tf-idf vector for each movie
def get_movie_id_to_tf_idf_vector_map(movie_id_to_index, movie_index_to_genre_index, genre_to_index, genre_to_number_of_movies, total_num_of_movies) -> dict:
    # Create a dictionary to map movie id's to tf-idf vectors
    movie_id_to_tf_idf_vector_map = {}
    # Read the movies.csv file
    with open('movies.csv', 'r', newline='') as file:
        read = csv.reader(file, delimiter=',')
        # Iterate through the rows
        for row in read:
            movie_id = row[0]
            # Ignore the first row
            if row[0] == 'movieId':
                continue
            # Get the tf-idf vector of the movie
            tf_idf_vector = get_tf_idf_vector(movie_id, genre_to_index, movie_id_to_index, movie_index_to_genre_index, genre_to_number_of_movies, total_num_of_movies)
            # Add the tf-idf vector to the dictionary
            movie_id_to_tf_idf_vector_map[movie_id] = tf_idf_vector
    # Return the dictionary
    return movie_id_to_tf_idf_vector_map

# Get tf-idf vectors for all movies a user has not rated. return a list of tuples of (movie id, tf-idf vector)
def get_tf_idf_vectors_for_unrated_movies_for_user(user_id, user_id_to_movie_id_list, movie_id_to_index, movie_id_to_tf_idf_vector_map) -> list:
    # Create a list to store the tf-idf vectors
    movie_id_and_tf_idf_vector_list = []
    # Get the list of tuples of (movie id, rating) for the user
    tuple_list = user_id_to_movie_id_list[user_id]
    # Get the list of movie ids the user has rated
    movie_id_list = []
    for tuple in tuple_list:
        movie_id_list.append(tuple[0])
    # Get the list of movie ids the user has not rated
    unrated_movie_id_list = []
    for movie_id in movie_id_to_index:
        if movie_id not in movie_id_list:
            unrated_movie_id_list.append(movie_id)
    # Iterate through the unrated movies
    for movie_id in unrated_movie_id_list:
        # Get the tf-idf vector of the movie
        tf_idf_vector = movie_id_to_tf_idf_vector_map[movie_id]
        # Append the (movie-id, tf-idf) vector to the list
        movie_id_and_tf_idf_vector_list.append((movie_id, tf_idf_vector))
    # Return the list
    return movie_id_and_tf_idf_vector_list

# Get a dictionary mapping user-index to list(movie-id, get_cosine_similarity(tf-idf_vector, user_profile))
def get_all_recommendations(user_profiles, user_id_to_movie_id_list, movie_id_to_index, movie_id_to_tf_idf_vector_map) -> dict:
    # Create a dictionary to map user-index to (movie-id, get_cosine_similarity(tf-idf_vector, user_profile))
    recommendations = {}

    for user_id in user_profiles:
        # Get the user profile
        user_profile = user_profiles[user_id]
        # Get list of (movie-id, tf-idf vectors) for all movies a user has not rated
        tuple_list = get_tf_idf_vectors_for_unrated_movies_for_user(user_id, user_id_to_movie_id_list, movie_id_to_index, movie_id_to_tf_idf_vector_map)
        # Iterate through the tf-idf vectors
        for movie_id, tf_idf_vector in tuple_list:
            # Get the cosine similarity between the user profile and the tf-idf vector
            cosine_similarity = get_cosine_similarity(user_profile, tf_idf_vector)
            # Add the (movie-id, cosine_similarity) to the dictionary
            user_id = int(user_id) 
            movie_id = int(movie_id)
            if user_id not in recommendations.keys():
                recommendations[user_id] = [(movie_id, cosine_similarity)]
            else:
                recommendations[user_id].append((movie_id, cosine_similarity))
    # Return the dictionary
    return recommendations

# Output all the recommendations to cbm_recommendations.csv
def output_recommendations(recommendations, movie_id_to_index):
    # Open the cbm_recommendations.csv file
    with open('cbm_recommendations.csv', 'w', newline='') as file:
        write = csv.writer(file, delimiter=',')
        # Iterate through the user-index to (movie-id, cosine_similarity) list
        for user_id in recommendations:
            # Iterate through the list of (movie-id, cosine_similarity)
            for movie_id_and_cosine_similarity in recommendations[user_id]:
                # Get the movie id
                movie_id = movie_id_and_cosine_similarity[0]
                # Get the cosine similarity
                cosine_similarity = movie_id_and_cosine_similarity[1]
                # Get the movie index
                movie_index = movie_id_to_index[str(movie_id)]
                user_index = user_id-1
                # Write the row
                write.writerow([user_index, movie_index, cosine_similarity])

# Get the top N recommendations for each user
def get_top_N_cbm_recommendations(recommendations, N) -> list:
    # Create a dict to store the top N recommendations for each user
    top_N_recommendations = {}
    # Iterate through the recommendations
    for user_id in recommendations:
        # Sort the recommendations for the user in descending order
        recommendations[user_id].sort(key=lambda x: x[1], reverse=True)
        # Add the top N recommendations to the dictionary
        user_id = int(user_id)
        top_N_recommendations[user_id] = recommendations[user_id][:N]
    # Return the dictionary
    return top_N_recommendations

# Map every user in testing.csv to the to the movie-id's they have rated
def map_test_user_index_to_movie_ids():
    # Initialize the variables
    user_to_movie_ids = {}

    # Read the testing.csv file and store the movie ids for each user in a dictionary
    with open('testing.csv', 'r', newline='') as testing_file:
        testing_read = csv.reader(testing_file, delimiter=',')
        # Iterate through all the rows in testing.csv to get the testing ratings
        for row in testing_read:
            # The index for the user
            user_index = int(row[0])
            # The movie id
            movie_id = int(row[1])

            # If the user index is not in the dictionary
            if user_index not in user_to_movie_ids.keys():
                # Add the movie id to the dictionary
                user_to_movie_ids[user_index] = [movie_id]
            else:
                # Add the movie id to the dictionary
                user_to_movie_ids[user_index].append(movie_id)
    
    return user_to_movie_ids

# Output the top recommendations list for each user
def output_top_recommendations(recommendations, movie_id_to_index):
    # Open the output file
    with open('cbm_top_list.csv', 'w', newline='') as output_file:
        # Create a csv writer
        output_writer = csv.writer(output_file, delimiter=',')
        # Iterate through the recommendations
        for user_id in recommendations:
            # Get the top N recommendations for the user
            top_N_recommendations = recommendations[user_id]
            # Iterate through the top N recommendations
            for recommendation in top_N_recommendations:
                # Get the movie id 
                movie_id = recommendation[0]
                # cosine_similarity = recommendation[1]

                user_index = int(user_id)-1
                movie_index = movie_id_to_index[str(movie_id)]

                # Write the movie id and the cosine similarity to the output file
                output_writer.writerow([user_index, movie_index])

# Output the top N recommendations list for each user
def output_top_N_recommendations(recommendations, movie_id_to_index, N):
    # Open the output file
    with open('data/cbm_top_list' + str(N) + '.csv', 'w', newline='') as output_file:
        # Create a csv writer
        output_writer = csv.writer(output_file, delimiter=',')
        # Iterate through the recommendations
        for user_id in recommendations:
            # Get the top N recommendations for the user
            top_N_recommendations = recommendations[user_id]
            # Iterate through the top N recommendations
            for recommendation in top_N_recommendations:
                # Get the movie id 
                movie_id = recommendation[0]
                # cosine_similarity = recommendation[1]

                user_index = int(user_id)-1
                movie_index = movie_id_to_index[str(movie_id)]

                # Write the movie id and the cosine similarity to the output file
                output_writer.writerow([user_index, movie_index])

def test_for_cbm_metrics():
    # Map movie id's to indices
    movie_id_to_index = map_movie_id_to_index()
    # Map genre to a unique index
    genre_to_index = map_genre_to_index()
    # Map all the different genres to the number of movies that have that genre
    genre_to_number_of_movies = get_num_of_movies_per_genre(genre_to_index)
    # Map movie index to a list of genres associated with the movie
    movie_index_to_genre_index = map_movie_index_to_genre_index(genre_to_index, movie_id_to_index)

    # Get map of user index to list of tuples of (movie id, rating)
    user_id_to_movie_id_list = map_train_user_id_to_movie_id_list()

    # Get the movie-ids that each user in testing.csv has rated
    test_user_to_movie_ids = map_test_user_index_to_movie_ids()

    # Get the user profiles for all users
    user_profiles = get_user_profiles(user_id_to_movie_id_list, movie_id_to_index, movie_index_to_genre_index, genre_to_index, genre_to_number_of_movies, len(movie_id_to_index))
    # print(user_profiles['1'])

    # Get movie id to tf-idf vector map
    movie_id_to_tf_idf_vector_map = get_movie_id_to_tf_idf_vector_map(movie_id_to_index, movie_index_to_genre_index, genre_to_index, genre_to_number_of_movies, len(movie_id_to_index))

    # Get all the recommendations
    all_recommendations = get_all_recommendations(user_profiles, user_id_to_movie_id_list, movie_id_to_index, movie_id_to_tf_idf_vector_map)

    # Output the recommendations to cbm_recommendations.csv
    output_recommendations(all_recommendations, movie_id_to_index)

    map_N_to_eval_metrics = {}

    # Iterate through the different N we are testing
    for N in [10, 15, 25, 50, 100, 500, 1000]:
        # Get the top 10 recommendations
        recommendations = get_top_N_cbm_recommendations(all_recommendations, N)

        # Output the top recommendations
        output_top_N_recommendations(recommendations, movie_id_to_index, N)

        average_precision, average_recall = get_average_precision_and_recall(recommendations, test_user_to_movie_ids)
        f_measure = get_f_measure(average_precision, average_recall)
        ndcg = get_average_normalized_discounted_cumulative_gain(recommendations, test_user_to_movie_ids)

        map_N_to_eval_metrics[N] = (average_precision, average_recall, f_measure, ndcg)
    
    return map_N_to_eval_metrics


def main():
    # Map movie id's to indices
    movie_id_to_index = map_movie_id_to_index()
    # Map genre to a unique index
    genre_to_index = map_genre_to_index()
    # Map all the different genres to the number of movies that have that genre
    genre_to_number_of_movies = get_num_of_movies_per_genre(genre_to_index)
    # Map movie index to a list of genres associated with the movie
    movie_index_to_genre_index = map_movie_index_to_genre_index(genre_to_index, movie_id_to_index)

    # Get map of user index to list of tuples of (movie id, rating)
    user_id_to_movie_id_list = map_train_user_id_to_movie_id_list()

    # Get the user profiles for all users
    user_profiles = get_user_profiles(user_id_to_movie_id_list, movie_id_to_index, movie_index_to_genre_index, genre_to_index, genre_to_number_of_movies, len(movie_id_to_index))
    # print(user_profiles['1'])

    # Get movie id to tf-idf vector map
    movie_id_to_tf_idf_vector_map = get_movie_id_to_tf_idf_vector_map(movie_id_to_index, movie_index_to_genre_index, genre_to_index, genre_to_number_of_movies, len(movie_id_to_index))

    # Get all the recommendations
    all_recommendations = get_all_recommendations(user_profiles, user_id_to_movie_id_list, movie_id_to_index, movie_id_to_tf_idf_vector_map)

    # Output the recommendations to cbm_recommendations.csv
    output_recommendations(all_recommendations, movie_id_to_index)

    # Get the top 10 recommendations
    recommendations = get_top_N_cbm_recommendations(all_recommendations, 10)

    # Output the top recommendations
    output_top_recommendations(recommendations, movie_id_to_index)

    # Get the movie-ids that each user in testing.csv has rated
    test_user_to_movie_ids = map_test_user_index_to_movie_ids()

    average_precision, average_recall = get_average_precision_and_recall(recommendations, test_user_to_movie_ids)
    print('Average Precision:', average_precision)
    print('Average Recall:', average_recall)

    f_measure = get_f_measure(average_precision, average_recall)
    print('F-Measure:', f_measure)

    dcg = 0
    idcg = 0
    for user in recommendations:
        dcg += get_discounted_cumulative_gain(user, recommendations, test_user_to_movie_ids)
        idcg += get_ideal_discounted_cumulative_gain(user, recommendations, test_user_to_movie_ids)
    print('Average Discounted Cumulative Gain:', dcg/len(recommendations))
    print('Average Ideal Discounted Cumulative Gain:', idcg/len(recommendations))
    print('Average Normalized Discounted Cumulative Gain:', get_average_normalized_discounted_cumulative_gain(recommendations, test_user_to_movie_ids))

if __name__ == '__main__':
    main()