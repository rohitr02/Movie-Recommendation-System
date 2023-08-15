import csv
import math
import random

from eval_metrics import *

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
        if id not in movie_id_to_index:
            movie_id_to_index[id] = index
            index += 1
    # Return the dictionary
    return movie_id_to_index

# Initializes p and q matrices for latent factor model
def initialize_p_and_q(num_factors, movie_id_to_index, init_strategy):
    # Stores the number of users
    num_users = 610

    # Initialize p matrix
    p_matrix = [[0] * num_factors for i in range(num_users)]
    # Populate p matrix with random values
    for i in range(len(p_matrix)):
        for j in range(len(p_matrix[0])):
            if init_strategy == 'zeros': p_matrix[i][j] = 0
            elif init_strategy == 'ones': p_matrix[i][j] = 1
            else: p_matrix[i][j] = random.random()
    
    # Initialize q matrix
    q_matrix = [[0] * num_factors for i in range(len(movie_id_to_index))]
    # Populate q matrix with random values
    for i in range(len(q_matrix)):
        for j in range(len(q_matrix[0])):
            if init_strategy == 'zeros': q_matrix[i][j] = 0
            elif init_strategy == 'ones': q_matrix[i][j] = 1
            else: q_matrix[i][j] = random.random()
    
    return p_matrix, q_matrix

# Uses SGD to update p and q matrices
def use_SGD(p_matrix, q_matrix, num_factors, movie_id_to_index, iterations):
    # Holds the value of the learning rate
    learning_rate = 0.01

    # For loop to iterate until convergence
    for i in range(iterations):
        # Read the training.csv
        with open('training.csv', 'r', newline='') as training_file:
            training_read = csv.reader(training_file, delimiter=',')
            # Iterate through all the rows in training.csv to get the training ratings
            for row in training_read:
                # The actual rating for the user and movie
                actual_rating = float(row[2])
                # The index for the user
                user_index = int(row[0]) - 1
                # The index for the movie
                movie_index = movie_id_to_index[row[1]]

                # Calculate the common value used when updating p and q
                common_value = 2 * (actual_rating - calculate_pq(p_matrix, q_matrix, user_index, movie_index, num_factors))

                # Update the p matrix
                common_q = multiply_value_and_row(common_value, q_matrix, movie_index, num_factors)
                vector_p = multiply_value_and_row(0.1, p_matrix, user_index, num_factors)
                subtracted_vectors_p = subtract_two_vectors(common_q, vector_p, num_factors)
                learning_rate_p = multiply_value_and_vector(learning_rate, subtracted_vectors_p)
                update_p_or_q(p_matrix, user_index, learning_rate_p, num_factors)

                # Update the q matrix
                common_p = multiply_value_and_row(common_value, p_matrix, user_index, num_factors)
                vector_q = multiply_value_and_row(0.1, q_matrix, movie_index, num_factors)
                subtracted_vectors_q = subtract_two_vectors(common_p, vector_q, num_factors)
                learning_rate_q = multiply_value_and_vector(learning_rate, subtracted_vectors_q)
                update_p_or_q(q_matrix, movie_index, learning_rate_q, num_factors)

# Multiplies each value in a row with the given value
def multiply_value_and_row(value, matrix, row_index, num_factors):
    # Holds the resulting vector
    result = [0] * num_factors
    # Iterate through the row and multiply each value with the given value
    for i in range(num_factors):
        result[i] = matrix[row_index][i] * value
    
    return result

# Calculates pq for a given user_index and movie_index
def calculate_pq(p_matrix, q_matrix, user_index, movie_index, num_factors):
    # Holds the current value of pq
    pq_value = 0
    # Iterate through the user's row in p and the movie's row in q
    for i in range(num_factors):
        pq_value += p_matrix[user_index][i] * q_matrix[movie_index][i]
    
    return pq_value

# Subtracts two vectors
def subtract_two_vectors(first, second, num_factors):
    # Holds the resulting vector
    result = [0] * num_factors
    # Iterate through the vectors and subtract them
    for i in range(num_factors):
        result[i] = first[i] - second[i]
    
    return result

# Multiplies each value in the vector with the given value
def multiply_value_and_vector(value, vector):
    # Holds the resulting vector
    result = [0] * len(vector)
    # Iterate through the vector and multiply with the given value
    for i in range(len(vector)):
        result[i] = vector[i] * value
    
    return result

# Update the row in the matrix by adding the given vector
def update_p_or_q(matrix, row_index, vector, num_factors):
    # Iterate through the row in the matrix
    for i in range(num_factors):
        matrix[row_index][i] += vector[i]

# Given a user-id and a movie-id, predict the rating
def predict_rating(p_matrix, q_matrix, user_index, movie_index, num_factors):
    # Calculate the predicted rating
    predicted_rating = calculate_pq(p_matrix, q_matrix, user_index, movie_index, num_factors)
    # Return the predicted rating
    return predicted_rating

# Go through testing.csv and using the user-id and movie-id, predict the rating
def make_predictions(p_matrix, q_matrix, num_factors, movie_id_to_index):
    # List to hold (user-id, movie-id, actual-rating, predicted-rating)
    predictions = []

    # Read the testing.csv
    with open('testing.csv', 'r', newline='') as testing_file:
        testing_read = csv.reader(testing_file, delimiter=',')
        # Iterate through all the rows in testing.csv to get the testing ratings
        for row in testing_read:
            # The index for the user
            user_index = int(row[0]) - 1
            # The original index for the movie
            orig_movie_index = int(row[1])
            # The index for the movie
            movie_index = movie_id_to_index[row[1]]
            # The actual rating
            actual_rating = float(row[2])
            # Calculate the predicted rating
            predicted_rating = predict_rating(p_matrix, q_matrix, user_index, movie_index, num_factors)
            # Append the predicted rating to the list
            predictions.append((user_index+1, orig_movie_index, actual_rating, predicted_rating))
    
    return predictions

# Output the predictions to a csv file called predictions.csv
def output_predictions(predictions):
    # Write the predictions to a csv file
    with open('predictions.csv', 'w', newline='') as predictions_file:
        predictions_write = csv.writer(predictions_file, delimiter=',')
        # Iterate through the predictions and write them to the csv file
        for row in predictions:
            predictions_write.writerow(row)

# Calculate the RMSE for the given predictions
def calculate_rmse(predictions):
    # Holds the sum of the squared errors
    sum_squared_errors = 0
    # Iterate through the predictions and calculate the sum of the squared errors
    for row in predictions:
        sum_squared_errors += (row[3] - row[2]) ** 2
    
    # Calculate the RMSE
    rmse = math.sqrt(sum_squared_errors / len(predictions))
    # Return the RMSE
    return rmse

# Calculate the MAE for the given predictions
def calculate_mae(predictions):
    # Holds the sum of the absolute errors
    sum_absolute_errors = 0
    # Iterate through the predictions and calculate the sum of the absolute errors
    for row in predictions:
        sum_absolute_errors += abs(row[3] - row[2])
    
    # Calculate the MAE
    mae = sum_absolute_errors / len(predictions)
    # Return the MAE
    return mae

# Get all recommendations for every user, output all recommendations, and return the dictionary.
def get_all_recommendations(movie_id_to_index, p_matrix, q_matrix, num_factors):
    # Store mapping between user and a list of (movie-id, predicted rating) tuples
    recommendations = {}

    # Store mapping of movies that have already been rated by a user
    movies_rated = {}

    # Read the training.csv file and store the movie indexes already rated for each user
    with open('training.csv', 'r', newline='') as file:
        read = csv.reader(file, delimiter=',')
        # Iterate through all the rows to get ratings
        for row in read:
            # The index for the user
            user_index = int(row[0]) - 1
            # The index for the movie
            movie_index = movie_id_to_index[row[1]]

            if user_index not in movies_rated.keys():
                movies_rated[user_index] = [movie_index]
            else:
                movies_rated[user_index].append(movie_index)

    # Go through the values of movie_id_to_index and if the value is not in movies_rated[user_index], add (movie_index, pred_rating) to recommendations[user_index]
    for movie_index in movie_id_to_index.values():
        for user_index in movies_rated.keys():
            if movie_index not in movies_rated[user_index]:
                if user_index not in recommendations.keys():
                    recommendations[user_index] = [(movie_index, predict_rating(p_matrix, q_matrix, user_index, movie_index, num_factors))]
                else:
                    recommendations[user_index].append((movie_index, predict_rating(p_matrix, q_matrix, user_index, movie_index, num_factors)))
    
    # Write the recommendations to a csv file
    with open('lfm_recommendations.csv', 'w', newline='') as recommendations_file:
        predictions_write = csv.writer(recommendations_file, delimiter=',')
        # Iterate through the predictions and write them to the csv file
        for user in recommendations:
            for (movie_index, predicted_rating) in recommendations[user]:
                predictions_write.writerow([user, movie_index, predicted_rating])

# Get the top N recommendations for every user.
def get_top_N_lfm_recommendations(n=10):
    movie_id_to_index = map_movie_id_to_index()

    # Store mapping between user and a list of (movie-id, predicted rating) tuples
    recommendations = {}
    movie_index_to_movie_id = {y:x for x,y in movie_id_to_index.items()}

    # Read the lfm_recommendations.csv file and store the predictions and actual ratings for each user in a dictionary
    with open('lfm_recommendations.csv', 'r', newline='') as recommendations_file:
        recommendations_read = csv.reader(recommendations_file, delimiter=',')
        # Iterate through all the rows in testing.csv to get the testing ratings
        for row in recommendations_read:
            # The index for the user
            user_index = int(row[0])
            # The index for the movie
            movie_index = int(row[1])
            # The predicted rating
            predicted_rating = float(row[2])
            # The movie id
            movie_id = movie_index_to_movie_id[movie_index]

            if user_index not in recommendations.keys():
                recommendations[user_index] = [(movie_id, predicted_rating)]
            else:
                recommendations[user_index].append((movie_id, predicted_rating))

    # Iterate through the dictionary
    for user in recommendations:
        # Get the tuple list for the user
        tuple_list = recommendations[user]
        # Sort the tuple list by predicted rating
        tuple_list = sorted(tuple_list, key=lambda x: x[1], reverse=True)
        # Add the top N movies to recommendations
        recommendations[user] = tuple_list[:n]

    return recommendations

# Output the top recommendation list for each user
def output_top_recommendations(recommendations, movie_id_to_index):
    # Write to file lfm_top_list.csv
    with open('lfm_top_list.csv', 'w', newline='') as toplfm_file:
        output_write = csv.writer(toplfm_file, delimiter=',')
        # Iterate through the recommendations
        for user_index in recommendations:
            # Get the top recommendations for user_index
            top_recommendations = recommendations[user_index]
            # Iterate through the top recommendations
            for rec in top_recommendations:
                # Get the movie id
                movie_id = rec[0]
                # Get the movie index
                movie_index = movie_id_to_index[movie_id]

                # Write the user and movie index to the output
                output_write.writerow([user_index, movie_index])

# Map every user to the movie ids they've watched in the testing.csv file
def map_user_to_movie_ids():
    # Initialize the variables
    user_to_movie_ids = {}

    # Read the testing.csv file and store the movie ids for each user in a dictionary
    with open('testing.csv', 'r', newline='') as testing_file:
        testing_read = csv.reader(testing_file, delimiter=',')
        # Iterate through all the rows in testing.csv to get the testing ratings
        for row in testing_read:
            # The index for the user
            user_index = int(row[0])-1
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


def test_for_rmse_and_mae(num_factors, numRounds, initialization):
     # Get the movie id to index mapping
    movie_id_to_index = map_movie_id_to_index()
    
    # Initialize p and q matrices for latent factor model
    p_matrix, q_matrix = initialize_p_and_q(num_factors, movie_id_to_index, initialization)

    # Use SGD to update p and q matrices
    use_SGD(p_matrix, q_matrix, num_factors, movie_id_to_index, numRounds)

    # Make predictions
    predictions = make_predictions(p_matrix, q_matrix, num_factors, movie_id_to_index)

    # Print errors
    return(calculate_rmse(predictions), calculate_mae(predictions))

# Output the top recommendation list for each user
def output_top_N_recommendations(recommendations, movie_id_to_index, N):
    # Write to file lfm_top_list.csv
    with open('data/lfm_top_list' + str(N) + '.csv', 'w', newline='') as toplfm_file:
        output_write = csv.writer(toplfm_file, delimiter=',')
        # Iterate through the recommendations
        for user_index in recommendations:
            # Get the top recommendations for user_index
            top_recommendations = recommendations[user_index]
            # Iterate through the top recommendations
            for rec in top_recommendations:
                # Get the movie id
                movie_id = rec[0]
                # Get the movie index
                movie_index = movie_id_to_index[movie_id]

                # Write the user and movie index to the output
                output_write.writerow([user_index, movie_index])

def test_for_lfm_metrics(num_factors, numRounds, initialization):
    # Get the movie id to index mapping
    movie_id_to_index = map_movie_id_to_index()
    
    # Initialize p and q matrices for latent factor model
    p_matrix, q_matrix = initialize_p_and_q(num_factors, movie_id_to_index, initialization)

    # Use SGD to update p and q matrices
    use_SGD(p_matrix, q_matrix, num_factors, movie_id_to_index, numRounds)

    get_all_recommendations(movie_id_to_index, p_matrix, q_matrix, num_factors)
    
    # Map users in testing.csv to movie ids they've watched
    user_to_movie_ids = map_user_to_movie_ids()

    # Map each N to (precision, recall, f1, ndcg)
    map_N_to_eval_metrics = {}

    # Iterate through the different N we are testing
    for N in [10, 15, 25, 50, 100, 500, 1000]:
        recommendations = get_top_N_lfm_recommendations(N)

        # Output the top recommendations
        output_top_N_recommendations(recommendations, movie_id_to_index, N)

        average_precision, average_recall = get_average_precision_and_recall(recommendations, user_to_movie_ids)
        f_measure = get_f_measure(average_precision, average_recall)
        ndcg = get_average_normalized_discounted_cumulative_gain(recommendations, user_to_movie_ids)

        # Add the N and the metrics to the map
        map_N_to_eval_metrics[N] = (average_precision, average_recall, f_measure, ndcg)
    
    return map_N_to_eval_metrics


def main():
    # Get the movie id to index mapping
    movie_id_to_index = map_movie_id_to_index()

    # Stores the number of factors
    num_factors = 10
    
    # Initialize p and q matrices for latent factor model
    p_matrix, q_matrix = initialize_p_and_q(num_factors, movie_id_to_index, 'random')

    # Use SGD to update p and q matrices
    use_SGD(p_matrix, q_matrix, num_factors, movie_id_to_index, 10)

    # Make predictions
    predictions = make_predictions(p_matrix, q_matrix, num_factors, movie_id_to_index)

    # Output the predictions
    #output_predictions(predictions)

    # Print errors
    print('RMSE:', calculate_rmse(predictions))
    print('MAE:', calculate_mae(predictions))

    get_all_recommendations(movie_id_to_index, p_matrix, q_matrix, num_factors)

    recommendations = get_top_N_lfm_recommendations(10)
    # print(0, recommendations[0])

    # Output the top recommendations
    output_top_recommendations(recommendations, movie_id_to_index)

    user_to_movie_ids = map_user_to_movie_ids()

    average_precision, average_recall = get_average_precision_and_recall(recommendations, user_to_movie_ids)
    print('Average Precision:', average_precision)
    print('Average Recall:', average_recall)

    f_measure = get_f_measure(average_precision, average_recall)
    print('F-Measure:', f_measure)

    dcg = 0
    idcg = 0
    for user in recommendations:
        dcg += get_discounted_cumulative_gain(user, recommendations, user_to_movie_ids)
        idcg += get_ideal_discounted_cumulative_gain(user, recommendations, user_to_movie_ids)
    print('Average Discounted Cumulative Gain:', dcg/len(recommendations))
    print('Average Ideal Discounted Cumulative Gain:', idcg/len(recommendations))
    print('Average Normalized Discounted Cumulative Gain:', get_average_normalized_discounted_cumulative_gain(recommendations, user_to_movie_ids))

if __name__ == "__main__":
    main()