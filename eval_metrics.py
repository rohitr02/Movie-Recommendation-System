import math

# Calculate the precision for a user
def get_user_precision_and_recall(user, recommendations, user_to_movie_ids):
    # Initialize the variables
    intersection = 0

    i = 0
    # Iterate through the tuple list
    for tuple in recommendations[user]:
        # Exit for loop after first 10 recommendations
        if i == 10:
            break
        i += 1
        # Get the movie id
        movie_id = tuple[0]
        # If the movie id is in the testing movie ids
        if int(movie_id) in user_to_movie_ids[user]:
            intersection += 1
    
    precision = intersection / 10
    recall = intersection / len(user_to_movie_ids[user])
    return precision, recall

# Get average precision and recall for all users
def get_average_precision_and_recall(recommendations, user_to_movie_ids):
    # Initialize the variables
    average_precision = 0
    average_recall = 0
    # Iterate through the dictionary
    for user in recommendations:
        # Get the precision and recall for the user
        precision, recall = get_user_precision_and_recall(user, recommendations, user_to_movie_ids)
        # Add the precision and recall to the average precision and recall
        average_precision += precision
        average_recall += recall
    
    # Divide the average precision and recall by the number of users
    average_precision /= len(recommendations)
    average_recall /= len(recommendations)
    return average_precision, average_recall

# Get the F-measure for a user
def get_f_measure(precision, recall):
    return (2 * precision * recall) / (precision + recall)

# Get the relevance of a recommendation
def get_relevance(user, movie_id, user_to_movie_ids):
    # If the user has watched the movie
    if movie_id in user_to_movie_ids[user]:
        return 1
    else:
        return 0

# Get the discounted cumulative gain for a user
def get_discounted_cumulative_gain(user, recommendations, user_to_movie_ids):
    # Initialize the variables
    dcg = 0
    index = 0
    i = 0
    # Iterate through the tuple list
    for tuple in recommendations[user]:
        # Exit for loop after first 10 recommendations
        if i == 10:
            break
        i += 1
        # Get the movie id
        movie_id = int(tuple[0])
        # Get the relevance of the movie
        relevance = get_relevance(user, movie_id, user_to_movie_ids)
        # Add the relevance to the discounted cumulative gain
        dcg += (math.pow(2, relevance) - 1) / math.log2(2 + index)
        index += 1
    
    return dcg

# Get the ideal discounted cumulative gain for a user
def get_ideal_discounted_cumulative_gain(user, recommendations, user_to_movie_ids):
    # Initialize the variables
    ideal_dcg = 0
    index = 0
    i = 0
    # Iterate through the tuple list
    for tuple in recommendations[user]:
        # Exit for loop after first 10 recommendations
        if i == 10:
            break
        i += 1
        # Get the movie id
        movie_id = int(tuple[0])
        # Get the relevance of the movie
        relevance = get_relevance(user, movie_id, user_to_movie_ids)
        # Add the relevance to the ideal discounted cumulative gain
        ideal_dcg += (math.pow(2, relevance) - 1) / math.log2(2 + index)
        if relevance == 1: index += 1
    
    return ideal_dcg

# Get average normalized discounted cumulative gain for all users
def get_average_normalized_discounted_cumulative_gain(recommendations, user_to_movie_ids):
    # Initialize the variables
    average_dcg = 0
    average_idcg = 0
    average_normalized_dcg = 0
    # Iterate through the dictionary
    for user in recommendations:
        # Get the discounted cumulative gain
        average_dcg += get_discounted_cumulative_gain(user, recommendations, user_to_movie_ids)
        # Get the ideal discounted cumulative gain
        average_idcg += get_ideal_discounted_cumulative_gain(user, recommendations, user_to_movie_ids)
    
    average_dcg /= len(recommendations)
    average_idcg /= len(recommendations)

    average_normalized_dcg = average_dcg / average_idcg
    return average_normalized_dcg
