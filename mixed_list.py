import csv
import time

from latent_factor_model import map_movie_id_to_index, map_user_to_movie_ids
from eval_metrics import *

def get_scores():
    # Dictionary used to store predicted scores and similarity scores
    scores = {}
    # Read through the lfm recommendations csv file and store scores in dictionary
    with open('lfm_recommendations.csv', 'r', newline='') as rec_file:
        rec_read = csv.reader(rec_file, delimiter=',')
        # Iterate through rows and store the predicted rating for each userid, movieid pair
        for row in rec_read:
            # Get the index of the user
            user_index = int(row[0])
            # Get the index of the movie
            movie_index = int(row[1])
            # Get the predicted rating of the movie
            rating = float(row[2])

            # Add the rating as value for the key (userid, movieid)
            if user_index not in scores.keys():
                scores[user_index] = {}
            scores[user_index][movie_index] = [rating]
    
    # Read through the cbm recommendations csv file and store scores in dictionary
    with open('cbm_recommendations.csv', 'r', newline='') as rec_file:
        rec_read = csv.reader(rec_file, delimiter=',')
        # Iterate through the rows and store the similarity score for each userid, movieid pair
        for row in rec_read:
            # Get the index of the user
            user_index = int(row[0])
            # Get the index of the movie
            movie_index = int(row[1])
            # Get the similarity score of the movie
            sim_score = float(row[2])

            # Add the similarity score to the value for key (userid, movieid)
            scores[user_index][movie_index].append(sim_score)
    
    return scores

def get_top_list(csv_file):
    # Dictionary to hold top recommendations for each user
    top_list = {}

    # Read through the csv file and get the top recommendations for each user
    with open(csv_file, 'r', newline='') as top_file:
        top_read = csv.reader(top_file, delimiter=',')
        # Iterate through the rows
        for row in top_read:
            # Get the index of the user
            user_index = int(row[0])
            # Get the index of the movie
            movie_index = int(row[1])

            # Add movie to user's list
            if user_index not in top_list.keys():
                top_list[user_index] = []
            top_list[user_index].append(movie_index)
    
    return top_list

def create_new_list(scores, lfm_top_list, cbm_top_list, movie_index_to_movie_id):
    # Dictionary to hold new recommendation list that is made from lfm and cbm recommendation lists
    new_lists = {}

    # Iterate through each user_index
    for user_index in scores:
        # List to hold movie_index and new score pair
        new_scores = []

        # Iterate through lfm recommendation lists and calculate new scores
        user_lfm_list = lfm_top_list[user_index]
        for i, movie_index in enumerate(user_lfm_list):
            # Calculate new score for movie for the user using following formula
            # R * S * C^B
            # R = predicted rating from lfm
            # S = similarity score from cbm
            # C = multiplier if movie is in both lfm and cbm lists
            # B = 0 if movie is not in both lists, 1 if movie is in both lists
            predicted_rating = scores[user_index][movie_index][0]
            similarity_score = scores[user_index][movie_index][1]
            multiplier = 1.5
            both_list = 0
            if movie_index in cbm_top_list[user_index]:
                both_list = 1
            new_score = predicted_rating * similarity_score * multiplier**both_list

            # Add movie_index and new_score pair to new_scores
            new_scores.append((movie_index, new_score))
        
        # Iterate through cbm recommendation lists and calculate new scores
        user_cbm_list = cbm_top_list[user_index]
        for i, movie_index in enumerate(user_cbm_list):
            # Skip movies whose new scores were already calculated
            if movie_index in user_lfm_list:
                continue
            predicted_rating = scores[user_index][movie_index][0]
            similarity_score = scores[user_index][movie_index][1]
            multiplier = 1.5
            both_list = 0
            new_score = predicted_rating * similarity_score * multiplier**both_list

            # Add movie_index and new_score pair to new_scores
            new_scores.append((movie_index, new_score))
        
        # Sort the new_scores
        new_scores = sorted(new_scores, key=lambda x: x[1], reverse=True)
        # Create new list with movie_id instead of movie_index
        new_scores_id = []
        for i in range(10):
            new_scores_id.append((movie_index_to_movie_id[new_scores[i][0]], new_scores[i][1]))
        
        # Add new list to new_lists for user_index
        new_lists[user_index] = new_scores_id
    
    return new_lists

def test_for_mixed_metrics():
    # Get movie id to index mapping
    movie_id_to_index = map_movie_id_to_index()
    # Get index to movie id mapping
    movie_index_to_movie_id = {y:x for x,y in movie_id_to_index.items()}

    # Test user ids to movie ids mapping
    user_to_movie_ids = map_user_to_movie_ids()

    # Get the predicted ratings and similarity scores for all users
    scores = get_scores()

    # Map N to (precision, recall, f1, ndcg)
    N_to_metrics = {}

    for N in [10, 15, 25, 50, 100, 500, 1000]:
        # Start recording time
        start_time = time.time()

        # Get the top ten list for lfm model
        lfm_top_list = get_top_list('data/lfm_top_list' + str(N) + '.csv')

        # Get the top ten list for cbm model
        cbm_top_list = get_top_list('data/cbm_top_list' + str(N) + '.csv')

        # Create new recommendation lists for each user that combines both of their
        # lfm and cbm recommendation lists
        new_lists = create_new_list(scores, lfm_top_list, cbm_top_list, movie_index_to_movie_id)

        # Get the total time for making the mixed list
        total_time = time.time() - start_time

        average_precision, average_recall = get_average_precision_and_recall(new_lists, user_to_movie_ids)
        f_measure = get_f_measure(average_precision, average_recall)
        ndcg = get_average_normalized_discounted_cumulative_gain(new_lists, user_to_movie_ids)
        N_to_metrics[N] = (average_precision, average_recall, f_measure, ndcg, total_time)
    
    return N_to_metrics

def main():
    # Get movie id to index mapping
    movie_id_to_index = map_movie_id_to_index()
    # Get index to movie id mapping
    movie_index_to_movie_id = {y:x for x,y in movie_id_to_index.items()}

    # Get the predicted ratings and similarity scores for all users
    scores = get_scores()

    # Get the top ten list for lfm model
    lfm_top_list = get_top_list('lfm_top_list.csv')

    # Get the top ten list for cbm model
    cbm_top_list = get_top_list('cbm_top_list.csv')

    # Create new recommendation lists for each user that combines both of their
    # lfm and cbm recommendation lists
    new_lists = create_new_list(scores, lfm_top_list, cbm_top_list, movie_index_to_movie_id)

    user_to_movie_ids = map_user_to_movie_ids()

    average_precision, average_recall = get_average_precision_and_recall(new_lists, user_to_movie_ids)
    print('Average Precision:', average_precision)
    print('Average Recall:', average_recall)

    f_measure = get_f_measure(average_precision, average_recall)
    print('F-Measure:', f_measure)

    dcg = 0
    idcg = 0
    for user in new_lists:
        dcg += get_discounted_cumulative_gain(user, new_lists, user_to_movie_ids)
        idcg += get_ideal_discounted_cumulative_gain(user, new_lists, user_to_movie_ids)
    print('Average Discounted Cumulative Gain:', dcg/len(new_lists))
    print('Average Ideal Discounted Cumulative Gain:', idcg/len(new_lists))
    print('Average Normalized Discounted Cumulative Gain:', get_average_normalized_discounted_cumulative_gain(new_lists, user_to_movie_ids))


if __name__ == "__main__":
    main()