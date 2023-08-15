import csv
import random

def preprocess():
    # Open file descriptors for training and testing sets
    training_file = open('training.csv', 'w', newline='')
    testing_file = open('testing.csv', 'w', newline='')

    # Create writers for training and testing sets
    training_write = csv.writer(training_file, delimiter=',')
    testing_write = csv.writer(testing_file, delimiter=',')

    # Read the ratings.csv
    with open('ratings.csv', 'r', newline='') as ratings_file:
        ratings_read = csv.reader(ratings_file, delimiter=',')
        # Holds the user id of current user
        current_user = 1
        # Holds the ratings of the current user
        user_ratings = []
        for row in ratings_read:
            # Ignore the first row
            if row[0] == 'userId':
                continue
            # Check that the row is still for current user
            if row[0] == current_user:
                # Append the rating to the user's list
                user_ratings.append(row)
            else:
                # Add 80% of user_ratings to training.csv and the other 20% to testing.csv
                add_training_and_testing(user_ratings, training_write, testing_write)
                # Update the current user and reset the user_ratings
                current_user = row[0]
                user_ratings = [row]
        # For the last user, add 80% of user_ratings to training.csv and the other 20% to testing.csv
        add_training_and_testing(user_ratings, training_write, testing_write)

    # Close file descriptors
    training_file.close()
    testing_file.close()

def main():
    preprocess()


def add_training_and_testing(user_ratings, training_write, testing_write):
    # Randomly shuffle the user_ratings
    random.shuffle(user_ratings)
    # Add the first 80% of ratings to training.csv
    num_training = round(len(user_ratings) * 0.8)
    for i in range(num_training):
        training_write.writerow(user_ratings[i])
    # Add the last 20% of ratings to testing.csv
    for i in range(num_training, len(user_ratings)):
        testing_write.writerow(user_ratings[i])


if __name__ == "__main__":
    main()