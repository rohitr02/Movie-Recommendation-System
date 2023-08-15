import csv
import matplotlib.pyplot as plt
from content_based_model import map_genre_to_index, get_num_of_movies_per_genre


genre_to_index = map_genre_to_index()
num_of_movies_per_genre = get_num_of_movies_per_genre(genre_to_index)

# Get number of users per rating value
def get_num_of_users_per_rating(user_to_ratings):
    # Initialize the variables
    num_of_users_per_rating = {'0.0': 0, '0.5': 0, '1.0': 0, '1.5':0, '2.0': 0, '2.5': 0, '3.0': 0, '3.5': 0, '4.0': 0, '4.5': 0, '5.0': 0}
    # Iterate through the dictionary
    for user in user_to_ratings:
        # Get the rating for the user
        ratings = user_to_ratings[user]
        for rating in ratings:
            # Increase the number of users per rating
            num_of_users_per_rating[str(rating)] += 1
    return num_of_users_per_rating

# Get user_to_ratings from the ratings.csv file
def get_user_to_ratings():
    # Initialize the variables
    user_to_rating = {}
    # Open the ratings.csv file
    with open('ratings.csv', 'r') as ratings_file:
        # Read the ratings.csv file
        ratings_csv = csv.reader(ratings_file)
        # Skip the header
        next(ratings_csv)
        # Iterate through the ratings.csv file
        for row in ratings_csv:
            # Get the user id
            user_id = int(row[0])
            # Get the rating
            rating = float(row[2])
            # Add the user id and rating to the user_to_rating dictionary
            if user_id not in user_to_rating: user_to_rating[user_id] = [rating]
            else: user_to_rating[user_id].append(rating)
    return user_to_rating


# Method to plot the number of movies per genre
def plot_num_of_movies_per_genre(num_of_movies_per_genre):
    # Initialize plot_list which will contain tuples of (genre, number of movies)
    plot_list = []
    # Iterate through the dictionary
    for genre in genre_to_index:
        # Append the tuple to the plot_list
        if genre == '(no genres listed)': plot_list.append(('no genre', num_of_movies_per_genre[genre_to_index[genre]]))
        else: plot_list.append((genre, num_of_movies_per_genre[genre_to_index[genre]]))
    # Sort the plot_list by alphabetical order of genre
    plot_list.sort(key=lambda x: x[0])
    # Initialize the x and y lists
    x = []
    y = []
    # Iterate through the plot_list
    for tuple in plot_list:
        # Append the genre and number of movies to the x and y lists
        x.append(tuple[0])
        y.append(tuple[1])
    # Plot the x and y lists
    plt.bar(x, y)
    # Set the x and y axis labels
    plt.xlabel('Genre')
    plt.ylabel('Number of movies')
    # Set the title
    plt.title('Number of movies per genre')
    # Increase horizontal spacing between bars
    plt.xticks(rotation=90)
    # Decrease x-axis tick size
    plt.tick_params(axis='x', which='major', labelsize=5)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    # Show the plot
    plt.show()

# Method to plot the number of users per rating
def plot_num_of_users_per_rating(num_of_users_per_rating):
    # Initialize plot_list which will contain tuples of (rating, number of users)
    plot_list = []
    # Iterate through the dictionary
    for rating in num_of_users_per_rating:
        # Append the tuple to the plot_list
        plot_list.append((rating, num_of_users_per_rating[rating]))
    # Sort the plot_list by alphabetical order of rating
    plot_list.sort(key=lambda x: x[0])
    # Initialize the x and y lists
    x = []
    y = []
    # Iterate through the plot_list
    for tuple in plot_list:
        # Append the rating and number of users to the x and y lists
        x.append(tuple[0])
        y.append(tuple[1])
    # Plot the x and y lists
    plt.bar(x, y)
    # Set the x and y axis labels
    plt.xlabel('Rating')
    plt.ylabel('Number of users')
    # Set the title
    plt.title('Number of users per rating')
    # Decrease x-axis tick size
    plt.tick_params(axis='x', which='major', labelsize=7)
    # Show the plot
    plt.show()

def main():
    # plot_num_of_movies_per_genre(num_of_movies_per_genre)
    plot_num_of_users_per_rating(get_num_of_users_per_rating(get_user_to_ratings()))
    

if __name__ == '__main__':
    main()