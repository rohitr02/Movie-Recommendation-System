In order to run the program, one first needs to add movies.csv and ratings.csv to the same directory as the python scripts. Both ratings.csv and movies.csv come from the MovieLens small dataset.

Next, run the preprocess.py script in order to split the dataset into the training and testing components. This is important because it will generate the training and testing csv files which are required by the other scripts.

Once the training and testing data csv files have been made, one can run the content_based_model.py or latent_factor_model.py scripts to test those models and generate their output csv files.

Once the output of content and latent factor models exists, then one can run the mixed_list.py script to see the results of that.

Before running collect_data.py, make a directory named 'data' and one named 'results'. This is where the output of collect_data.py will go.

The rest of the scripts can be run at this point without error.
