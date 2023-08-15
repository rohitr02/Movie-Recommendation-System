import csv

from preprocess import preprocess
from latent_factor_model import test_for_rmse_and_mae, test_for_lfm_metrics
from content_based_model import test_for_cbm_metrics
from mixed_list import test_for_mixed_metrics

def initialize_lfm_data():
    # Dictionary to hold lfm related data
    lfm_data = {}
    # Set up lfm_data
    lfm_data['factors'] = {}
    lfm_data['rounds'] = {}
    testing_values = [1, 2, 5, 10, 25, 50, 100]
    for value in testing_values:
        lfm_data['factors'][value] = [0,0]
        lfm_data['rounds'][value] = [0,0]
    lfm_data['initialization'] = {}
    testing_initializations = ['zeros', 'random', 'ones']
    for i in testing_initializations:
        lfm_data['initialization'][i] = [0,0]
    
    return lfm_data

def initialize_metric_data():
    # Dictionary to hold metric data for the models
    metric_data = {}
    # Set up metric_data
    metric_data['lfm'] = {}
    metric_data['cbm'] = {}
    metric_data['mixed'] = {}
    testing_metrics = ['precision', 'recall', 'f_measure', 'ndcg']
    for metric in testing_metrics:
        metric_data['lfm'][metric] = 0
        metric_data['cbm'][metric] = 0
        metric_data['mixed'][metric] = 0
    
    return metric_data

def initialize_mixed_data():
    # Dictionary to hold mixed related data
    mixed_data = {}
    # Set up mixed_data
    testing_rec = [10, 15, 25, 50, 100, 500, 1000]
    for rec in testing_rec:
        mixed_data[rec] = {}
    testing_metrics = ['precision', 'recall', 'f_measure', 'ndcg', 'time']
    for rec in testing_rec:
        for metric in testing_metrics:
            mixed_data[rec][metric] = 0
    
    return mixed_data

def test_latent_factor_model(lfm_data, num_factors, num_rounds, initialization, test_parameter, test_value):
    # Test with the given parameters and get the resulting rmse and mae
    result = test_for_rmse_and_mae(num_factors, num_rounds, initialization)

    # Update the rmse and mae of the parameter that was tested
    lfm_data[test_parameter][test_value][0] += result[0]
    lfm_data[test_parameter][test_value][1] += result[1]

    return

def update_metrics(metric_data, mixed_data):
    # Get the metrics for latent factor model
    lfm_result = test_for_lfm_metrics(10, 10, 'random')
    # Get the metrics for content based model
    cbm_result = test_for_cbm_metrics()
    # Get the metrics for mixed list
    mixed_result = test_for_mixed_metrics()

    # Used to hold the metrics
    testing_metrics = ['precision', 'recall', 'f_measure', 'ndcg']

    # Update metrics
    for i, metric in enumerate(testing_metrics):
        metric_data['lfm'][metric] += lfm_result[10][i]
        metric_data['cbm'][metric] += cbm_result[10][i]
        metric_data['mixed'][metric] += mixed_result[25][i]

    # Update mixed list data
    testing_metrics.append('time')
    for num_rec in mixed_result:
        for i, metric in enumerate(testing_metrics):
            mixed_data[num_rec][metric] += mixed_result[num_rec][i]

    return

def collect_metrics(num_tests, lfm_data, metric_data, mixed_data):
    # Iterate 10 times for 10 different training and testing csv files
    for i in range(num_tests):
        # Divide ratings.csv to training and testing csv files randomly
        preprocess()

        # Holds different values of number of factors and number of rounds to test
        testing_values = [1, 2, 5, 10, 25, 50, 100]
        # Test latent factor model with different number of factors and record data
        for num_factors in testing_values:
            test_latent_factor_model(lfm_data, num_factors, 10, 1, 'factors', num_factors)
        # Test latent factor model with different number of rounds and record data
        for num_rounds in testing_values:
            test_latent_factor_model(lfm_data, 10, num_rounds, 1, 'rounds', num_rounds)
        # Test latent factor model with different initialization strategies and record data
        testing_initializations = ['zeros', 'random', 'ones']
        for initialization in testing_initializations:
            test_latent_factor_model(lfm_data, 10, 10, initialization, 'initialization', initialization)
        
        # Update the metrics for each model and mixed list
        update_metrics(metric_data, mixed_data)

def write_lfm_data(num_tests, file_name, data_to_write, index):
    # Open the file to write
    with open(file_name, 'w', newline='') as res_file:
        res_write = csv.writer(res_file, delimiter=',')
        for value in data_to_write:
            res_write.writerow([value, data_to_write[value][index] / num_tests])

def write_metrics_data(num_tests, metric_data):
    # Open the file to write
    with open('results/metrics.csv', 'w', newline='') as res_file:
        res_write = csv.writer(res_file, delimiter=',')
        for value in metric_data:
            for metric in metric_data[value]:
                res_write.writerow([value, metric, metric_data[value][metric] / num_tests])

def write_mixed_data(num_tests, file_name, mixed_data, index):
    # Open the file to write
    with open(file_name, 'w', newline='') as res_file:
        res_write = csv.writer(res_file, delimiter=',')
        for value in mixed_data:
            res_write.writerow([value, mixed_data[value][index] / num_tests])

def main():
    # Dictionary to hold lfm related data
    lfm_data = initialize_lfm_data()
    # Dictionary to hold metric data
    metric_data = initialize_metric_data()
    # Dictionary to hold mixed related data
    mixed_data = initialize_mixed_data()

    # Used to hold the number of tests
    num_tests = 1

    # Collect metrics
    collect_metrics(num_tests, lfm_data, metric_data, mixed_data)

    # Write the lfm data
    write_lfm_data(num_tests, 'results/lfm_factor_rsme.csv', lfm_data['factors'], 0)
    write_lfm_data(num_tests, 'results/lfm_factor_mae.csv', lfm_data['factors'], 1)
    write_lfm_data(num_tests, 'results/lfm_rounds_rsme.csv', lfm_data['rounds'], 0)
    write_lfm_data(num_tests, 'results/lfm_rounds_mae.csv', lfm_data['rounds'], 1)
    write_lfm_data(num_tests, 'results/lfm_initialization_rsme.csv', lfm_data['initialization'], 0)
    write_lfm_data(num_tests, 'results/lfm_initialization_mae.csv', lfm_data['initialization'], 1)

    # Write the metrics data
    write_metrics_data(num_tests, metric_data)

    # Write the mixed data
    write_mixed_data(num_tests, 'results/mixed_precision.csv', mixed_data, 'precision')
    write_mixed_data(num_tests, 'results/mixed_recall.csv', mixed_data, 'recall')
    write_mixed_data(num_tests, 'results/mixed_f_measure.csv', mixed_data, 'f_measure')
    write_mixed_data(num_tests, 'results/mixed_ndcg.csv', mixed_data, 'ndcg')
    write_mixed_data(num_tests, 'results/mixed_time.csv', mixed_data, 'time')

if __name__ == "__main__":
    main()