import argparse
import json
import os
from Trip_duration_predictor_utils_data import Load_and_prepare_data, Get_preprocessor,Define_Features
from Trip_duration_predictor_utils_modeling import Get_model,save_config_and_results_json,find_and_save_best_config
from Trip_duration_predictor_utils_eval import create_config_and_metrics,evalute
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trip Duration Predictor')

    parser.add_argument('--dataset_path', type=str, default=r"E:\ML homework\Projects\project 1  Trip Duration Prediction\Code\Data\train.csv", help='dataset path')
    parser.add_argument('--validation_path', type=str, default=r"E:\ML homework\Projects\project 1  Trip Duration Prediction\Code\Data\val.csv", help='Validation dataset path')

    parser.add_argument('--preprocessor', type=int, default=2, help=
    '''
    1 for MinMaxScaler
    2 for StandardScaler
    ''')

    parser.add_argument('--poly_degree', type=int, default=1, help='Degree of polynomial features')

    parser.add_argument('--model', type=int, default=1, help=
    '''
    1 for Ridge
    ''')


    parser.add_argument('--hyperparameters', type=str, default='[]', help='List of hyperparameter dictionaries')


    args = parser.parse_args()

    # Load hyperparameters from JSON string
    hyperparameters_list = json.loads(args.hyperparameters)


    # Directory to save results
    base_path = r"E:\ML homework\Projects\project 1  Trip Duration Prediction\configs_and_results"



    experiment_number=1

    # Iterate over hyperparameter combinations
    for i, hyperparameters in enumerate(hyperparameters_list, start=1):
        config_and_metrics = create_config_and_metrics(args, hyperparameters)

        # Step 1: Load data
        train= Load_and_prepare_data(args.dataset_path)
        numeric_features, categorical_features, features = Define_Features()

        # Step 2: Preprocessing
        preprocessor = Get_preprocessor(args.preprocessor,args.poly_degree)

        # Step 3: Modeling
        returned_model = Get_model(args.model, hyperparameters)
        model=Pipeline(steps=[('preprocessor',preprocessor),('model',returned_model)])
        predictor=model.fit(train[features], train['log_trip_duration'])
        training_metrics =evalute(predictor, train[features], train['log_trip_duration'],"training")
        config_and_metrics["metrics"]["training"].update(training_metrics)


        # Step 5: Evaluation
        test= Load_and_prepare_data(args.validation_path)
        testing_metrics = evalute(predictor, test[features], test['log_trip_duration'],"Validation")
        config_and_metrics["metrics"]["Validation"].update(testing_metrics)

        # Step 6: Saving configs and results
        json_file_name = f"experiment_number_{experiment_number}.json"
        json_path = os.path.join(base_path, json_file_name)
        save_config_and_results_json(config_and_metrics, json_path=json_path)

        experiment_number += 1

    # Step 7: Find and save the best configuration
    find_and_save_best_config(base_path)

