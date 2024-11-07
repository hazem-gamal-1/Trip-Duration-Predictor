from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import json
import os
def Get_model(option, hyperparams):
    if option==1:
        return Ridge(**hyperparams)
    

def save_config_and_results_json(config, json_path):
    with open(json_path, "w") as file:
        json.dump(config, file, indent=2)


def find_and_save_best_config(directory, output_file_name="best_configuration.json"):
    best_r2 = -1
    best_config = None

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                    config = json.load(file)


                    R2 = config.get("metrics", {}).get("Validation", {}).get("R2", None)


                    if R2 is not None and R2 > best_r2:
                        best_r2 = R2
                        best_config = config



    output_path = os.path.join(directory, output_file_name)
    with open(output_path, 'w') as output_file:
        json.dump(best_config, output_file, indent=4)

