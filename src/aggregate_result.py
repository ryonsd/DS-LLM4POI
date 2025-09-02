import json
import glob
import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    output_dir = "output/DS-LLM4POI"
    model_name_list = ["gpt-4o", "Qwen2.5-7B-Instruct"]
    dataset_name_list = ["nyc", "tky", "ca"]

    output_file_path_list = []
    for model_name in model_name_list:
        for dataset_name in dataset_name_list:
            output_file_paths = glob.glob(f"{output_dir}/{model_name}/{dataset_name}/*.json")
            output_file_path_list.extend(output_file_paths)

    result_list = []
    for output_file_path in output_file_path_list:
        with open(output_file_path, "r") as f:
            output = json.load(f)
        print(output_file_path)
        result = {
            "model_name": output["config"]["model"]["name"],
            "dataset_name": output["config"]["dataset"]["name"],
            "demo_selection": output["config"]["prompting"]["demo_selection"],
            "num_demo": output["config"]["prompting"]["num_demo"],
            "random_seed": output["config"]["prompting"]["random_seed"],
            "similarity": output["config"]["prompting"]["similarity"],
            "ACC@1": output["result"]["ACC@1"],
            "ACC@1_category": output["result"]["ACC@1_category"],
            "Average_num_tokens": output["result"]["Average_num_tokens"],
            "Cannot_answer_num": output["result"]["Cannot_answer_num"]
        }

        result_list.append(result)

    result_df = pd.DataFrame(result_list)

    # sort
    result_df = result_df.sort_values(
        ["model_name", "dataset_name", "demo_selection", "similarity", "num_demo", "random_seed"])

    result_df.to_csv("results/FS-LLM4POIrec.csv", index=False)
