import random
import os
import re
import json
import pickle
import tiktoken
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import datetime
from omegaconf import OmegaConf
from vllm import LLM, SamplingParams

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.data_utils import load_data
from src.utils.prompt_utils import create_demos_prompt, create_case_dict, create_prompt, create_historical_data, create_prompt_llm_mob
from src.utils.demo_selection import random_selection, date_base_selection, similarity_based_selection


from openai import OpenAI
api_key = "your_openai_api_key"
client = OpenAI(api_key=api_key)


def predict(client, prompt, model="gpt-4o-mini", json_mode=True, max_tokens=1200):
    """
    args:
        client: the openai client object (new in 1.x version)
        prompt: the prompt to be completed
        model: specify the model to use
        json_mode: whether return the response in json format (new in 1.x version)
    """
    messages = [{"role": "user", "content": prompt}]
    if json_mode:
        completion = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0,  # the degree of randomness of the model's output
            max_tokens=max_tokens  # the maximum number of tokens to generate
        )
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens
        )
    res_content = completion.choices[0].message.content
    
    return res_content


def predict_local_llm(llm, prompt, max_new_tokens=2000):
    messages = [{"role": "user", "content": prompt}]
    outputs = llm.chat(
        messages=messages,
        sampling_params= SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            ),
        use_tqdm=False
        )
    return outputs[0].outputs[0].text 


def extract_json(res_content):
    pattern = r'```json\s*([\s\S]*?)```'
    match = re.search(pattern, res_content)
    
    if not match:
        return ""
    
    json_text = match.group(1).strip()
    return json_text


def get_num_tokens(prompt, config):
    encoding = tiktoken.encoding_for_model(config.model.name)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens


def organaize_data(test_data, historical_data, traj_id, config, additional_data=False):
    user_current_data = test_data[test_data['pseudo_session_trajectory_id'] == traj_id]
    target_poi_id = user_current_data.iloc[-1]['PoiId']
    traget_poi_category = user_current_data.iloc[-1]['PoiCategoryName']
    user_id = user_current_data.iloc[0]['UserId']

    if config.prompting.data_bank == "the_user":
        historical_data = historical_data[historical_data['UserId'] == user_id]
    elif config.prompting.data_bank == "all_users":
        pass

    # creaate prompt
    input_prompt = create_case_dict(user_current_data, is_test=True)

    if config.prompting.type == "LLM-Mob":
        historical_data = historical_data.copy()
        historical_data = historical_data.sort_values(by='UTCTimeOffset', ascending=True).reset_index(drop=True)
        historical_data_traj_ids = historical_data['pseudo_session_trajectory_id'].unique().tolist()

        # historical_data = historical_data[historical_data['pseudo_session_trajectory_id'].isin(historical_data_traj_ids[-config.prompting.num_demo:])]
        filtered_data = pd.DataFrame()
        for traj_id in historical_data_traj_ids[-config.prompting.num_demo:]:
            filtered_data = pd.concat(
                [filtered_data, historical_data[historical_data['pseudo_session_trajectory_id'] == traj_id]])
        historical_data = filtered_data

        historical_prompt = create_historical_data(historical_data, with_category=True)
        prompt = create_prompt_llm_mob(historical_prompt, input_prompt)

    else: # 
        # demo selection
        if config.prompting.demo_selection == "random":
            demo_traj_id_list = random_selection(user_current_data, historical_data, config)
            demos = create_demos_prompt(historical_data, demo_traj_id_list)
        elif config.prompting.demo_selection == "date":
            demo_traj_id_list = date_base_selection(user_current_data, historical_data, config)
            demos = create_demos_prompt(historical_data, demo_traj_id_list)
        else: # similarity
            demo_traj_id_list = similarity_based_selection(user_current_data, historical_data, additional_data, config)
            demos = create_demos_prompt(historical_data, demo_traj_id_list)

        prompt = create_prompt(demos=demos, input=input_prompt)

    return prompt, target_poi_id, traget_poi_category


def check_res_content(res_content, target_poi_id, traget_poi_category):
    try:
        res = json.loads(res_content)
        pred_poi_id = res['place_id']
        pred_poi_category = res['place_category']
        is_true_num = 1 if pred_poi_id == target_poi_id else 0
        is_true_num_category = 1 if pred_poi_category == traget_poi_category else 0
    except:
        pred_poi_id = -1
        pred_poi_category = -1
        is_true_num = 0
        is_true_num_category = 0

    return is_true_num, is_true_num_category, pred_poi_id, pred_poi_category


def save_output(output, config):
    output_dir = f"{config.default.output_dir}/{config.model.name}/{config.dataset.name}/{config.prompting.data_bank}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if config.prompting.type == "LLM-Mob":
        output_path = os.path.join(output_dir, f"num_demo_{config.prompting.num_demo}.json")
    else:
        if config.prompting.demo_selection == "random":
            output_path = os.path.join(output_dir, f"{config.prompting.demo_selection}_{config.prompting.num_demo}_{config.prompting.random_seed}.json")
        elif config.prompting.demo_selection == "date":
            if config.prompting.reverse_order:
                output_path = os.path.join(output_dir, f"{config.prompting.demo_selection}_{config.prompting.num_demo}_reverse.json")
            else:
                output_path = os.path.join(output_dir, f"{config.prompting.demo_selection}_{config.prompting.num_demo}.json")
        else:
            if config.prompting.reverse_order:
                output_path = os.path.join(output_dir, f"{config.prompting.demo_selection}_{config.prompting.similarity}_{config.prompting.num_demo}_reverse.json")
            else:
                output_path = os.path.join(output_dir, f"{config.prompting.demo_selection}_{config.prompting.similarity}_{config.prompting.num_demo}.json")
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)


def run(test_data, historical_data, config):
    traj_ids = test_data['pseudo_session_trajectory_id'].unique()

    if config.prompting.demo_selection == "similarity":
        similarity_data_path = f"datasets/{config.dataset.name}/preprocessed/similarity/{config.prompting.similarity}.pkl"
        with open(similarity_data_path, 'rb') as f:
            similarity_data = pickle.load(f)
    else:
        similarity_data = None

    true_num = 0
    true_num_category = 0
    cannot_answer_num = 0
    num_tokens_list = []
    
    output = {
        "config": OmegaConf.to_container(config),
        "result": {
            "ACC@1": 0,
            "ACC@1_category": 0,
            "Cannot_answer_num": 0,
            "Average_num_tokens": 0,
        },
        "input_output": {}
    }

    if config.model.name.startswith("gpt"):
        # create a new client
        client = OpenAI(api_key=api_key)
    else:
        client = LLM(model=config.model.name)

    for traj_id in tqdm(traj_ids):
        # create prompt
        prompt, target_poi_id, traget_poi_category = organaize_data(
            test_data, historical_data, traj_id, config, similarity_data)
        #num_tokens = get_num_tokens(prompt, config)
        #num_tokens_list.append(num_tokens)

        # predict
        if config.model.name == "gpt-4o" or config.model.name == "gpt-4o-mini":
            res_content = predict(client, prompt, model=config.model.name)
        else:
            res_content = predict_local_llm(client, prompt)
            res_content = extract_json(res_content)

        # check result
        is_true_num, is_true_num_category, pred_poi_id, pred_poi_category = check_res_content(res_content, target_poi_id, traget_poi_category)

        if pred_poi_id == -1:
            cannot_answer_num += 1
        else:
            true_num += is_true_num
            true_num_category += is_true_num_category
        
        output['input_output'][str(traj_id)] = {
            "prompt": prompt, 
            "res_content": res_content,
            "target": {"place_id": int(target_poi_id), "place_category": traget_poi_category}, 
        }

    # save output
    acc_1 = true_num/len(traj_ids)
    acc_1_category = true_num_category/len(traj_ids)
    avg_num_tokens = sum(num_tokens_list)/len(traj_ids)

    output["result"]["ACC@1"] = float(acc_1)
    output["result"]["ACC@1_category"] = float(acc_1_category)
    output["result"]["Cannot_answer_num"] = int(cannot_answer_num)
    output["result"]["Average_num_tokens"] = float(avg_num_tokens)

    print(output["result"])

    save_output(output, config)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="config.yaml")
    # args = parser.parse_args()
    # config = OmegaConf.load(args.config)
    # print(config)

    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--dataset_name", type=str, default="nyc", choices=['nyc', 'tky', 'ca'])
    parser.add_argument("--prompting_type", type=str, default="DS-LLM4POI", choices=['DS-LLM4POI', 'LLM-Mob'])
    parser.add_argument("--data_bank", type=str, default="the_user", choices=['the_user', 'all_users'])
    parser.add_argument("--demo_selection", type=str, default="random", choices=['random', 'date', 'similarity'])
    parser.add_argument("--num_demo", type=int, default=5, choices=[5, 15, 30])
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--similarity", type=str, default="", choices=['dtw', 'jaccard', 'lcs', 'llm4poi'])
    parser.add_argument("--reverse_order", action="store_true", default=False)
    args = parser.parse_args()

    output_dir = "output/" + args.prompting_type

    config = OmegaConf.create({
        "default": {
            "output_dir": output_dir,
        },
        "model": {
            "name": args.model_name,
        },
        "dataset": {
            "name": args.dataset_name,
        },
        "prompting": {
            "type": args.prompting_type,
            "data_bank": args.data_bank,
            "demo_selection": args.demo_selection,
            "num_demo": args.num_demo,
            "random_seed": args.random_seed,
            "similarity": args.similarity,
            "reverse_order": args.reverse_order,
        }
    })
    print(config)

    historical_data, test_data = load_data(f"datasets/{config.dataset.name}/preprocessed/")

    # for running test
    # traj_ids = test_data['pseudo_session_trajectory_id'].unique()
    # random.seed(0)
    # traj_ids = random.sample(list(traj_ids), 10)
    # test_data = test_data[test_data['pseudo_session_trajectory_id'].isin(traj_ids)].reset_index(drop=True)

    run(test_data, historical_data, config)

