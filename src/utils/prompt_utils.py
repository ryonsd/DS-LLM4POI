import numpy as np
import pandas as pd



def record2text(checkin_record, with_category=True):
    if with_category:
        return f"({checkin_record['start_time']}, {checkin_record['day_of_week']}, {checkin_record['PoiId']}, {checkin_record['PoiCategoryName']})"
    else:
        return f"({checkin_record['start_time']}, {checkin_record['day_of_week']}, {checkin_record['PoiId']})"


def record2text_target(checkin_record, with_category=True):
    if with_category:
        return f"({checkin_record['start_time']}, {checkin_record['day_of_week']}, <next_place_id>, <next_place_category>)"
    else:
        return f"({checkin_record['start_time']}, {checkin_record['day_of_week']}, <next_place_id>)"


def create_case_dict(checkin_seq, is_test=False, with_category=True):
    """
    input: dataframe of checkin sequence
    output: dictionary of text representation of context and target stay
    """
    seq = {"context_stay": "", "target_stay": ""}

    if len(checkin_seq) != 1:
        context_checkins = checkin_seq.iloc[:-1]
        for i, row in context_checkins.iterrows():
            if i == len(context_checkins) - 1:
                seq["context_stay"] += record2text(row, with_category=with_category)
            else:
                seq["context_stay"] += record2text(row, with_category=with_category) + ", "
    
    target_checkin = checkin_seq.iloc[-1]

    if is_test:
        seq["target_stay"] = record2text_target(target_checkin, with_category=with_category)
    else:
        seq["target_stay"] = record2text(target_checkin, with_category=with_category)

    return seq


def create_demos_prompt(data, demo_traj_id_list):
    demos = []
    # len_data = 0
    for traj_id in demo_traj_id_list:
        demo_data = data[data["pseudo_session_trajectory_id"] == traj_id]
        demo_seq = create_case_dict(demo_data)
        demos.append(demo_seq)

        # len_data += len(demo_data)
    # print(f"Total number of stays in demo data: {len_data}")
    return demos


def create_prompt(demos, input):
    """
    - task description
    - examples (demos)
    - current data (input)
    """
    prompt = """
    Your task is to predict a user's next location based on his/her activity pattern. You will be provided with some examples of the user's historical stay sequences.
    One sequence consists of <context> and <target>. <context> provide contextual information about where and when this user has been to before the last stay. <target> is the last stay in the sequence. Stays in <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, place_id, place_category). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    place_id: an integer representing the unique place ID, which indicates where the stay is.
    place_category: a string representing the category of the place (e.g., Train Station, Park, etc.).

    Then you need to do next location prediction on <target_current> based on <context_current> and examples. <target_current> is the prediction target with unknown place ID denoted as <next_place_id> and unknown place category name denoted as <next_place_category>, while temporal information is provided.      
    
    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you learned from examples, e.g., repeated visit to a certain place during certain time.
    2. the context stays in <context_current>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and day_of_week) of target stay, which is important because people's activity varies during different times (e.g., nighttime versus daytime) and on different days (e.g., weekday versus weekend).

    Please organize your answer in a JSON object containing following keys: \"place_id\" and \"place_category\". Do not include reasons in your output.

    The examples are as follows:
    """

    if demos:
        for demo in demos:
            prompt += f"""
    <context>: {demo['context_stay']}
    <target>: {demo['target_stay']}
    """

    prompt += f"""
    The current data are as follows:
    <context_current>: {input['context_stay']}
    <target_current>: {input['target_stay']}
    """

    return prompt.strip()


# for LLM-Mob
def create_historical_data(checkin_seq, with_category=False):
    historical_data = ""
    for i, row in checkin_seq.iterrows():
        if i == len(checkin_seq) - 1:
            historical_data += record2text(row, with_category=with_category)
        else:
            historical_data += record2text(row, with_category=with_category) + ", "
    return historical_data



def create_prompt_llm_mob(historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, place_id, place_category). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    place_id: an integer representing the unique place ID, which indicates where the stay is.
    place_category: a string representing the category of the place (e.g., Train Station, Park, etc.).

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and unknown place category name denoted as <next_place_category>, while temporal information is provided.      
    
    Please infer what the <next_place_id> might be (please output the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you learned from <history>, e.g., repeated visits to certain places during certain times;
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and day_of_week) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Please organize your answer in a JSON object containing following keys:
    "place_id" (the ID of the most probable place) and "place_category" (the cateogry name of the most probable place). Do not include reasons in your output.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <target_stay>: {X['target_stay']}
    """

    return prompt


# def create_prompt_llm_mob_w_reason(historical_data, X): # original
#     """
#     Make a single query.
#     param: 
#     X: one single sample containing context_stay and target_stay
#     """
#     prompt = f"""
#     Your task is to predict a user's next location based on his/her activity pattern.
#     You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
#     Each stay takes on such form as (start_time, day_of_week, place_id, place_category). The detailed explanation of each element is as follows:
#     start_time: the start time of the stay in 12h clock format.
#     day_of_week: indicating the day of the week.
#     place_id: an integer representing the unique place ID, which indicates where the stay is.
#     place_category: a string representing the category of the place (e.g., Train Station, Park, etc.).

#     Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and unknown place category name denoted as <next_place_category>, while temporal information is provided.      
    
#     Please infer what the <next_place_id> might be (please output the most likely place ID), considering the following aspects:
#     1. the activity pattern of this user that you leared from <history>, e.g., repeated visits to certain places during certain times;
#     2. the context stays in <context>, which provide more recent activities of this user; 
#     3. the temporal information (i.e., start_time and day_of_week) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
#     and on different days (e.g., weekday versus weekend).

#     Please organize your answer in a JSON object containing following keys:
#     "place_id" (the ID of the most probable place), "place_category" (the cateogry name of the most probable place) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

#     The data are as follows:
#     <history>: {historical_data}
#     <context>: {X['context_stay']}
#     <target_stay>: {X['target_stay']}
#     """

#     return prompt


# def create_prompt_llm_mob_wo_category(historical_data, X):
#     """
#     Make a single query.
#     param: 
#     X: one single sample containing context_stay and target_stay
#     """
#     prompt = f"""
#     Your task is to predict a user's next location based on his/her activity pattern.
#     You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
#     Each stay takes on such form as (start_time, day_of_week, place_id). The detailed explanation of each element is as follows:
#     start_time: the start time of the stay in 12h clock format.
#     day_of_week: indicating the day of the week.
#     place_id: an integer representing the unique place ID, which indicates where the stay is.

#     Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> while temporal information is provided.      
    
#     Please infer what the <next_place_id> might be (please output the most likely place ID), considering the following aspects:
#     1. the activity pattern of this user that you leared from <history>, e.g., repeated visits to certain places during certain times;
#     2. the context stays in <context>, which provide more recent activities of this user; 
#     3. the temporal information (i.e., start_time and day_of_week) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
#     and on different days (e.g., weekday versus weekend).

#     Please organize your answer in a JSON object containing following keys:
#     "place_id" (the ID of the most probable place) and "place_category" (the cateogry name of the most probable place). Do not include reasons in your output.

#     The data are as follows:
#     <history>: {historical_data}
#     <context>: {X['context_stay']}
#     <target_stay>: {X['target_stay']}
#     """

#     return prompt
