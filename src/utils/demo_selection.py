import random


def random_selection(user_current_data, user_historical_data, config):
    user_historical_traj_id_list = user_historical_data['pseudo_session_trajectory_id'].unique().tolist()

    num_demo = config.prompting.num_demo
    random.seed(config.prompting.random_seed)
    num_demo = min(num_demo, len(user_historical_traj_id_list))
    demo_traj_id_list = random.sample(
        list(user_historical_traj_id_list), num_demo)

    return demo_traj_id_list


def date_base_selection(user_current_data, user_historical_data, config):
    ascending = not config.prompting.reverse_order
    user_historical_data = user_historical_data.sort_values(by='UTCTimeOffsetEpoch', ascending=ascending).reset_index(drop=True)
    user_historical_traj_id_list = user_historical_data['pseudo_session_trajectory_id'].unique().tolist()
    demo_traj_id_list = user_historical_traj_id_list[-config.prompting.num_demo:]

    return demo_traj_id_list



def similarity_based_selection(user_current_data, user_historical_data, additional_data, config):
    dist_data = additional_data
    traj_id = user_current_data['pseudo_session_trajectory_id'].iloc[0]

    user_historical_traj_id_list = user_historical_data['pseudo_session_trajectory_id'].unique().tolist()

    demo_traj_id_list = dist_data[str(traj_id)]
    demo_traj_id_list = [int(k) for k in demo_traj_id_list]

    if not config.prompting.reverse_order:
        demo_traj_id_list = demo_traj_id_list[::-1]

    sorted_list = [id for id in demo_traj_id_list if id in user_historical_traj_id_list]
    demo_traj_id_list = sorted_list

    demo_traj_id_list = demo_traj_id_list[-config.prompting.num_demo:]

    return demo_traj_id_list
