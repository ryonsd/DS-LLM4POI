import pandas as pd

def encode_day_of_week(x):
    if x == 0:
        return 'Monday'
    if x == 1:
        return 'Tuesday'
    if x == 2:
        return 'Wednesday'
    if x == 3:
        return 'Thursday'
    if x == 4:
        return 'Friday'
    if x == 5:
        return 'Saturday'
    if x == 6:
        return 'Sunday'


def format_time(row):
    time_str = row['UTCTimeOffset'].strftime('%H:%M:%S')
    hour, minute, _ = map(int, time_str.split(':'))
    if row['UTCTimeOffsetHour'] == 0:
        return f"00:{minute:02d} AM"
    elif row['UTCTimeOffsetHour'] == 12:
        return f"00:{minute:02d} PM"
    return row['start_time']


def add_time_features(df):
    df['UTCTimeOffset'] = pd.to_datetime(df['UTCTimeOffset'])
    df['UTCTimeOffsetWeekday'] = df['UTCTimeOffset'].apply(
        lambda x: x.weekday())
    df['UTCTimeOffsetHour'] = df['UTCTimeOffset'].apply(lambda x: x.hour)

    df["day_of_week"] = df["UTCTimeOffsetWeekday"].apply(encode_day_of_week)
    df["start_time"] = df["UTCTimeOffset"].apply(
        lambda x: x.strftime("%I:%M %p"))

    time_list = []
    for i, row in df.iterrows():
        time_list.append(format_time(row))
    df["start_time"] = time_list

    return df

def load_data(data_path):
    train_data = pd.read_csv(data_path+"train_sample.csv")
    test_data = pd.read_csv(data_path+"test_sample_with_traj.csv")

    train_data = add_time_features(df=train_data)
    test_data = add_time_features(df=test_data)

    return train_data, test_data
