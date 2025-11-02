import re
import pandas as pd

def preprocess(data):
    # Regex pattern for WhatsApp date-time format (handles AM/PM)
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)\s?-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # Create DataFrame
    df = pd.DataFrame({'message_date': dates, 'user_message': messages})

    # Convert message_date column to datetime
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p - ', errors='coerce')

    # Rename for clarity
    df.rename(columns={'message_date': 'message_date'}, inplace=True)

    # Split user and message
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)
        if len(entry) > 1:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extract date and time components
    df['day_name'] = df['message_date'].dt.day_name()
    df['only_date'] = df['message_date'].dt.date
    df['year'] = df['message_date'].dt.year
    df['month_num'] = df['message_date'].dt.month
    df['month'] = df['message_date'].dt.month_name()
    df['day'] = df['message_date'].dt.day
    df['hour'] = df['message_date'].dt.hour
    df['minute'] = df['message_date'].dt.minute

    # ✅ FIX for ValueError (ensure hour is integer)
    df['hour'] = df['hour'].fillna(0).astype(int)
    df['period'] = df['hour'].apply(lambda x: f"{x:02d}-{(x + 1) % 24:02d}")

    return df



