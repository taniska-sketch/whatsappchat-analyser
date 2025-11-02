import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import emoji
from textblob import TextBlob

extractor = URLExtract()

# ====================================
# BASIC STATS
# ====================================
def fetch_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    total_messages = df.shape[0]
    total_words = sum(len(msg.split()) for msg in df['message'])
    total_media = df[df['message'] == '<Media omitted>\n'].shape[0]
    total_links = sum(len(extractor.find_urls(msg)) for msg in df['message'])
    return total_messages, total_words, total_media, total_links


# ====================================
# MOST BUSY USERS
# ====================================
def most_busy_users(df):
    df = df[df['user'] != 'group_notification']
    word_counts = df.groupby('user')['message'].apply(lambda x: sum(len(m.split()) for m in x)).sort_values(ascending=False).reset_index()
    word_counts.columns = ['User', 'Total Words Typed']
    return word_counts


# ====================================
# WORD CLOUD
# ====================================
def create_wordcloud(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    text = " ".join(df['message'])
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(text)


# ====================================
# MOST COMMON WORDS
# ====================================
def most_common_words(selected_user, df):
    f = open("stop_hinglish.txt", "r", encoding='utf-8')
    stop_words = f.read()

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    if selected_user != "Overall":
        temp = temp[temp['user'] == selected_user]

    words = []
    for msg in temp['message']:
        for word in msg.lower().split():
            if word not in stop_words:
                words.append(word)

    return pd.DataFrame(Counter(words).most_common(20), columns=['Word', 'Count'])


# ====================================
# EMOJI ANALYSIS
# ====================================
def emoji_helper(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    emojis = []
    for msg in df['message']:
        emojis.extend([c for c in msg if c in emoji.EMOJI_DATA])

    return pd.DataFrame(Counter(emojis).most_common(), columns=['Emoji', 'Count'])


# ====================================
# TIMELINES
# ====================================
def monthly_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline


def daily_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    return df.groupby('only_date').count()['message'].reset_index()


# ====================================
# ACTIVITY MAPS
# ====================================
def week_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)


# ====================================
# NEW FEATURES
# ====================================

def hourly_activity(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    df['hour'] = df['message_date'].dt.hour
    return df['hour'].value_counts().sort_index()


def message_type_analysis(df):
    def classify(msg):
        if msg == '<Media omitted>\n':
            return 'Media'
        elif extractor.find_urls(msg):
            return 'Link'
        elif any(char in emoji.EMOJI_DATA for char in msg):
            return 'Emoji'
        else:
            return 'Text'
    df['Type'] = df['message'].apply(classify)
    return df['Type'].value_counts()


def sentiment_analysis(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    df['sentiment'] = df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return round(df['sentiment'].mean(), 3)


