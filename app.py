import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📊 WhatsApp Chat Analyzer", layout="wide")
st.sidebar.title("📱 WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat text file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    st.dataframe(df)

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis for:", user_list)

    if st.sidebar.button("Show Analysis"):
        st.header("📈 Chat Statistics")

        total_messages, total_words, total_media, total_links = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Messages", total_messages)
        with col2:
            st.metric("Total Words", total_words)
        with col3:
            st.metric("Media Shared", total_media)
        with col4:
            st.metric("Links Shared", total_links)

        # Monthly Timeline
        st.subheader("🗓️ Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.subheader("📅 Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='orange')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity Map
        st.subheader("📊 Activity Map")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Most Busy Day**")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("**Most Busy Month**")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='blue')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Heatmap
        st.subheader("🔥 Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        if user_heatmap.empty:
            st.warning("No activity data available to display heatmap.")
        else:
            fig, ax = plt.subplots()
            sns.heatmap(user_heatmap, cmap="YlGnBu", ax=ax)
            st.pyplot(fig)

        # Most Active Users
        if selected_user == "Overall":
            st.subheader("💬 Most Active Users (by Words)")
            busy_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            ax.bar(busy_df['User'], busy_df['Total Words Typed'], color='skyblue')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
            st.dataframe(busy_df)

        # Word Cloud
        st.subheader("☁️ Word Cloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most Common Words
        st.subheader("🔠 Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df['Word'], most_common_df['Count'])
        st.pyplot(fig)

        # Emoji Analysis
        st.subheader("😀 Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            if not emoji_df.empty:
                fig, ax = plt.subplots()
                ax.pie(emoji_df['Count'].head(), labels=emoji_df['Emoji'].head(), autopct="%0.2f")
                st.pyplot(fig)

        # =========================
        # NEW FEATURES
        # =========================
        st.subheader("⏰ Hourly Activity Pattern")
        hourly = helper.hourly_activity(selected_user, df)
        if not hourly.empty:
            fig, ax = plt.subplots()
            ax.bar(hourly.index, hourly.values, color='plum')
            plt.xlabel("Hour of the Day")
            plt.ylabel("Messages Sent")
            st.pyplot(fig)

        st.subheader("📂 Message Type Breakdown")
        msg_type = helper.message_type_analysis(df)
        fig, ax = plt.subplots()
        ax.pie(msg_type.values, labels=msg_type.index, autopct="%0.1f%%", startangle=90)
        st.pyplot(fig)

        st.subheader("💭 Sentiment Analysis")
        sentiment = helper.sentiment_analysis(selected_user, df)
        st.metric("Average Sentiment Score", sentiment)
        if sentiment > 0:
            st.success("Chat tone is mostly Positive 😊")
        elif sentiment < 0:
            st.error("Chat tone is mostly Negative 😠")
        else:
            st.info("Chat tone is Neutral 😐")
