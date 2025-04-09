# dashboard.py: for visualizing example graphs/charts for subreddits
# Cassandra M. @ 2025-03-15
# Grp. 2 @ BDC800NAA, Assignment 5

import streamlit as st
from PIL import Image
import os

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# NOTE: More Streamlit documentation and methods found in: https://docs.streamlit.io/get-started/fundamentals/main-concepts

MAIN_TITLE = "Subreddit Analysis Dashboard"

st.set_page_config(
    page_title=MAIN_TITLE,
    page_icon="reddit_favicon.jpeg",
    layout="wide",
)
st.title(MAIN_TITLE)

subreddit_name = st.selectbox("Select a subreddit:",
["r/politics", "r/liberals", "r/communist", "r/conservative", "r/canada", "r/usa", "r/trump", "r/fascist", "r/opinions"])

if subreddit_name:
    st.write(f"You entered: {subreddit_name}")

    subreddit_name_clean = subreddit_name.replace('r/', '')
    subreddit_folder = f"subreddits/{subreddit_name_clean}"

    # Each subreddit has a folder with the following files:
    # - political_leanings.png
    # - word_cloud_left.png
    # - word_cloud_right.png

    # EXAMPLE files are provided in the subreddits folder

    if os.path.exists(subreddit_folder):

        try:
            political_leanings_path = os.path.join(subreddit_folder, "example_political_leanings.png")
            political_leanings_img = Image.open(political_leanings_path)
            st.write(f"Political Leanings of {subreddit_name}:")
            st.image(political_leanings_img)
        except FileNotFoundError:
            st.error(f"Political leanings visualization not found for {subreddit_name}.")

        try:
            word_cloud_left_path = os.path.join(subreddit_folder, "example_word_cloud_left.png")
            word_cloud_left_img = Image.open(word_cloud_left_path)
            st.write(f"Word Cloud for 'Left' Terms in {subreddit_name}:")
            st.image(word_cloud_left_img)
        except FileNotFoundError:
            st.error(f"Left word cloud visualization not found for {subreddit_name}.")

        try:
            word_cloud_right_path = os.path.join(subreddit_folder, "example_word_cloud_right.png")
            word_cloud_right_img = Image.open(word_cloud_right_path)
            st.write(f"Word Cloud for 'Right' Terms in {subreddit_name}:")
            st.image(word_cloud_right_img)
        except FileNotFoundError:
            st.error(f"Right word cloud visualization not found for {subreddit_name}.")

        except Exception as e:
            st.error(f"Error loading images: {e}")
else:
    st.error(f"Subreddit folder not found: {subreddit_folder}")