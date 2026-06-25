#!/usr/bin/env python
# coding: utf-8

# In[43]:


import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# In[44]:


# LOAD FILES
user_book_matrix = pickle.load(open('user_book_matrix.pkl', 'rb'))
user_similarity = pickle.load(open('user_similarity.pkl', 'rb'))
final_df = pickle.load(open('final_df.pkl', 'rb'))


# In[45]:


# POPULAR BOOKS FUNCTION

def get_popular_books(top_n=10):

    popular_books = (
        final_df.groupby([
            'Book-Title',
            'Book-Author',
            'Image-URL-M'
        ])['Book-Rating']
        .count()
        .reset_index()
        .rename(columns={'Book-Rating': 'Rating-Count'})
        .sort_values('Rating-Count', ascending=False)
        .head(top_n)
    )

    return popular_books


# =========================================
# USER INFORMATION
# =========================================

def get_user_info(user_id):

    user_info = final_df[
        final_df['User-ID'] == user_id
    ][[
        'Age',
        'city',
        'state',
        'country'
    ]].drop_duplicates()

    return user_info.head(1)


# =========================================
# TOP SIMILAR USERS
# =========================================

def get_similar_users(user_id, top_n=3):

    if user_id not in user_book_matrix.index:
        return []

    user_index = user_book_matrix.index.get_loc(user_id)

    similarity_scores = list(
        enumerate(user_similarity[user_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:top_n+1]

    similar_users = [
        user_book_matrix.index[i[0]]
        for i in similarity_scores
    ]

    return similar_users


# =========================================
# RECOMMENDATION FUNCTION
# =========================================

def recommend_books(user_id, top_n=10):

    # New user → popular books
    if user_id not in user_book_matrix.index:
        return get_popular_books(top_n)

    # Get user details
    user_info = get_user_info(user_id)

    if user_info.empty:
        return get_popular_books(top_n)

    user_country = user_info['country'].values[0]

    # Similar users
    similar_users = get_similar_users(user_id)

    if len(similar_users) == 0:
        return get_popular_books(top_n)

    # Books from similar users
    recommendations = final_df[
        (final_df['User-ID'].isin(similar_users)) &
        (final_df['Book-Rating'] >= 5)
    ]

    # Filter by same country
    recommendations = recommendations[
        recommendations['country'] == user_country
    ]

    # Remove already read books
    user_books = final_df[
        final_df['User-ID'] == user_id
    ]['ISBN']

    recommendations = recommendations[
        ~recommendations['ISBN'].isin(user_books)
    ]

    # Aggregate recommendations
    recommendations = (
        recommendations.groupby([
            'Book-Title',
            'Book-Author',
            'Image-URL-M'
        ])['Book-Rating']
        .mean()
        .reset_index()
        .sort_values('Book-Rating', ascending=False)
        .head(top_n)
    )

    return recommendations



   


# In[46]:


# =========================================
# STREAMLIT UI
# =========================================

st.set_page_config(
    page_title="Book Recommendation System"
)

st.title("📚 Book Recommendation System")


user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    step=1
)


if st.button("Recommend Books"):


    # USER DETAILS
  

    if user_id in final_df['User-ID'].values:

        user_info = get_user_info(user_id)

        st.subheader(" User Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Age", user_info['Age'].values[0])

        with col2:
            st.metric("City", user_info['city'].values[0])

        with col3:
            st.metric("State", user_info['state'].values[0])

        with col4:
            st.metric("Country", user_info['country'].values[0])

    else:
        st.warning(
            "User ID not found. Showing popular books."
        )
 
    # -------------------------------------
    # RECOMMENDATIONS
    # -------------------------------------

    books = recommend_books(user_id)

    st.subheader("📖 Recommended Books")

    for _, row in books.iterrows():

        col1, col2 = st.columns([1, 4])

        # Book image
        with col1:

            st.image(
                row['Image-URL-M'],
                width=120
            )

        # Book details
        with col2:

            st.markdown(
                f"### {row['Book-Title']}"
            )

            st.write(
                f"✍ Author: {row['Book-Author']}"
            )
            
            if 'Book-Rating' in row:
                st.write(
                    f"⭐ Rating Score: {round(row['Book-Rating'], 2)}"
                )

        st.divider()

            


# In[ ]:




