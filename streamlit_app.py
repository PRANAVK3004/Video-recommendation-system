import streamlit as st
import pandas as pd
from preprocess import DataPreprocessor
from utils import RecommendationEngine
from evaluation_metrics import RecommendationMetrics
from datetime import timedelta


st.title('Video Recommendation System')

BASE_URL = "https://api.socialverseapp.com"
FLIC_TOKEN = "flic_b9c73e760ec8eae0b7468e7916e8a50a8a60ea7e862c32be44927f5a5ca69867"

preprocessor = DataPreprocessor(BASE_URL, FLIC_TOKEN)
user_video_matrix = preprocessor.create_user_interaction_matrix()
video_features = preprocessor.create_video_features()
recommendation_engine = RecommendationEngine(user_video_matrix, video_features)
recommendation_engine.compute_similarity_matrices()


metrics = RecommendationMetrics(preprocessor)


st.sidebar.header("Input Parameters")

user_id = st.sidebar.text_input("Enter User ID:")
n_recommendations = st.sidebar.slider("Number of Recommendations", 1, 20, 5)

if st.sidebar.button("Get Recommendations"):
    if user_id:
        try:
           
            recommendations = recommendation_engine.get_hybrid_recommendations(
                user_id,
                n_recommendations=n_recommendations
            )

            
            metrics.log_recommendation(user_id, recommendations)

            trending_scores = preprocessor.calculate_trending_scores(video_features)

            
            recommendation_details = []
            for video_id in recommendations:
                video_data = video_features[video_features['video_id'] == video_id].iloc[0].to_dict()
                video_data['trending_score'] = trending_scores[video_features['video_id'] == video_id].iloc[0]
                recommendation_details.append(video_data)

           
            st.subheader(f"Recommendations for User: {user_id}")
            st.write(pd.DataFrame(recommendation_details))

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please enter a User ID to get recommendations.")


st.sidebar.header("Log Video Clicks")

log_click_user_id = st.sidebar.text_input("Enter User ID for Click Log:")
log_click_video_id = st.sidebar.text_input("Enter Video ID for Click Log:")

if st.sidebar.button("Log Click"):
    if log_click_user_id and log_click_video_id:
        try:
            metrics.log_click(log_click_user_id, log_click_video_id)
            st.success(f"Click logged for User {log_click_user_id} on Video {log_click_video_id}.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please provide both User ID and Video ID to log a click.")


st.sidebar.header("Recommendation Metrics")

days = st.sidebar.number_input("Time Window (Days)", min_value=1, max_value=365, value=7)

if st.sidebar.button("Get Metrics"):
    try:
        
        time_window = timedelta(days=days) if days else None
        summary = metrics.get_metric_summary(time_window)
        st.subheader("Metric Summary")
        st.write(summary)
    except Exception as e:
        st.error(f"Error: {str(e)}")


st.sidebar.header("Metrics History")

if st.sidebar.button("Get Metrics History"):
    try:
        df = metrics.plot_metric_history()
        st.subheader("Metrics History")
        st.write(df)
    except Exception as e:
        st.error(f"Error: {str(e)}")
