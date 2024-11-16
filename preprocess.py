import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime

class DataPreprocessor:
    def __init__(self, base_url, flic_token):
        self.base_url = base_url
        self.headers = {'Flic-Token': flic_token}
        self.scaler = MinMaxScaler()
    
    def fetch_data(self, endpoint, params):
        """Fetch data from API endpoint"""
        try:
            response = requests.get(f"{self.base_url}{endpoint}", headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {self.base_url}{endpoint}: {str(e)}")
            return None

    def create_user_interaction_matrix(self):
        """Create user-video interaction matrix combining views, likes, and ratings"""
        # Fetch data from all endpoints
        viewed_data = self.fetch_data('/posts/view', {'page': 1, 'page_size': 1000})
        liked_data = self.fetch_data('/posts/like', {'page': 1, 'page_size': 1000})
        rated_data = self.fetch_data('/posts/rating', {'page': 1, 'page_size': 1000})
        
        if not all([viewed_data, liked_data, rated_data]):
            raise ValueError("Failed to fetch required data")
            
        # Create separate DataFrames for each interaction type
        viewed_df = pd.DataFrame([{
            'user_id': str(post['username']),  # Ensure string type
            'video_id': str(post['id']),       # Ensure string type
            'interaction_type': 'view',
            'timestamp': post['created_at']
        } for post in viewed_data['posts']])
        
        liked_df = pd.DataFrame([{
            'user_id': str(post['username']),  # Ensure string type
            'video_id': str(post['id']),       # Ensure string type
            'interaction_type': 'like',
            'timestamp': post['created_at']
        } for post in liked_data['posts']])
        
        rated_df = pd.DataFrame([{
            'user_id': str(post['username']),  # Ensure string type
            'video_id': str(post['id']),       # Ensure string type
            'interaction_type': 'rating',
            'rating_value': float(post['average_rating']),  # Ensure float type
            'timestamp': post['created_at']
        } for post in rated_data['posts']])
        
        # Combine all interactions
        all_interactions = pd.concat([viewed_df, liked_df, rated_df], ignore_index=True)
        
        # Create interaction weights
        interaction_weights = {
            'view': 1.0,    # Ensure float type
            'like': 3.0,    # Ensure float type
            'rating': 5.0   # Ensure float type
        }
        
        # Calculate weighted interaction scores
        all_interactions['weight'] = all_interactions['interaction_type'].map(interaction_weights)
        
        # Create user-video matrix
        user_video_matrix = all_interactions.pivot_table(
            index='user_id',
            columns='video_id',
            values='weight',
            aggfunc='sum',
            fill_value=0.0  # Ensure float type
        )
        
        return user_video_matrix.astype(float)  # Ensure all values are float

    def create_video_features(self):
        """Create video feature matrix from video metadata"""
        videos_data = self.fetch_data('/posts/summary/get', {'page': 1, 'page_size': 1000})
        
        if not videos_data:
            raise ValueError("Failed to fetch video data")
            
        video_features = []
        for video in videos_data['posts']:
            features = {
                'video_id': str(video['id']),  # Ensure string type
                'category_id': str(video['category']['id']),  # Ensure string type
                'view_count': float(video['view_count']),     # Ensure float type
                'like_count': float(video['upvote_count']),   # Ensure float type
                'comment_count': float(video['comment_count']), # Ensure float type
                'rating_count': float(video['rating_count']),   # Ensure float type
                'average_rating': float(video['average_rating']), # Ensure float type
                'share_count': float(video['share_count']),      # Ensure float type
                'chain_id': str(video['chain_id']),           # Ensure string type
                'created_at': float(video['created_at'])      # Ensure float type
            }
            video_features.append(features)
        
        video_df = pd.DataFrame(video_features)
        
        # Define numerical columns
        numerical_cols = ['view_count', 'like_count', 'comment_count', 'rating_count', 
                         'average_rating', 'share_count']
        
        # Ensure numerical columns are float type
        for col in numerical_cols:
            video_df[col] = pd.to_numeric(video_df[col], errors='coerce').fillna(0.0).astype(float)
        
        # Normalize numerical features
        video_df[numerical_cols] = self.scaler.fit_transform(video_df[numerical_cols])
        
        # One-hot encode categorical features
        categorical_cols = ['category_id', 'chain_id']
        for col in categorical_cols:
            dummies = pd.get_dummies(video_df[col], prefix=col)
            video_df = pd.concat([video_df, dummies], axis=1)
            video_df.drop(col, axis=1, inplace=True)
        
        return video_df

    def calculate_trending_scores(self, video_df, time_decay_factor=0.1):
        """Calculate trending scores for videos based on recent engagement"""
        current_time = float(datetime.now().timestamp() * 1000)  # Ensure float type
        
        # Ensure created_at is numeric and handle missing values
        video_df['created_at'] = pd.to_numeric(video_df['created_at'], errors='coerce').fillna(current_time)
        
        # Calculate time difference in days
        video_df['time_diff'] = (current_time - video_df['created_at']) / (1000 * 3600 * 24)
        
        # Ensure all numerical columns are float
        numerical_cols = ['view_count', 'like_count', 'comment_count', 'share_count', 'time_diff']
        for col in numerical_cols:
            video_df[col] = pd.to_numeric(video_df[col], errors='coerce').fillna(0.0).astype(float)
        
        # Calculate trending score
        trending_score = (
            video_df['view_count'] * 1.0 +
            video_df['like_count'] * 2.0 +
            video_df['comment_count'] * 3.0 +
            video_df['share_count'] * 4.0
        ) * np.exp(-time_decay_factor * video_df['time_diff'])
        
        return trending_score.astype(float)