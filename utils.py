from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

class RecommendationEngine:
    def __init__(self, user_video_matrix, video_features):
        self.user_video_matrix = user_video_matrix.astype(float)  # Ensure float type
        self.video_features = video_features
        self.user_similarity_matrix = None
        self.video_similarity_matrix = None
        
    def compute_similarity_matrices(self):
        """Compute user-user and item-item similarity matrices"""
        # Ensure matrices are float type before computing similarities
        user_matrix = self.user_video_matrix.astype(float)
        
        # Remove non-numeric columns from video features
        video_matrix = self.video_features.drop(['video_id', 'created_at'], axis=1).astype(float)
        
        # Compute similarities
        self.user_similarity_matrix = cosine_similarity(user_matrix)
        self.video_similarity_matrix = cosine_similarity(video_matrix)
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=5):
        """Generate collaborative filtering recommendations"""
        if user_id not in self.user_video_matrix.index:
            return []
        
        user_idx = self.user_video_matrix.index.get_loc(user_id)
        user_similarities = self.user_similarity_matrix[user_idx]
        
        # Convert to float and compute weighted scores
        weighted_scores = np.zeros(self.user_video_matrix.shape[1], dtype=float)
        for other_user_idx in range(len(self.user_video_matrix)):
            if other_user_idx != user_idx:
                weighted_scores += (user_similarities[other_user_idx] * 
                                 self.user_video_matrix.iloc[other_user_idx].values.astype(float))
        
        # Filter out watched videos
        user_interactions = self.user_video_matrix.loc[user_id].values.astype(float)
        weighted_scores[user_interactions > 0] = float('-inf')
        
        # Get top N recommendations
        top_video_indices = np.argsort(weighted_scores)[-n_recommendations:][::-1]
        return [self.user_video_matrix.columns[idx] for idx in top_video_indices]
    
    def get_content_based_recommendations(self, user_id, n_recommendations=5):
        """Generate content-based recommendations"""
        if user_id not in self.user_video_matrix.index:
            return []
        
        # Get user's watched videos and their weights
        user_profile = self.user_video_matrix.loc[user_id].astype(float)
        watched_videos = user_profile[user_profile > 0]
        
        if len(watched_videos) == 0:
            return []
        
        # Calculate weighted average of video features
        feature_cols = self.video_features.drop(['video_id', 'created_at'], axis=1).columns
        user_feature_profile = np.zeros(len(feature_cols), dtype=float)
        
        for video_id, weight in watched_videos.items():
            video_features = self.video_features[
                self.video_features['video_id'] == video_id
            ][feature_cols].iloc[0].values.astype(float)
            user_feature_profile += weight * video_features
        
        user_feature_profile /= len(watched_videos)
        
        # Calculate similarities
        video_matrix = self.video_features[feature_cols].values.astype(float)
        similarities = cosine_similarity([user_feature_profile], video_matrix)[0]
        
        # Filter out watched videos
        watched_indices = [
            self.video_features[self.video_features['video_id'] == vid].index[0]
            for vid in watched_videos.index
        ]
        similarities[watched_indices] = float('-inf')
        
        # Get top N recommendations
        top_indices = np.argsort(similarities)[-n_recommendations:][::-1]
        return [self.video_features.iloc[idx]['video_id'] for idx in top_indices]
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=5, 
                                 collab_weight=0.7, content_weight=0.3):
        """Generate hybrid recommendations"""
        collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations)
        content_recs = self.get_content_based_recommendations(user_id, n_recommendations)
        
        rec_scores = defaultdict(float)
        
        # Weight collaborative recommendations
        for i, video_id in enumerate(collab_recs):
            rec_scores[video_id] += collab_weight * float(n_recommendations - i)
            
        # Weight content-based recommendations
        for i, video_id in enumerate(content_recs):
            rec_scores[video_id] += content_weight * float(n_recommendations - i)
        
        # Sort and return top recommendations
        sorted_recs = sorted(rec_scores.items(), key=lambda x: float(x[1]), reverse=True)
        return [video_id for video_id, _ in sorted_recs[:n_recommendations]]