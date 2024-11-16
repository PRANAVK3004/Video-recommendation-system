from flask import Flask, jsonify, request
from preprocess import DataPreprocessor
from utils import RecommendationEngine
from evaluation_metrics import RecommendationMetrics
from datetime import timedelta

app = Flask(__name__)

# Initialize preprocessor and recommendation engine
BASE_URL = "https://api.socialverseapp.com"
FLIC_TOKEN = "flic_b9c73e760ec8eae0b7468e7916e8a50a8a60ea7e862c32be44927f5a5ca69867"

preprocessor = DataPreprocessor(BASE_URL, FLIC_TOKEN)
user_video_matrix = preprocessor.create_user_interaction_matrix()
video_features = preprocessor.create_video_features()
recommendation_engine = RecommendationEngine(user_video_matrix, video_features)
recommendation_engine.compute_similarity_matrices()

# Initialize metrics
metrics = RecommendationMetrics(preprocessor)

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id').strip()
    n_recommendations = int(request.args.get('n_recommendations', 5))
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    try:
        # Get hybrid recommendations
        recommendations = recommendation_engine.get_hybrid_recommendations(
            user_id,
            n_recommendations=n_recommendations
        )
        
        # Log recommendations for metrics
        metrics.log_recommendation(user_id, recommendations)
        
        # Add trending scores
        trending_scores = preprocessor.calculate_trending_scores(video_features)
        
        # Combine recommendations with video details and trending scores
        recommendation_details = []
        for video_id in recommendations:
            video_data = video_features[video_features['video_id'] == video_id].iloc[0].to_dict()
            video_data['trending_score'] = trending_scores[video_features['video_id'] == video_id].iloc[0]
            recommendation_details.append(video_data)
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendation_details
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/log-click', methods=['POST'])
def log_video_click():
    """Log when a user clicks on a recommended video"""
    data = request.get_json()
    
    if not data or 'user_id' not in data or 'video_id' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
        
    try:
        metrics.log_click(data['user_id'], data['video_id'])
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get current recommendation metrics"""
    try:
        # Get time window from query parameters (optional)
        days = request.args.get('days', type=int)
        time_window = timedelta(days=days) if days else None
        
        summary = metrics.get_metric_summary(time_window)
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics/history', methods=['GET'])
def get_metrics_history():
    """Get historical metrics data"""
    try:
        df = metrics.plot_metric_history()
        return jsonify(df.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)