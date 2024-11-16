# evaluation_metrics.py

import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

class RecommendationMetrics:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.recommendations_log = defaultdict(list)  # {user_id: [(timestamp, [video_ids])]}}
        self.clicks_log = defaultdict(list)  # {user_id: [(timestamp, video_id)]}
        self.daily_stats = defaultdict(lambda: {
            'recommendations': 0,
            'clicks': 0,
            'unique_users': set()
        })

    def log_recommendation(self, user_id, video_ids):
        """Log when recommendations are made"""
        timestamp = datetime.now()
        self.recommendations_log[user_id].append((timestamp, video_ids))
        
        # Update daily stats
        date_key = timestamp.date()
        self.daily_stats[date_key]['recommendations'] += len(video_ids)
        self.daily_stats[date_key]['unique_users'].add(user_id)

    def log_click(self, user_id, video_id):
        """Log when a user clicks on a recommended video"""
        timestamp = datetime.now()
        self.clicks_log[user_id].append((timestamp, video_id))
        
        # Update daily stats
        date_key = timestamp.date()
        self.daily_stats[date_key]['clicks'] += 1

    def calculate_user_ctr(self, user_id, time_window=None):
        """Calculate CTR for a specific user within time window"""
        if time_window:
            cutoff_time = datetime.now() - time_window
        else:
            cutoff_time = datetime.min

        # Get relevant recommendations
        total_recs = 0
        recommended_videos = set()
        for timestamp, videos in self.recommendations_log[user_id]:
            if timestamp >= cutoff_time:
                total_recs += len(videos)
                recommended_videos.update(videos)

        # Get relevant clicks
        clicks = set()
        for timestamp, video_id in self.clicks_log[user_id]:
            if timestamp >= cutoff_time and video_id in recommended_videos:
                clicks.add(video_id)

        return len(clicks) / total_recs if total_recs > 0 else 0.0

    def calculate_overall_ctr(self, time_window=None):
        """Calculate overall CTR across all users"""
        total_recommendations = 0
        total_clicks = 0

        for date, stats in self.daily_stats.items():
            if not time_window or datetime.now().date() - date <= time_window:
                total_recommendations += stats['recommendations']
                total_clicks += stats['clicks']

        return total_clicks / total_recommendations if total_recommendations > 0 else 0.0

    def get_metric_summary(self, time_window=None):
        """Get comprehensive metrics summary"""
        summary = {
            'timestamp': datetime.now(),
            'overall_metrics': {
                'ctr': self.calculate_overall_ctr(time_window),
                'total_recommendations': 0,
                'total_clicks': 0,
                'unique_users': 0
            },
            'user_metrics': {
                'ctr': {},
                'engagement_stats': {}
            },
            'daily_stats': {},
            'top_performing_videos': []
        }

        # Calculate per-user metrics
        for user_id in self.recommendations_log.keys():
            user_ctr = self.calculate_user_ctr(user_id, time_window)
            summary['user_metrics']['ctr'][user_id] = user_ctr
            
            # Add user engagement stats
            total_recs = sum(len(videos) for _, videos in self.recommendations_log[user_id])
            total_clicks = len(self.clicks_log[user_id])
            summary['user_metrics']['engagement_stats'][user_id] = {
                'recommendations_received': total_recs,
                'clicks': total_clicks,
                'engagement_rate': total_clicks / total_recs if total_recs > 0 else 0
            }

        # Add daily statistics
        for date, stats in self.daily_stats.items():
            if not time_window or datetime.now().date() - date <= time_window:
                summary['daily_stats'][date.strftime('%Y-%m-%d')] = {
                    'recommendations': stats['recommendations'],
                    'clicks': stats['clicks'],
                    'unique_users': len(stats['unique_users']),
                    'ctr': stats['clicks'] / stats['recommendations'] if stats['recommendations'] > 0 else 0
                }
                summary['overall_metrics']['total_recommendations'] += stats['recommendations']
                summary['overall_metrics']['total_clicks'] += stats['clicks']
                summary['overall_metrics']['unique_users'] += len(stats['unique_users'])

        return summary

    def plot_metric_history(self):
        """Get historical metrics data for plotting"""
        dates = sorted(self.daily_stats.keys())
        history_data = []

        for date in dates:
            stats = self.daily_stats[date]
            history_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'recommendations': stats['recommendations'],
                'clicks': stats['clicks'],
                'unique_users': len(stats['unique_users']),
                'ctr': stats['clicks'] / stats['recommendations'] if stats['recommendations'] > 0 else 0
            })

        return pd.DataFrame(history_data)