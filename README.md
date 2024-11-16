# Video Recommendation System

## Overview

The Video Recommendation System is designed to provide personalized video recommendations to users based on their viewing history and preferences. It employs a hybrid approach, combining collaborative filtering and content-based filtering techniques to enhance the accuracy and relevance of recommendations.

## Features

- **Collaborative Filtering**: Utilizes user interaction data to recommend videos based on similarities between users.
- **Content-Based Filtering**: Analyzes video features to suggest similar videos based on the content the user has previously watched.
- **Hybrid Recommendations**: Combines both collaborative and content-based recommendations for improved results.
- **User Interaction Logging**: Allows logging of user interactions with videos for continuous improvement of recommendations.
- **Metrics Evaluation**: Provides tools to evaluate the effectiveness of recommendations through various metrics.

## Installation

To set up the Video Recommendation System, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd video-recommendation-system
   ```

2. **Install Required Packages**:
   Ensure you have Python installed, then install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Access**:
   Update the `BASE_URL` and `FLIC_TOKEN` in the code with your API endpoint and token.

4. **Run the Application**:
   Launch the Streamlit application by running:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Input Parameters**:
   - Enter a User ID to get personalized recommendations.
   - Use the slider to select the number of recommendations (1 to 20).

2. **Get Recommendations**:
   Click on "Get Recommendations" to fetch personalized video suggestions for the specified user.

3. **Log Video Clicks**:
   - Enter a User ID and a Video ID to log interactions.
   - Click "Log Click" to record this interaction.

4. **View Recommendation Metrics**:
   - Specify a time window (in days) to evaluate recommendation metrics.
   - Click "Get Metrics" to view a summary of metrics over the specified period.

5. **Metrics History**:
   - Click "Get Metrics History" to visualize historical performance metrics.

## Code Structure

- `app.py`: Main application file that runs the Streamlit interface.
- `preprocess.py`: Contains data preprocessing functions for creating user interaction matrices and video features.
- `utils.py`: Includes the `RecommendationEngine` class that implements recommendation algorithms.
- `evaluation_metrics.py`: Contains classes and functions for logging interactions and evaluating recommendation performance.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please submit a pull request or open an issue.


## Acknowledgments

- [Streamlit](https://streamlit.io/) for building interactive web applications.
- [Scikit-learn](https://scikit-learn.org/stable/) for machine learning utilities, including cosine similarity calculations.

---

This README provides a comprehensive overview of how to use and contribute to the Video Recommendation System, ensuring users can easily navigate its features and functionalities.
