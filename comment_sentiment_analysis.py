"""
Author: Bozheng Long
Created Date: 2024-05-22
Last Modified Date: 2024-05-24
Description:
This file is used to analyze the emotion distribution of the comments in a YouTube video.
"""

# %%
import googleapiclient.discovery
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torchviz import make_dot

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from transformers import BertTokenizer, BertModel
import joblib
import requests
from IPython.display import Image

from train import preprocess_with_bert, SimpleNN

# %%

# labels is the list of emotion labels
labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 
        'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 
        'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

# define the emotion groups and each group's corresponding emotions
emotion_groups = {
    'positive': ['admiration', 'amusement', 'approval', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief', 'caring', 'desire'],
    'negative': ['anger', 'annoyance', 'confusion', 'disappointment', 'disapproval', 'disgust', 'fear', 'grief', 'remorse', 'sadness', 'embarrassment', 'nervousness'],
    'neutral': ['curiosity', 'excitement', 'surprise', 'realization', 'neutral']
}

# assign colors to each emotion group
color_mapping = {
    'positive': '#66c2a5',
    'negative': '#fc8d62',
    'neutral': '#8da0cb'
}

# generate a color mapping for each emotion
emotion_colors = {}
for group, emotions in emotion_groups.items():
    for emotion in emotions:
        emotion_colors[emotion] = color_mapping[group]


def plot_emotion_weights(weights, title):
    plt.figure(figsize=(14, 7))
    colors = [emotion_colors[col] for col in weights.index]
    weights.plot(kind='bar', color=colors)
    plt.xlabel('Emotions')
    plt.ylabel('Average Weights')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

def load_and_init_bert(device):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    return tokenizer, bert_model

def get_youtube_comments(api_key, video_id):
    # construct the youtube service
    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None

    while True:
        # get the comments from the video
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    
    return comments

def plot_model_structure(model, device, input_dim):
    # plot the model structure

    x = torch.randn(1, input_dim).to(device)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render("model_structure")

def cluster_analysis(df, num_clusters=5):

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(df)
    df['cluster'] = clusters

    # cluster_centers = kmeans.cluster_centers_
    # cluster_centers_df = pd.DataFrame(cluster_centers, columns=df.columns[:-1])  # exclude the 'cluster' column

    # # visualize the emotion weights of each cluster center
    # for cluster in range(num_clusters):
    #     plot_emotion_weights(cluster_centers_df.loc[cluster], f'Cluster {cluster} Center Emotion Weights')

    # visualize the average emotion weights of each cluster
    for cluster in range(num_clusters):
        cluster_data = df[df['cluster'] == cluster].drop(columns=['cluster'])
        plot_emotion_weights(cluster_data.mean(), f'Average Emotion Weights for Cluster {cluster}')

def plot_tsne(df, labels):
    '''
    use t-SNE to reduce the dimensionality of the data to 2D and plot the clusters
    '''
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(df[labels])
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(14, 7))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="cluster",
        palette=sns.color_palette("hsv", len(df['cluster'].unique())),
        data=df,
        legend="full",
        alpha=0.7
    )
    plt.title('t-SNE Plot of Clusters')
    plt.show()

def plot_emotion_pie_chart(df, emotion_groups):
    '''
    plot a pie chart of the overall emotion weights distribution
    '''
    positive_weight = df[emotion_groups['positive']].sum().sum()
    negative_weight = df[emotion_groups['negative']].sum().sum()
    neutral_weight = df[emotion_groups['neutral']].sum().sum()

    weights = [positive_weight, negative_weight, neutral_weight]
    labels = ['Positive', 'Negative', 'Neutral']
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']

    plt.figure(figsize=(8, 8))
    plt.pie(weights, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Overall Emotion Weights Distribution')
    plt.show()

# %%
def main():
# %%
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # define the model hyperparameters(used in the SimpleNN class)
    input_dim = 768  # 768 is the BERT embedding size
    hidden_dim = 64
    output_dim = 28  # 28 is the number of emotion labels
    drop_prob = 0.25

    model = SimpleNN(input_dim, hidden_dim, output_dim, drop_prob).to(device)
    model.load_state_dict(torch.load('model.pt', map_location=device))
    model.eval()

    # %%
    # plot the model structure
    plot_model_structure(model, device, input_dim)
    Image(filename="model_structure.png")

    # %%
    tokenizer, bert_model = load_and_init_bert(device)

    API_KEY = '...'   # fill in your own API key
    VIDEO_ID = '...'    # fill in the youtube video id

    comments = get_youtube_comments(API_KEY, VIDEO_ID)
    print(comments[:5])  # print the first 5 comments
    for i, comment in enumerate(comments[:5], 1):
        print(f'Comment {i}: {comment}\n')
    print(f'Total comments: {len(comments)}')

    # %%
    # use BERT to embed each comment, this step may take a long time
    # I strongly recommend you to save the results to the local environment
    # If you have already run this step, you can skip it
    bert_embeddings = [preprocess_with_bert(tokenizer, bert_model, comment, device) for comment in comments]
    # save the embeddings to the local environment
    joblib.dump(bert_embeddings, 'ytb_video_comment_bert_embeddings.joblib')
    # %%
    # load the embeddings from the local environment
    bert_embeddings = joblib.load('ytb_video_comment_bert_embeddings.joblib')

    X = pd.DataFrame(bert_embeddings)
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)

    # predict the emotion weights for each comment
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    # %%
    # transform the predicted emotion weights to a DataFrame
    df = pd.DataFrame(y_pred, columns=labels)

    # perform clustering analysis
    cluster_analysis(df, num_clusters=5)
    
    average_weights = df.drop(columns=['cluster']).mean()
    plot_emotion_weights(average_weights, 'Average Emotion Weights Across All Comments')
    plot_tsne(df, labels)
    plot_emotion_pie_chart(df, emotion_groups)

# %%
if __name__ == "__main__":
    main()