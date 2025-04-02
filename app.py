from flask import Flask, render_template, request
from googleapiclient.discovery import build
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import requests
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

API_KEY = API_ENV_YT
youtube = build('youtube', 'v3', developerKey=API_KEY)

HF_API_TOKEN = API_ENV_HF
HF_API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def query_hf_api(payload):
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        print(f"HF API Raw Response: {response.text}")
        return response.json()
    except Exception as e:
        print(f"HF API Error: {e}")
        return {"error": str(e)}

def get_playlist_data(playlist_id):
    videos = []
    next_page_token = None

    # Fetch playlist details (including title and thumbnail)
    playlist_response = youtube.playlists().list(
        part='snippet',
        id=playlist_id
    ).execute()
    playlist_title = "Unknown Playlist"
    playlist_thumbnail = None
    if playlist_response['items']:
        snippet = playlist_response['items'][0]['snippet']
        playlist_title = snippet.get('title', playlist_title)
        playlist_thumbnail = snippet.get('thumbnails', {}).get('default', {}).get('url', None)

    while True:
        res = youtube.playlistItems().list(
            playlistId=playlist_id,
            part='snippet',
            maxResults=50,
            pageToken=next_page_token
        ).execute()
        videos += res['items']
        next_page_token = res.get('nextPageToken')
        if not next_page_token:
            break

    video_data = []
    channel_image = None
    channel_name = None

    for video in videos:
        video_id = video['snippet']['resourceId']['videoId']
        stats = youtube.videos().list(part='statistics,snippet', id=video_id).execute()
        if stats['items']:
            video_item = stats['items'][0]
            statistics = video_item['statistics']
            snippet = video_item['snippet']
            video_data.append({
                'title': snippet['title'],
                'views': int(statistics.get('viewCount', 0)),
                'likes': int(statistics.get('likeCount', 0)),
                'comments': int(statistics.get('commentCount', 0))
            })
            if not channel_name:
                channel_name = snippet.get('channelTitle', 'Unknown Channel')
                channel_id = snippet.get('channelId')
                if channel_id:
                    channel_info = youtube.channels().list(
                        part='snippet', id=channel_id
                    ).execute()
                    if channel_info['items']:
                        channel_image = channel_info['items'][0]['snippet']['thumbnails']['default']['url']

    return pd.DataFrame(video_data), channel_image, channel_name, playlist_title, playlist_thumbnail

def create_plot(df, y_column, title, top_n=20, color="#4fc3f7"):
    plt.switch_backend('Agg')
    df_sorted = df.sort_values(y_column, ascending=False).head(top_n)
    truncated_titles = [t[:40] + '...' if len(t) > 40 else t for t in df_sorted['title']]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#141e30")
    ax.set_facecolor("#141e30")
    ax.bar(truncated_titles, df_sorted[y_column], color=color)
    ax.set_title(title, color="white", fontsize=16)
    ax.tick_params(axis='x', rotation=90, colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def create_scatter_plot(df, x_col, y_col, title):
    plt.switch_backend('Agg')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#141e30")
    ax.set_facecolor("#141e30")
    ax.scatter(df[x_col], df[y_col], alpha=0.6, edgecolors='w', linewidth=0.5, color="#e74c3c")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Views (log scale)', color="white", fontsize=12)
    ax.set_ylabel('Likes (log scale)', color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    ax.tick_params(colors='white')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def generate_summary(df):
    total_videos = len(df)
    if total_videos == 0:
        return "No videos found in this playlist."

    top_video = df.loc[df['views'].idxmax()]
    avg_views = df['views'].mean()
    avg_likes = df['likes'].mean()
    df['engagement'] = df['likes'] / df['views']
    avg_engagement = df['engagement'].mean() * 100  

    prompt = (
        f"Analyze this YouTube playlist data:\n"
        f"- Playlist contains {total_videos} videos\n"
        f"- Playlist Top Video: \"{top_video['title']}\" with {top_video['views']} views and {top_video['likes']} likes\n"
        f"- Average views: {avg_views:.0f}\n"
        f"- Average likes: {avg_likes:.0f}\n"
        f"- Average engagement rate: {avg_engagement:.2f}%\n"
        "Provide a concise summary with key performance insights."
    )
    print(f"Sending prompt to API: {prompt}")
    response_json = query_hf_api({"inputs": prompt})
    print(f"API Response content: {json.dumps(response_json, indent=2)}")

    if isinstance(response_json, list) and response_json and 'summary_text' in response_json[0]:
        return response_json[0]['summary_text']
    elif isinstance(response_json, dict) and 'summary_text' in response_json:
        return response_json['summary_text']
    else:
        return (f"This playlist contains {total_videos} videos. The top video is \"{top_video['title']}\" "
                f"with {top_video['views']} views and {top_video['likes']} likes. On average, videos receive "
                f"{avg_views:.0f} views and {avg_likes:.0f} likes with an engagement rate of {avg_engagement:.2f}%.")

@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    if request.method == 'POST':
        try:
            playlist_url = request.form['playlist_url']
            if 'list=' not in playlist_url:
                return render_template('index.html', error="Invalid YouTube playlist URL. Please include 'list=' in the URL.")
            playlist_id = playlist_url.split('list=')[1].split('&')[0] if '&' in playlist_url.split('list=')[1] else playlist_url.split('list=')[1]
            print(f"Fetching data for playlist ID: {playlist_id}")
            df, channel_image, channel_name, playlist_title, playlist_thumbnail = get_playlist_data(playlist_id)
            if df.empty:
                return render_template('index.html', error="No videos found in this playlist or the playlist might be private.")

            views_plot = create_plot(df, 'views', 'Top 20 Videos by Views', color="#4fc3f7")
            likes_plot = create_plot(df, 'likes', 'Top 20 Videos by Likes', color="#81d4fa")
            scatter_plot = create_scatter_plot(df, 'views', 'likes', 'Views vs Likes Correlation (Log Scale)')
            ai_summary = generate_summary(df)

            return render_template('results.html',
                                   views_plot=views_plot,
                                   likes_plot=likes_plot,
                                   scatter_plot=scatter_plot,
                                   summary=ai_summary,
                                   channel_image=channel_image,
                                   channel_name=channel_name,
                                   playlist_title=playlist_title,
                                   playlist_thumbnail=playlist_thumbnail)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(f"Error processing request: {e}")
    return render_template('index.html', error=error_message)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
