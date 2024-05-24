# youtube-sentiment-analysis

This project aims to analyze the sentiment of YouTube video comments using a pre-trained BERT model and KMeans clustering. The analysis includes visualization of sentiment distributions and clustering results.



## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/BozhengLong/youtube-sentiment-analysis.git
cd youtube-sentiment-analysis
```

2. **Create and activate a virtual or conda environment:**

```bash
conda create --n myenv
conda activate myenv
```

3. **Install the required packages:**
```bash
pip install -r requirements.txt
```


## Usage

1. **Load and initialize the BERT model:**

Modify the `API_KEY` and `VIDEO_ID` in `comment_sentiment_analysis.py` with your own YouTube API key and video ID.

2. **Run the sentiment analysis:**
```bash
python comment_sentiment_analysis.py
```
