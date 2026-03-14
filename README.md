# News AI Project

An end-to-end AI pipeline for news analysis combining computer vision (CNN) and natural language processing (NLP).

## Project Structure

```
news_ai_project/
├── data/
│   ├── raw/            # Raw scraped data
│   ├── processed/      # Cleaned and tokenized data
│   └── images/         # News article images
├── models/
│   ├── cnn_model.pth   # Trained CNN weights
│   └── nlp_model.pth   # Trained NLP weights
├── src/
│   ├── data/
│   │   └── scraper.py      # News web scraper
│   ├── vision/
│   │   └── cnn_model.py    # CNN architecture
│   ├── nlp/
│   │   └── nlp_model.py    # LSTM-based NLP model
│   ├── pipeline/
│   │   └── inference.py    # Combined inference pipeline
│   └── utils/
│       └── preprocess.py   # Text & image preprocessing
├── api/
│   └── app.py          # FastAPI REST API
├── notebooks/
│   └── experiment.ipynb
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn api.app:app --reload
```

## API Endpoints

| Method | Endpoint         | Description         |
| ------ | ---------------- | ------------------- |
| GET    | `/health`        | Health check        |
| POST   | `/predict/text`  | Classify news text  |
| POST   | `/predict/image` | Classify news image |
