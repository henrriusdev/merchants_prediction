# Merchant ML Classifier

A Flask-based API that classifies transaction names as either merchant or personal names using machine learning.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application locally:
```bash
python app.py
```

## API Usage

Send POST requests to `/predict` endpoint with JSON body:

```json
{
    "merchant_name": "your merchant name",
    "transaction_name": "your transaction name"
}
```

## Deployment to fly.io

1. Install flyctl if not already installed
2. Login to fly.io:
```bash
fly auth login
```

3. Deploy the application:
```bash
fly deploy
```

## Testing the API

Example curl request:
```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"merchant_name": "Walmart", "transaction_name": "WM SUPERCENTER"}'
```
