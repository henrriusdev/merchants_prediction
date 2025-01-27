from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import Dict, Any, List, Tuple
import pickle
import re

app = Flask(__name__)

DELIVERY_MERCHANT_EXAMPLES = [
    ("Uber Eats", "McDonald's"),
    ("Uber Eats", "Burger King"),
    ("Uber Eats", "Pizza Shop"),
    ("Uber Eats", "TacosLocos Shop"),
    ("DoorDash", "Chipotle"),
    ("DoorDash", "Wendy's"),
    ("GrubHub", "Local Restaurant"),
]

RIDESHARE_MERCHANT_EXAMPLES = [
    ("Uber", "UberX Trip"),
    ("Uber", "Uber Trip"),
    ("Lyft", "Lyft Ride"),
    ("Lyft", "Lyft Trip"),
]

ONLINE_RETAIL_MERCHANT_EXAMPLES = [
    ("Amazon", "Amazon.com"),
    ("Amazon", "AMZN Digital"),
    ("Amazon", "Prime Video"),
    ("Walmart", "Walmart.com"),
    ("Walmart", "WM SUPERCENTER"),
    ("Target", "Target.com"),
    ("Target", "TRGT Online"),
]

SUBSCRIPTION_MERCHANT_EXAMPLES = [
    ("Netflix", "Netflix Monthly"),
    ("Netflix", "Netflix.com"),
    ("Spotify", "Spotify Premium"),
    ("Spotify", "Spotify.com"),
]

PHYSICAL_STORE_MERCHANT_EXAMPLES = [
    ("McDonald's", "MCD"),
    ("McDonald's", "McD's"),
    ("Starbucks", "SBUX"),
    ("Starbucks", "SB STORE"),
    ("Shell", "Shell Oil"),
    ("Shell", "Shell Gas"),
    ("Exxon", "Exxon Gas"),
    ("BP", "BP Gas Station"),
    ("CVS", "CVS Pharmacy"),
    ("CVS", "CVS/PHARM"),
    ("Walgreens", "WALGR"),
    ("Walgreens", "WAG STORE"),
]

OTHER_DELIVERY_MERCHANT_EXAMPLES = [
    ("Instacart", "INST WALMART"),
    ("Instacart", "INST COSTCO"),
    ("Shipt", "SHIPT TARGET"),
    ("Shipt", "SHIPT.COM")
]

MERCHANT_TRAINING_PAIRS = (
    DELIVERY_MERCHANT_EXAMPLES +
    RIDESHARE_MERCHANT_EXAMPLES +
    ONLINE_RETAIL_MERCHANT_EXAMPLES +
    SUBSCRIPTION_MERCHANT_EXAMPLES +
    PHYSICAL_STORE_MERCHANT_EXAMPLES +
    OTHER_DELIVERY_MERCHANT_EXAMPLES
)

PERSONAL_PAYMENT_EXAMPLES = [
    ("John Smith", "Payment"),
    ("John Smith", "Transfer"),
    ("Mary Johnson", "Venmo Payment"),
    ("Mary Johnson", "Zelle Transfer"),
    ("Robert Brown", "Cash App"),
    ("Robert Brown", "Payment Sent"),
    ("Emily Davis", "Money Transfer"),
    ("Michael Wilson", "Bill Payment"),
    ("Sarah Davis", "Rent Payment"),
    ("James Taylor", "Check Payment"),
    ("Jennifer White", "Monthly Payment"),
    ("Jose Rodriguez", "Transfer Out"),
    ("Maria Garcia", "Payment Sent"),
    ("David Martinez", "Bill Pay"),
    ("Lisa Thompson", "Rent"),
    ("William Taylor", "Payment"),
    ("Patricia Moore", "Transfer"),
    ("Robert Lee", "Venmo"),
    ("Elizabeth Clark", "Zelle"),
    ("Joseph Wright", "Cash App"),
]

PERSONAL_INTERNATIONAL_EXAMPLES = [
    ("Juan Carlos", "Payment"),
    ("Maria Elena", "Transfer"),
    ("François Dubois", "Payment"),
    ("Hans Schmidt", "Transfer"),
    ("Giuseppe Romano", "Payment"),
    ("Yuki Tanaka", "Transfer"),
    ("Wei Chen", "Payment"),
    ("Claudia Monroy", "Transfer")
]

PERSONAL_TRAINING_PAIRS = PERSONAL_PAYMENT_EXAMPLES + PERSONAL_INTERNATIONAL_EXAMPLES

DELIVERY_SERVICE_NAMES = {
    "uber eats": "Uber Eats",
    "doordash": "DoorDash",
    "grubhub": "GrubHub",
    "postmates": "Postmates",
    "instacart": "Instacart",
    "shipt": "Shipt"
}

FINANCIAL_TRANSACTION_PATTERNS = [
    r'loan(\s+repayment)?',
    r'mortgage(\s+payment)?',
    r'student\s+loan',
    r'credit\s+card(\s+payment)?',
    r'(automatic|auto)?\s*payment',
    r'direct\s+dep(osit)?',
    r'(bank\s+)?transfer',
    r'(ach|wire)(\s+transfer)?',
    r'intrst\s+pymnt',
    r'int(ere)?st\s+payment',
    r'plaid',
    r'thank\s+you',
    r'pymt',
    r'payment\s+received',
    r'dep(osit)?',
    r'direct\s+dep(osit)?',
    r'mobile\s+dep(osit)?',
    r'atm\s+dep(osit)?',
    r'cash\s+dep(osit)?',
    r'check\s+dep(osit)?',
    r'bill\s+pay(ment)?',
    r'autopay',
    r'auto\s+pay',
    r'recurring\s+payment',
    r'online\s+payment',
    r'web\s+payment',
    r'mobile\s+payment',
    r'balance\s+transfer'
]

def prepare_transaction_features(merchant_name: str, transaction_name: str) -> str:
    """
    Prepare features by combining merchant and transaction names for classification.
    Uses a special separator to maintain distinction between the two names.
    """
    normalized_merchant = merchant_name.lower().strip() if merchant_name else ""
    normalized_transaction = transaction_name.lower().strip() if transaction_name else ""
    return f"{normalized_merchant} ||| {normalized_transaction}"

def create_merchant_classifier() -> Tuple[RandomForestClassifier, TfidfVectorizer]:
    """
    Create and train the merchant classification model using the training data.
    Returns the trained model and the TF-IDF vectorizer.
    """
    
    text_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        analyzer='char_wb',  
        stop_words='english'
    )
    
    merchant_features = [prepare_transaction_features(m, t) for m, t in MERCHANT_TRAINING_PAIRS]
    personal_features = [prepare_transaction_features(m, t) for m, t in PERSONAL_TRAINING_PAIRS]
    
    all_features = merchant_features + personal_features
    training_labels = [1] * len(merchant_features) + [0] * len(personal_features)
    
    feature_matrix = text_vectorizer.fit_transform(all_features)
    
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    classifier.fit(feature_matrix, training_labels)
    
    return classifier, text_vectorizer

def is_financial_transaction(text: str) -> bool:
    """
    Check if the given text matches any known financial transaction patterns.
    """
    return any(bool(re.search(pattern, text.lower())) for pattern in FINANCIAL_TRANSACTION_PATTERNS)

def get_prediction_reason(is_merchant: bool, confidence: float) -> str:
    """
    Generate a human-readable reason for the prediction based on confidence level.
    """
    if is_merchant:
        if confidence > 0.9:
            return "Strong merchant pattern match"
        elif confidence > 0.7:
            return "Moderate merchant pattern match"
        return "Weak merchant pattern match"
    else:
        if confidence > 0.9:
            return "Strong personal transaction pattern"
        elif confidence > 0.7:
            return "Moderate personal transaction pattern"
        return "Weak personal transaction pattern"

merchant_classifier, feature_vectorizer = create_merchant_classifier()
with open('merchant_classifier_v3.pkl', 'wb') as model_file:
    pickle.dump((merchant_classifier, feature_vectorizer), model_file)

@app.route('/predict', methods=['POST'])
def predict() -> Any:
    """
    Predict whether a transaction involves a merchant based on merchant and transaction names.
    """
    try:
        request_data: Dict[str, Any] = request.get_json()
        merchant_name: str = request_data.get("merchant_name", "")
        transaction_name: str = request_data.get("transaction_name", "")

        if not transaction_name:
            return jsonify({"error": "'transaction_name' is required in the request body."}), 400

        merchant_lower = merchant_name.lower() if merchant_name else ""
        transaction_lower = transaction_name.lower()

        if any(is_financial_transaction(name) for name in [merchant_lower, transaction_lower] if name):
            return jsonify({
                "merchant_name": merchant_name,
                "transaction_name": transaction_name,
                "is_merchant": False,
                "reason": "Common financial transaction pattern",
                "confidence": 0.95
            })

        for service_name, proper_name in DELIVERY_SERVICE_NAMES.items():
            if service_name in merchant_lower or service_name in transaction_lower:
                return jsonify({
                    "merchant_name": proper_name,
                    "transaction_name": transaction_name,
                    "is_merchant": True,
                    "reason": f"Known delivery service: {proper_name}",
                    "confidence": 1.0
                })

        with open('merchant_classifier_v3.pkl', 'rb') as model_file:
            merchant_classifier, feature_vectorizer = pickle.load(model_file)

        features = [prepare_transaction_features(merchant_name, transaction_name)]
        feature_matrix = feature_vectorizer.transform(features)
        
        is_merchant = merchant_classifier.predict(feature_matrix)[0]
        probabilities = merchant_classifier.predict_proba(feature_matrix)[0]
        confidence = probabilities[1] if is_merchant else probabilities[0]
        
        reason = get_prediction_reason(is_merchant, confidence)

        return jsonify({
            "merchant_name": merchant_name,
            "transaction_name": transaction_name,
            "is_merchant": bool(is_merchant),
            "reason": reason,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
