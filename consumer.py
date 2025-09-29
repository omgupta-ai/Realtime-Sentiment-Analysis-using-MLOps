# File: streaming/consumer.py (Updated)
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import json
import requests
import time
import sys

API_ENDPOINT = "http://sentiment-api:8000/predict"

# --- Retry Loop to connect to Kafka ---
print("Consumer: Attempting to connect to Kafka...")
consumer = None
for i in range(30): # Try for 30 * 5s = 150 seconds
    try:
        consumer = KafkaConsumer(
            'sentiment-topic',
            bootstrap_servers='kafka:29092',
            auto_offset_reset='earliest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        print("Consumer: Successfully connected to Kafka!")
        break
    except NoBrokersAvailable:
        print(f"Consumer: Could not connect to Kafka, retrying in 5 seconds...")
        time.sleep(5)

if not consumer:
    print("Consumer: Failed to connect to Kafka after multiple retries. Exiting.")
    sys.exit(1)

# --- Listen for Messages and Process ---
print("Consumer: Listening for messages...")
for message in consumer:
    try:
        tweet_text = message.value['text']
        response = requests.post(API_ENDPOINT, json={"text": tweet_text})

        if response.status_code == 200:
            prediction = response.json()
            print(f"TWEET: {tweet_text[:80]:<80} | PREDICTION: {prediction['sentiment']}")
        else:
            print(f"API Error: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")