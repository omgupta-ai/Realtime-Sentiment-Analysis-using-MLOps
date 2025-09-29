# File: streaming/producer.py (Updated)
import time
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import json
import sys

# --- Retry Loop to connect to Kafka ---
print("Producer: Attempting to connect to Kafka...")
producer = None
for i in range(30): # Try for 30 * 5s = 150 seconds
    try:
        producer = KafkaProducer(
            bootstrap_servers='kafka:29092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        print("Producer: Successfully connected to Kafka!")
        break
    except NoBrokersAvailable:
        print(f"Producer: Could not connect to Kafka, retrying in 5 seconds...")
        time.sleep(5)

if not producer:
    print("Producer: Failed to connect to Kafka after multiple retries. Exiting.")
    sys.exit(1)

# --- Load Data and Send Messages ---
df = pd.read_csv('./training.1600000.processed.noemoticon.csv',
                 encoding='latin-1', header=None)
df = df[[5]]
df.columns = ['text']

print("Producer: Starting to send messages...")
for index, row in df.iterrows():
    message = {'text': row['text']}
    producer.send('sentiment-topic', value=message)
    print(f"Producer: Sent: {message['text'][:50]}...")
    time.sleep(0.1)

producer.flush()
print("Producer: All messages sent.")