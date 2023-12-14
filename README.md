from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from flask_pymongo import PyMongo
from bson import ObjectId
app = Flask(_name_)
CORS(app)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app.config["MONGO_URI"] = "mongodb://localhost:27017/MOIS"
mongo = PyMongo(app)
def read_messages(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    messages = data.split('-------------------------\n')
    return messages

def preprocess_text(text):
    stop_words = set(stopwords.words('russian'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    processed_text = ' '.join(processed_tokens)
    return processed_text

file1_messages = read_messages('ИЗМЕНЕННЫЕ_СООБЩЕНИЯ1.txt')
file2_messages = read_messages('text.txt')

texts = []
categories = []

for message in file1_messages + file2_messages:
    lines = message.split('\n')
    for line in lines:
        if line.startswith('Категория'):
            category = line.split(': ')[1]
            categories.append(category)
            text = '\n'.join(lines[3:-5])
            preprocessed_text = preprocess_text(text)
            texts.append(preprocessed_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = np.array(categories)

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X, y)

def classify_message(message):
    message_vectorized = vectorizer.transform([message])
    prediction = gb.predict(message_vectorized)
    return prediction[0]

@app.route('/save_message', methods=['POST'])
def save_message():
    data = request.get_json()
    recipient = data['recipient']
    sender = data['sender']
    text = data['text']
    category = data['category']

    message_data = {
        'recipient': recipient,
        'sender': sender,
        'text': text,
        'category': category
    }

    # Сохранение данных в MongoDB
    result = mongo.db.messages.insert_one(message_data)
    
    # Преобразование ObjectId в строку
    inserted_id = str(result.inserted_id)
    message_data['_id'] = inserted_id
    
    return jsonify({'message': 'Message saved successfully', 'data': message_data})


@app.route('/predict_category', methods=['POST'])
def predict_category():
    data = request.get_json()
    message = data['message']
    prediction = classify_message(message)
    print(prediction)
    return jsonify({'category': prediction})

# Получение сообщений по категориям
@app.route('/get_messages_by_category/<category>', methods=['GET'])
def get_messages_by_category(category):
    if category == 'Входящие':  # Если выбрана категория "Входящие"
        messages = mongo.db.messages.find()  # Получить все сообщения
    else:
        messages = mongo.db.messages.find({'category': category})  # Иначе фильтровать по категории

    message_list = [{'recipient': message['recipient'], 'sender': message['sender'], 'text': message['text']} for message in messages]
    return jsonify({'messages': message_list})
if _name_ == '_main_':
    app.run(port=5888, debug=True)
 
# -
