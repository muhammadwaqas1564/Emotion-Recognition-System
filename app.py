import os
import pickle
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import model_from_json
from flask import jsonify

# Load the text emotion recognition model
lstm_model = tf.keras.models.load_model("Text model/lstm_model.h5")
with open('Text model/tokenizer.pickle', 'rb') as handle:
    word_tokenizer = pickle.load(handle)
with open('Text model/encoder.pickle', 'rb') as handle:
    encoder_text = pickle.load(handle)

# Load the speech emotion recognition model
json_file = open('Speech model/CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
speech_model = model_from_json(loaded_model_json)
speech_model.load_weights("Speech model/CNN_model_weights.weights.h5")
with open('Speech model/scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)
with open('Speech model/encoder2.pickle', 'rb') as f:
    encoder_speech = pickle.load(f)

max_len = 35  # Maximum length used in text model training

# Initialize Flask app
app = Flask(__name__)

# Preprocessing for text input
def preprocess_text(text):
    return text.lower()

# Feature extraction for speech input
def zcr(data, frame_length, hop_length):
    return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))

def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool=True):
    mfcc_feat = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_feat.T) if not flatten else np.ravel(mfcc_feat.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.hstack((zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)))
    return result

def get_predict_feat(path):
    data, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(data)
    if res.shape[0] != scaler2.n_features_in_:
        res = np.pad(res, (0, scaler2.n_features_in_ - res.shape[0]), 'constant')
    res = np.reshape(res, newshape=(1, res.size))
    return np.expand_dims(scaler2.transform(res), axis=2)

def predict_text_emotion(input_text):
    processed_text = preprocess_text(input_text)
    text_seq = word_tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(text_seq, padding='post', maxlen=max_len)
    prediction = lstm_model.predict(padded_seq)
    predicted_label = np.argmax(prediction, axis=1)
    return encoder_text.inverse_transform(predicted_label)[0]

def predict_speech_emotion(file_path):
    res = get_predict_feat(file_path)
    predictions = speech_model.predict(res)
    return encoder_speech.inverse_transform(predictions)[0][0]

# Route for home page
from flask import jsonify

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    prediction_speech = None

    if request.method == 'POST':
        # For text input
        if 'text' in request.form and request.form['text']:
            input_text = request.form['text']
            prediction_text = predict_text_emotion(input_text)

        # For speech input
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file:
                file_path = os.path.join("uploads", file.filename)
                try:
                    file.save(file_path)  # Save the uploaded file
                    prediction_speech = predict_speech_emotion(file_path)
                except Exception as e:
                    prediction_speech = f"Error in predicting speech emotion: {e}"
                finally:
                    os.remove(file_path)  # Remove file after processing

        # Return predictions as JSON response
        return jsonify(prediction_text=prediction_text, prediction_speech=prediction_speech)

    return render_template('index.html', prediction_text=prediction_text, prediction_speech=prediction_speech)


if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
