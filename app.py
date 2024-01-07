from flask import Flask, render_template, request, redirect, url_for
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the pickled model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']

        if not message.strip():  # Check if the message is empty or contains only whitespaces
            # Display an alert notification using JavaScript
            return '''
            <script>
                alert("Please enter a message.");
                window.location.href = "/";
            </script>
            '''

        message_features = vectorizer.transform([message])
        prediction = model.predict(message_features)

        result = "Spam" if prediction[0] == 0 else "Ham"

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
