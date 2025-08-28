from flask import Flask, render_template_string, request
import requests

app = Flask(__name__)
API_URL = "http://localhost:8000/predict"
API_KEY = "mysecretkey"

COLUMNS = [
    "AGE", "GENDER", "SMOKING", "FINGER_DISCOLORATION", "MENTAL_STRESS", "EXPOSURE_TO_POLLUTION",
    "LONG_TERM_ILLNESS", "ENERGY_LEVEL", "IMMUNE_WEAKNESS", "BREATHING_ISSUE", "ALCOHOL_CONSUMPTION",
    "THROAT_DISCOMFORT", "OXYGEN_SATURATION", "CHEST_TIGHTNESS", "FAMILY_HISTORY",
    "SMOKING_FAMILY_HISTORY", "STRESS_IMMUNE"
]

HTML_FORM = '''
<!DOCTYPE html>
<html>
<head>
    <title>Lung Cancer Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f7f7f7; }
        .container { max-width: 500px; margin: 40px auto; background: #fff; padding: 30px 40px; border-radius: 10px; box-shadow: 0 2px 8px #ccc; }
        h2 { text-align: center; color: #2c3e50; }
        label { font-weight: bold; margin-top: 10px; display: block; }
        input[type=text], input[type=password] { width: 100%; padding: 8px; margin: 5px 0 15px 0; border: 1px solid #ccc; border-radius: 4px; }
        input[type=submit] { background: #27ae60; color: #fff; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; }
        input[type=submit]:hover { background: #219150; }
        .result { text-align: center; font-size: 18px; margin-top: 20px; }
        .error { color: #c0392b; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Lung Cancer Prediction</h2>
        <form method="post">
            <label for="api_key">API Key</label>
            <input type="password" name="api_key" id="api_key" required>
            {% for col in columns %}
                <label for="{{col}}">{{col.replace('_', ' ').title()}}</label>
                <input type="text" name="{{col}}" id="{{col}}" required>
            {% endfor %}
            <input type="submit" value="Predict">
        </form>
        {% if prediction is not none %}
            <div class="result {{'error' if 'Error' in prediction else ''}}">Prediction: {{ prediction }}</div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            api_key = request.form.get('api_key', '')
            features = [float(request.form[col]) for col in COLUMNS]
            payload = {"instances": [features]}
            headers = {"content-type": "application/json", "X-API-Key": api_key}
            resp = requests.post(API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            pred = resp.json()['predictions'][0][0]
            prediction = 'Positive' if pred > 0.5 else 'Negative'
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template_string(HTML_FORM, prediction=prediction, columns=COLUMNS)

if __name__ == '__main__':
    app.run(debug=True)
