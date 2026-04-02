from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    humidity = data.get('humidity')
    temperature = data.get('temperature')

    # Example logic (replace with your ML model)
    if humidity > 70:
        result = "High Risk"
    else:
        result = "Low Risk"

    return jsonify({
        "prediction": result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)