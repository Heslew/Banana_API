from flask import Flask, request, jsonify
import numpy as np
import pickle
import joblib
import sys

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════
# Load your trained ML model pipeline
# ═══════════════════════════════════════════════════════════════
pipeline = None
model = None
scaler = None
label_encoder = None

try:
    # Load the complete pipeline
    pipeline = joblib.load('banana_model_pipeline.pkl')
    print("✓ ML model pipeline loaded successfully")
    print(f"  Pipeline type: {type(pipeline)}")
    
    # Extract components from pipeline
    if hasattr(pipeline, 'named_steps'):
        # sklearn Pipeline
        if 'scaler' in pipeline.named_steps:
            scaler = pipeline.named_steps['scaler']
            print("✓ Feature scaler loaded from pipeline")
        if 'model' in pipeline.named_steps:
            model = pipeline.named_steps['model']
            print(f"✓ Model loaded from pipeline: {type(model)}")
            if hasattr(model, 'classes_'):
                print(f"  Model classes: {model.classes_}")
    else:
        # Assume it's the model directly
        model = pipeline
        print("✓ Model loaded directly")
        if hasattr(model, 'classes_'):
            print(f"  Model classes: {model.classes_}")
    
except FileNotFoundError as e:
    print(f"⚠ ERROR: banana_model_pipeline.pkl not found: {e}")
    sys.exit(1)
except Exception as e:
    print(f"⚠ ERROR loading pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

@app.route('/')
def home():
    return "Flask Banana Disease Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Get all sensor inputs
        temperature = data.get('temperature', 0)
        humidity = data.get('humidity', 0)
        moisture = data.get('moisture', 0)
        rainfall = data.get('rainfall', 0)
        
        print(f"[DEBUG] Input: temp={temperature}, humidity={humidity}, moisture={moisture}, rainfall={rainfall}")
        
        # Prepare input for ML model
        features = np.array([[temperature, humidity, moisture, rainfall]])
        
        # Normalize features if scaler available
        if scaler is not None:
            try:
                features = scaler.transform(features)
                print(f"[DEBUG] Features normalized")
            except Exception as e:
                print(f"[DEBUG] Scaler error: {e}, using raw features")
        
        # Make prediction using ML model
        if model is not None:
            try:
                # Predict class
                prediction_class = int(model.predict(features)[0])
                print(f"[DEBUG] Prediction class: {prediction_class}")
                
                # Get probabilities and confidence
                probabilities = model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
                
                print(f"[DEBUG] Raw probabilities: {probabilities}")
                print(f"[DEBUG] Max confidence (raw): {confidence}")
                
                # Convert confidence from 0-1 to 0-1 range (already correct)
                # If confidence < 0.5, it's not confident - might need to adjust
                if confidence < 0.33:
                    # No class has >33% confidence, model is uncertain
                    print(f"[DEBUG] Model uncertain (confidence={confidence:.2f})")
                
                # Map numeric class to disease name
                if label_encoder is not None:
                    try:
                        disease_name = label_encoder.inverse_transform([prediction_class])[0].lower()
                    except:
                        disease_map = {0: "healthy", 1: "panama", 2: "sigatoka"}
                        disease_name = disease_map.get(prediction_class, "healthy")
                else:
                    disease_map = {0: "healthy", 1: "panama", 2: "sigatoka"}
                    disease_name = disease_map.get(prediction_class, "healthy")
                
                print(f"[DEBUG] Predicted disease: {disease_name} with confidence {confidence:.2f}")
                
                # Determine risk level based on disease and environmental factors
                if disease_name == "healthy":
                    risk_level = "Healthy"
                elif disease_name == "panama":
                    # Panama prefers high moisture and warmth
                    if moisture > 70 and temperature > 25:
                        risk_level = "High Risk"
                    elif moisture > 60 and temperature > 23:
                        risk_level = "Medium Risk"
                    else:
                        risk_level = "Low Risk"
                elif disease_name == "sigatoka":
                    # Sigatoka prefers high humidity and moderate temp
                    if humidity > 85 and 24 <= temperature <= 28:
                        risk_level = "High Risk"
                    elif humidity > 80 and 23 <= temperature <= 29:
                        risk_level = "Medium Risk"
                    else:
                        risk_level = "Low Risk"
                else:
                    risk_level = "Low Risk"
                
                print(f"[DEBUG] Risk level: {risk_level}")
                
                return jsonify({
                    "prediction": disease_name,
                    "risk_level": risk_level,
                    "confidence": confidence
                })
            except Exception as e:
                print(f"[ERROR] Model prediction error: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to threshold logic
                return _fallback_prediction(temperature, humidity, moisture, rainfall)
        else:
            # Use fallback logic if model not loaded
            print("[WARNING] Model is None, using fallback")
            return _fallback_prediction(temperature, humidity, moisture, rainfall)
            
    except Exception as e:
        print(f"[ERROR] Request processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "prediction": "healthy",
            "risk_level": "Low Risk",
            "confidence": 0.5
        }), 400

def _fallback_prediction(temperature, humidity, moisture, rainfall):
    """Fallback prediction logic based on environmental thresholds"""
    
    print("[INFO] Using fallback threshold-based prediction")
    
    # Determine primary risk based on environmental conditions
    panama_score = 0
    sigatoka_score = 0
    
    # Panama Risk Factors (soil moisture + warmth)
    if moisture > 80 and temperature > 26:
        panama_score += 3
    elif moisture > 70 and temperature > 25:
        panama_score += 2
    elif moisture > 60:
        panama_score += 1
    
    # Sigatoka Risk Factors (humidity + moderate temps)
    if humidity > 85 and 24 <= temperature <= 28:
        sigatoka_score += 3
    elif humidity > 80 and 23 <= temperature <= 29:
        sigatoka_score += 2
    elif humidity > 75:
        sigatoka_score += 1
    
    # Determine disease
    if panama_score > sigatoka_score and panama_score > 0:
        disease_name = "panama"
        if panama_score >= 3:
            risk_level = "High Risk"
        elif panama_score >= 2:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"
    elif sigatoka_score > panama_score and sigatoka_score > 0:
        disease_name = "sigatoka"
        if sigatoka_score >= 3:
            risk_level = "High Risk"
        elif sigatoka_score >= 2:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"
    else:
        disease_name = "healthy"
        risk_level = "Healthy"
    
    # Map score to confidence (0.5-0.9 range for fallback)
    max_score = max(panama_score, sigatoka_score)
    fallback_confidence = 0.5 + (max_score / 10.0)
    fallback_confidence = min(fallback_confidence, 0.90)  # Cap at 0.90
    
    print(f"[INFO] Fallback: {disease_name} {risk_level} confidence={fallback_confidence:.2f}")
    
    return jsonify({
        "prediction": disease_name,
        "risk_level": risk_level,
        "confidence": fallback_confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)