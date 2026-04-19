#!/usr/bin/env python3
import joblib
import pickle

print("=" * 60)
print("CHECKING MODEL CLASSES AND ENCODER")
print("=" * 60)

# Load the model
try:
    model = joblib.load('banana_model.pkl')
    print("\n✓ Model loaded successfully")
    print(f"  Model classes (numeric): {model.classes_}")
except Exception as e:
    print(f"\n✗ Error loading model: {e}")
    exit(1)

# Load the encoder
try:
    with open('banana_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    print(f"\n✓ Encoder loaded successfully")
    print(f"  Encoder type: {type(encoder)}")
    
    # Try to get the class names from the encoder
    if hasattr(encoder, 'classes_'):
        print(f"  Encoder classes: {encoder.classes_}")
        print("\n  MAPPING:")
        for idx, class_name in enumerate(encoder.classes_):
            print(f"    {idx} → {class_name}")
    else:
        print(f"  Encoder attributes: {dir(encoder)}")
except Exception as e:
    print(f"\n✗ Error loading encoder: {e}")

print("\n" + "=" * 60)
