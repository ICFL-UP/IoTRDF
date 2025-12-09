import requests
import json

print("🔍 Deep API Discovery...")

# Get the OpenAPI specification to understand the exact format
try:
    r = requests.get("http://localhost:8000/openapi.json")
    openapi_spec = r.json()
    
    print("📋 Available paths:")
    for path, methods in openapi_spec['paths'].items():
        print(f"  {path}: {list(methods.keys())}")
    
    # Check the predict endpoint specifically
    if '/predict' in openapi_spec['paths']:
        predict_info = openapi_spec['paths']['/predict']
        if 'post' in predict_info:
            post_info = predict_info['post']
            print(f"\n🎯 Predict endpoint details:")
            print(f"   Summary: {post_info.get('summary', 'N/A')}")
            print(f"   Description: {post_info.get('description', 'N/A')}")
            
            # Check request body schema
            if 'requestBody' in post_info:
                content = post_info['requestBody']['content']
                if 'application/json' in content:
                    schema = content['application/json']['schema']
                    print(f"   Expected schema: {json.dumps(schema, indent=4)}")
                    
except Exception as e:
    print(f"Error reading OpenAPI spec: {e}")

# Try to get more details from the error response
print("\n🔍 Getting detailed error information...")
try:
    payload = {"x": [0.1, 0.2, 0.3, 0.4]}
    r = requests.post("http://localhost:8000/predict", json=payload)
    print(f"Error response: {r.text}")
    print(f"Error headers: {dict(r.headers)}")
except Exception as e:
    print(f"Error: {e}")

# Test with different feature dimensions
print("\n🔍 Testing different feature dimensions:")
for dim in [1, 4, 10, 32, 64, 100]:
    try:
        payload = {"x": [0.1] * dim}
        r = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
        print(f"Dimension {dim}: Status {r.status_code}")
        if r.status_code == 200:
            print(f"✅ SUCCESS with {dim} features! Response: {r.json()}")
            break
    except Exception as e:
        print(f"Dimension {dim}: Error {e}")
