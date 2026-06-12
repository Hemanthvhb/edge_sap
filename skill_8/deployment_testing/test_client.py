import requests

def test_prediction():
    url = "http://localhost:8000/predict"
    payload = {"data": [1.0, 2.0, 3.0, 4.0]}
    
    try:
        print(f"Sending request to {url} with data {payload['data']}...")
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("Response received successfully!")
            print(f"Prediction: {response.json()['prediction']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_prediction()
