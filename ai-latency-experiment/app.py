import time
from flask import Flask

app = Flask(__name__)

def simulate_inference(data):
    time.sleep(0.01)

@app.route("/")
def predict():
    start_time = time.time()
    simulate_inference(None)
    latency = (time.time() - start_time) * 1000
    return f"Latency: {latency:.2f} ms"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

