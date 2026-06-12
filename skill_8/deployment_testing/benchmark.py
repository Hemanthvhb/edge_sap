import time
import requests
import statistics
import subprocess
import json

def measure_cold_start(url):
    print(f"Measuring cold start for {url}...")
    start_time = time.time()
    while True:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                end_time = time.time()
                cold_start_time = end_time - start_time
                print(f"Cold start successful! Time: {cold_start_time:.2f} seconds")
                return cold_start_time
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        if time.time() - start_time > 60:
            print("Timeout waiting for microservice to start.")
            return None
        time.sleep(0.5)

def measure_latency(url, data, num_requests=100):
    print(f"Measuring latency over {num_requests} requests...")
    latencies = []
    for _ in range(num_requests):
        start_time = time.perf_counter()
        response = requests.post(f"{url}/predict", json={"data": data})
        end_time = time.perf_counter()
        if response.status_code == 200:
            latencies.append((end_time - start_time) * 1000) # ms
        else:
            print(f"Error: {response.status_code} - {response.text}")
    
    if latencies:
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        print(f"Average Latency: {avg_latency:.2f} ms")
        print(f"P95 Latency: {p95_latency:.2f} ms")
        return avg_latency, p95_latency
    return None, None

def get_docker_stats(container_name):
    try:
        result = subprocess.run(
            ["docker", "stats", container_name, "--no-stream", "--format", "{{.CPUPerc}}\t{{.MemUsage}}"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "N/A"

if __name__ == "__main__":
    BASE_URL = "http://localhost:8000"
    TEST_DATA = [1.0, 2.0, 3.0, 4.0]
    
    # Note: This script assumes the container is being started or is already running.
    # For a true cold start measurement, run this script right after 'docker run'.
    
    cold_start = measure_cold_start(BASE_URL)
    avg_l, p95_l = measure_latency(BASE_URL, TEST_DATA)
    stats = get_docker_stats("onnx-service") # Assumes container name is onnx-service
    
    results = {
        "cold_start_time_s": cold_start,
        "avg_latency_ms": avg_l,
        "p95_latency_ms": p95_l,
        "docker_stats": stats
    }
    
    with open("performance_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n--- Summary ---")
    print(f"Cold Start: {cold_start:.2f}s" if cold_start else "Cold Start: N/A")
    print(f"Avg Latency: {avg_l:.2f}ms" if avg_l else "Avg Latency: N/A")
    print(f"P95 Latency: {p95_l:.2f}ms" if p95_l else "P95 Latency: N/A")
    print(f"Docker Stats: {stats}")
