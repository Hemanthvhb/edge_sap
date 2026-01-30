import numpy as np
import time

# ----------------------------
# Cloud workload parameters
# ----------------------------
N = 1000       # Matrix size (adjust for heavier workload)
M = 10         # Number of batches/iterations

# Cloud instance cost: $0.096/hour -> $/second
instance_rate_per_s = 0.096 / 3600

# Store per-batch times
batch_times = []

print(f"Simulating {M} cloud workload batches (N={N})...\n")

cumulative_time = 0.0

# Run batch workload
for i in range(1, M+1):
    start = time.perf_counter()

    # Simulate heavy computation (matrix multiplication)
    matrix_a = np.random.rand(N, N).astype(np.float32)
    matrix_b = np.random.rand(N, N).astype(np.float32)
    _ = np.dot(matrix_a, matrix_b)

    end = time.perf_counter()

    batch_time = end - start
    batch_times.append(batch_time)
    cumulative_time += batch_time

    # Output per-batch compute time in seconds
    print(f"Batch {i}: {batch_time:.2f} s")

# ----------------------------
# Summary
# ----------------------------
payg_cost = cumulative_time * instance_rate_per_s

print("\n--- Summary ---")
print(f"Cumulative Compute Time: {cumulative_time:.2f} s")
print(f"Equivalent Pay-As-You-Go Cost: ${payg_cost:.4f}")
