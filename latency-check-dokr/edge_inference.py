import time
import numpy as np
import tensorflow as tf

# Build a small edge-style model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile model (not training here, just inference)
model.compile(optimizer='adam', loss='mse')

# Dummy single-user input (one request)
input_data = np.random.rand(1, 16).astype(np.float32)

# Warm-up run (important to avoid cold-start bias)
model.predict(input_data)

# Measure raw compute time
start_time = time.perf_counter()
model.predict(input_data)
end_time = time.perf_counter()

latency_ms = (end_time - start_time) * 1000
print(f"Inference Latency: {latency_ms:.3f} ms")
