import tensorflow as tf, psutil, time
model = tf.keras.Sequential([tf.keras.layers.Dense(32), tf.keras.layers.Dense(10)])  # small model
x = tf.random.normal((1, 128))
p = psutil.Process()
start_cpu = p.cpu_percent()
start_mem = p.memory_info().rss / (1024**2)
for _ in range(1000): model(x)
time.sleep(1)
end_cpu = p.cpu_percent()
end_mem = p.memory_info().rss / (1024**2)#”RESIDENT  SET SIZE”
print("CPU Usage:", end_cpu-start_cpu, "%")
print("RAM Usage:", end_mem-start_mem, "MB")
