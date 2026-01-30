import time
import onnxruntime as ort
import numpy as np
import asyncio
import websockets
import json

DASHBOARD_WS = "ws://dashboard:8000"

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

async def run_inference():
    async with websockets.connect(DASHBOARD_WS) as ws:
        while True:
            dummy_input = np.random.rand(1, 10).astype(np.float32)

            start = time.time()
            session.run(None, {input_name: dummy_input})
            latency = (time.time() - start) * 1000

            metric = {
                "onnx_latency_ms": latency,
                "timestamp": time.time()
            }

            await ws.send(json.dumps(metric))
            print("🧠 ONNX Latency:", latency)

            await asyncio.sleep(3)

asyncio.run(run_inference())
