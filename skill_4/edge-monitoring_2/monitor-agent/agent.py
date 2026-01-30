import psutil
import time
import json
import asyncio
import websockets

DASHBOARD_WS = "ws://dashboard:8000"

async def send_metrics():
    async with websockets.connect(DASHBOARD_WS) as ws:
        while True:
            metrics = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "timestamp": time.time()
            }

            await ws.send(json.dumps(metrics))
            print("📡 Sent:", metrics)

            await asyncio.sleep(2)

asyncio.run(send_metrics())
