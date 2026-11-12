import asyncio
import logging
import os
import time
from aiohttp import web
from datetime import datetime


class Queue:
    def __init__(self) -> None:
        self.is_free = {}
        self.backend_port = None
        self.drop_after = int(os.environ["DROP_AFTER"]) # ms, zero means no drop
        self.total_requests = 0
        self.total_dropped = 0
        self.queue_name = f"stage-{os.environ['STAGE_INDEX']}"
        self.queue = asyncio.Queue()
        self._batch_lock = asyncio.Lock()
        self.logger = logging.getLogger()
        logging.basicConfig(level = logging.INFO)
    
    async def receive(self, data: dict):
        self.total_requests += 1
        await self.queue.put({f"arrival-{self.queue_name}": time.time(), f"RID-{self.queue_name}": self.total_requests, **data})
        self.logger.info(f"datetime={str(datetime.now())}, enqueued new request, {self.total_requests}")

        return {"received": True}
    
    async def fetch(self, data):
        st = time.perf_counter()
        async with self._batch_lock:
            batch_size = data["batch_size"]
            batch_timeout = data["batch_timeout"]
            batch = []
            t = time.perf_counter()
            while len(batch) < batch_size:
                try:
                    query = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.002)
                    if 1000 * (time.perf_counter() - t) > batch_timeout:
                        break
                    continue
                if self.drop_after:
                    if 1000 * (time.time() - query[f"arrival-stage-0"]) >= self.drop_after:
                        self.total_dropped += 1
                        continue
                batch.append(query)
            self.logger.info(
                f"datetime={str(datetime.now())}, fetch request, bs={batch_size}, bt={batch_timeout}"
            )
            if batch:
                self.logger.info(
                    f"datetime={str(datetime.now())}, b={len(batch)}, first={batch[0][f"RID-{self.queue_name}"]}, last={batch[-1][f"RID-{self.queue_name}"]}, took={round(1000 * (time.perf_counter() - t), 3)}, overal={round(1000 * (time.perf_counter() - st), 3)}"
                )
            for i in range(len(batch)):
                batch[i][f"leaving-{self.queue_name}"] = time.time()
            return batch

queue = Queue()


async def submit(request):
    data = await request.json()
    return web.json_response(await queue.receive(data))

async def fetch(request):
    data = await request.json()
    response = await queue.fetch(data)
    return web.json_response(response)


async def export_request_count(request):
    content = "# HELP queue_requests_total Total number of requests\n"
    content += "# TYPE queue_requests_total counter\n"
    if queue.queue_name:
        content += f'queue_requests_total{{stage="{queue.queue_name}"}} {queue.total_requests}\n'
    content += "# HELP queue_dropped_total Total number of dropped requests\n"
    content += "# TYPE queue_dropped_total counter\n"
    if queue.queue_name:
        content += f'queue_dropped_total{{stage="{queue.queue_name}"}} {queue.total_dropped}\n'
    content += "# HELP queue_size Total number of dropped requests\n"
    content += "# TYPE queue_size gauge\n"
    if queue.queue_name:
        content += f'queue_size{{stage="{queue.queue_name}"}} {queue.queue.qsize()}\n'
        
    return web.Response(body=content)


app = web.Application()
routes = [
    web.post("/submit", submit),
    web.post("/fetch", fetch)
]
if os.getenv("EXPORT_REQUESTS_TOTAL"):
    routes.append( web.get("/metrics", export_request_count))

app.add_routes(routes)

if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("QUEUE_PORT", 8000)), access_log=None)
