import asyncio
import json
import os
import time
import torch

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web


stop_event = asyncio.Event()


class ModelServer:

    def __init__(self) -> None:
        self.next_target_endpoint = os.getenv("NEXT_TARGET_ENDPOINT", "localhost")
        self.queue_endpoint = os.environ["QUEUE_ENDPOINT"]
        self.url_path = os.getenv("URL_PATH", "/submit")
        self.queue_url_path = "/fetch"
        self.session = None
        self.model = None
        self.batch_size = None
        self.batch_timeout = None
    
    async def initialize(self, request):
        t = time.perf_counter()
        torch.set_num_interop_threads(int(os.getenv("INTEROP_THREADS", 1)))
        torch.set_num_threads(int(request["threads"]))
        self.batch_size = int(request["batch_size"])
        self.batch_timeout = int(request["batch_timeout"])
        print(f"Model parallelism: inter={torch.get_num_interop_threads()}, intra={torch.get_num_threads()}, batch={self.batch_size}, timeout={self.batch_timeout}")
        self.model = self.load_model()
        print("model loading time", self.__class__.__name__, time.perf_counter() - t)
        self.session = ClientSession(
            base_url=f"http://{self.next_target_endpoint}",
            timeout=ClientTimeout(total=int(os.getenv("TIMEOUT", 30))),
            connector=TCPConnector(limit=0)
        )
        self.queue_session = ClientSession(
            base_url=f"http://{self.queue_endpoint}",
            # timeout=ClientTimeout(total=int(os.getenv("QUEUE_TIMEOUT", 30))),
        )
        t = time.perf_counter()
        self.warmup()
        asyncio.create_task(self.main_loop())
        print("warmup took:", self.__class__.__name__, time.perf_counter() - t)
    
    async def main_loop(self):
        global stop_event
        while not stop_event.is_set():
            await asyncio.sleep(0.005)
            request = await self.fetch_request()
            if request:
                await self.infer(request)
    
    async def fetch_request(self):
        async with self.queue_session.post(self.queue_url_path, json={"batch_size": self.batch_size, "batch_timeout": self.batch_timeout}) as response:
            response = await response.json()
            return response

    def load_model(self):
        raise NotImplementedError
    
    def warmup(self):
        pass
    
    def preprocess(self, data):
        return data
    
    def batch_preprocess(self, batch: list):
        return batch
    
    def inference(self, batch):
        return self.model(batch)
    
    def convert_batch_result(self, preds):
        return preds
    
    def get_next_target_data(self, pred):
        return pred

    async def infer(self, req) -> dict:
        batch = []
        arrival_time = time.time()
        c = 0
        for query in req:
            data = query["data"]
            query[f"arrival-{self.__class__.__name__}"] = arrival_time
            t = time.perf_counter()
            inp = self.preprocess(data)
            query[f"{self.__class__.__name__}-prep-time"] = round(time.perf_counter() - t, 3)
            batch.append(inp)
            c += 1

        t = time.perf_counter()
        batch = self.batch_preprocess(batch)
        batch_prep_time = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        with torch.no_grad():
            preds = self.inference(batch)
        infer_time = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        preds = self.convert_batch_result(preds)
        convert_batch_time = round(time.perf_counter() - t, 3)
        
        
        tasks = []
        for i in range(c):
            to_send = req[i]
            to_send["data"] = self.get_next_target_data(preds[i])
            to_send[f"{self.__class__.__name__}-batch-prep-time-{c}"] = batch_prep_time
            to_send[f"{self.__class__.__name__}-batch-inference-time-{c}"] = infer_time
            to_send[f"{self.__class__.__name__}-batch-convert-time-{c}"] = convert_batch_time
            to_send[f"{self.__class__.__name__}-node"] = os.uname()[1]
            to_send[f"leaving-{self.__class__.__name__}"] = time.time()

            tasks.append(asyncio.create_task(self.send(to_send)))
        return {"received": True, "node": os.uname()[1]}
    
    async def send(self, data):
        async with self.session.post(self.url_path, data=json.dumps(data)) as response:
            await response.text()
            return


def add_base_routes(app: web.Application, model_server: ModelServer):
    async def on_shutdown(app):
        global stop_event
        print("Received shutdown signal, stopping main loop...")
        stop_event.set()
    app.on_shutdown.append(on_shutdown)

    async def initialize(request):
        if model_server.model is None:
            req = await request.json()
            await model_server.initialize(req)
            return web.json_response({"success": True})
        return web.json_response({"success": False, "message": "Already initialized."})

    async def update_threads(request):
        req = await request.json()
        threads = req.get("threads")
        if threads:
            torch.set_num_threads(int(req["threads"]))
        
        t = time.perf_counter()
        if os.getenv("WARMUP_AFTER_THREAD_UPDATE", "false").lower() == "true" and threads and threads > 1:
            model_server.warmup()
        warmup_time = round(time.perf_counter() - t, 3)

        return web.json_response({"success": True, "warmup_took": warmup_time})
    
    async def update_batch(request):
        req = await request.json()
        batch_size = req.get("batch_size")
        batch_timeout = req.get("batch_timeout")
        if batch_size:
            model_server.batch_size = batch_size
        if batch_timeout:
            model_server.batch_timeout = batch_timeout

        return web.json_response({"success": True})

    
    app.add_routes(
        [
            web.post("/initialize", initialize),
            web.post("/update-threads", update_threads),
            web.post("/update-batch", update_batch),
        ]
    )

