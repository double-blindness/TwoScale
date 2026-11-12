import sys
from aiohttp import web
import json
import os
from datetime import datetime
from prometheus_client import start_http_server, Histogram
from kube_resources.pods import get_pod

DEFAULT_BUCKETS = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.2, 1.5, 1.7, 2, 3, 4, 5, 7, 10, float("inf")]
DEFAULT_BUCKETS.append(float(sys.argv[1]))  # Adding SLO to buckets
DEFAULT_BUCKETS.sort()

queue_stage0_histogram = Histogram(
    'queue_stage0_latency',
    'queue stage0 latency',
    buckets=DEFAULT_BUCKETS
)
detector_histogram = Histogram(
    'detector_latency',
    'Detector latency',
    buckets=DEFAULT_BUCKETS
)
queue_stage1_histogram = Histogram(
    'queue_stage1_latency',
    'queue stage1 latency',
    buckets=DEFAULT_BUCKETS
)
classifier_histogram = Histogram(
    'classifier_latency', 
    'Classifier latency',
    buckets=DEFAULT_BUCKETS
)
e2e_histogram = Histogram(
    'hvscale_requests_latency', 
    'End-to-end pipeline latency',
    buckets=DEFAULT_BUCKETS
)

class Exporter:
    def __init__(self) -> None:
        self.c = 0
        self.per_request_list = []
        self.pod_nodes = {}
        self.k8s_namespace = os.getenv("K8S_NAMESPACE", "mehran")

    def server_for_pod(self, pod_name):
        if pod_name in self.pod_nodes:
            return self.pod_nodes[pod_name]
        try:
            node = get_pod(pod_name, namespace=self.k8s_namespace)["node"]
        except Exception as exc:
            print(f"failed to resolve node for pod {pod_name}: {exc}")
            return None
        if node:
            self.pod_nodes[pod_name] = node
        return node
        
    async def receive(self, data: dict):
        self.c += 1
        data["queue-stage0"] = round(data["leaving-stage-0"] - data["arrival-stage-0"], 3)
        data["detector-e2e"] = round(data["leaving-Detector"] - data["arrival-Detector"], 3)
        data["queue-stage1"] = round(data["leaving-stage-1"] - data["arrival-stage-1"], 3)
        data["classifier-e2e"] = round(data["leaving-Classifier"] - data["arrival-Classifier"], 3)
        data["e2e"] = round(data["leaving-Classifier"] - data["arrival-stage-0"], 3)
        queue_stage0_histogram.observe(data["queue-stage0"])
        detector_histogram.observe(data["detector-e2e"])
        queue_stage1_histogram.observe(data["queue-stage1"])
        classifier_histogram.observe(data["classifier-e2e"])
        e2e_histogram.observe(data["e2e"])
        
        per_request = {"data": data["data"]}
        for k in ["queue-stage0", "detector-e2e", "queue-stage1", "classifier-e2e", "Detector-node", "Classifier-node", f"RID-stage-0" ,f"RID-stage-1", "Classifier-prep-time", "e2e"]:
            per_request[k] = data[k]
        per_request["Detector-server"] = self.server_for_pod(data["Detector-node"])
        per_request["Classifier-server"] = self.server_for_pod(data["Classifier-node"])
        for b in range(1, 7):
            if data.get(f"Classifier-batch-inference-time-{b}"):
                per_request[f"Classifier-batch-prep-time-{b}"] = data[f"Classifier-batch-prep-time-{b}"]
                per_request[f"Classifier-batch-inference-time-{b}"] = data[f"Classifier-batch-inference-time-{b}"]
                per_request[f"Classifier-batch-convert-time-{b}"] = data[f"Classifier-batch-convert-time-{b}"]
        per_request["timestamp"] = str(datetime.now())
        self.per_request_list.append(per_request)
        
        if data["queue-stage0"] > 0.77 or data["queue-stage1"] > 0.77:
            print("************************************", data["queue-stage0"], data["queue-stage1"])
        
        print(
            f'c={self.c}, {per_request}'
        )
        print()
 
        return {"saved": True}


exporter = Exporter()


async def receive(request):
    data = await request.json()
    return web.json_response(await exporter.receive(data))

filepath = os.path.dirname(os.path.dirname(__file__))
async def save_to_file(request):
    data = await request.json()
    with open(f"{filepath}/video_{data['adapter']}.json", "w") as f:
        json.dump(exporter.per_request_list, f, indent=2)
    return web.json_response({"saved": True})


app = web.Application()
app.add_routes(
    [
        web.post("/submit", receive),
        web.post("/save", save_to_file),
    ]
)

if __name__ == '__main__':
    start_http_server(8009)
    web.run_app(app, host="0.0.0.0", port=8008)
