import requests
import time
import logging
import os
from threading import Thread
import json
import redis


logger = logging.getLogger()
adapter_type = os.getenv("ADAPTER_TYPE")
assert adapter_type is not None, "Pass ADAPTER_TYPE as env var. It gets one of hv, ho, and vo values"

def signal():
    path = "decide"
    if adapter_type in ["ho", "vo", "il"]:
        path = path + f"-{adapter_type}"
    
    try:
        requests.post(f"http://localhost:8000/{path}", timeout=10)
    except (TimeoutError, requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
        logger.error(f"The reconfiguration timed out")
        
    
    
if __name__ == "__main__":
    state = json.loads(redis.Redis(os.environ["REDIS_HOST"], os.environ["REDIS_PORT"]).get(f"hvscale-{os.environ['PIPELINE_NAME']}") or "{}")
    if len(state) == 0:
        time.sleep(int(60 * float(os.environ["FIRST_DECIDE_DELAY_MINUTES"])))
    else:
        time.sleep(5)
    decision_interval = float(os.environ["DECISION_INTERVAL"])
    logger.info(f"Adapter type is {adapter_type}")
    while True:
        thread = Thread(target=signal)
        thread.start()
        time.sleep(decision_interval)
        thread.join()
