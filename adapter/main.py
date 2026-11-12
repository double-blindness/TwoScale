import asyncio
import copy
import json
import logging
import math
# import numpy as np
import os
import random
import string
import time
# import tensorflow as tf
from aiohttp import ClientSession, web
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
# from tensorflow.keras import saving
from typing import Dict, List
import redis
import socket
from kube_resources.pods import create_pod, get_pod, update_pod, delete_pod, get_pods
# from kube_resources.services import create_service, get_service
# import dns.message

from optimizer import (
    horizontal_2d,
    sponge_optimizer,
    twoscale_vertical,
    cores_throughput,
    latency as get_latency,
    core_to_G_RAM,
)


def load_pipeline_data(d: dict):
    data = {}
    for k, v in json.loads(d).items():
        data[int(k)] = v
        
    return data


class Adapter:
    def __init__(self) -> None:
        self.pipeline_name = os.environ["PIPELINE_NAME"]
        self.horizontal_stabilization = int(os.getenv("HORIZONTAL_STABILIZATION", 10))
        self.history_seconds = int(os.getenv("HISTORY_SECONDS", 60))
        self.step_seconds = int(os.getenv("STEP_SECONDS", 10))
        self.horizontal_stabilization_counter = 0
        self.adapt_lock = asyncio.Lock()
        self.horizontal_task = None
        self.pending_pod_names = set()
        self.pod_creation_timeout = float(os.getenv("POD_CREATION_TIMEOUT", 20))
        self.latency_models = load_pipeline_data(os.environ["LATENCY_MODELS"])
        self.k8s_namespace = os.environ["K8S_NAMESPACE"]
        self.base_pod_names = load_pipeline_data(os.environ["BASE_POD_NAMES"])
        self.pod_labels = load_pipeline_data(os.environ["POD_LABELS"])
        self.pod_ports = load_pipeline_data(os.environ["POD_PORTS"])
        self.container_configs = load_pipeline_data(os.environ["CONTAINER_CONFIGS"])
        self.memory_per_stage_replica = {}
        self.max_batch_sizes = load_pipeline_data(os.environ["MAX_BATCH_SIZE"])
        self.max_cores = load_pipeline_data(os.environ["MAX_CPU_CORES"])
        self.latency_slo = int(os.environ["LATENCY_SLO"])
        self.horizontal_cmax = {}
        for i in range(len(self.container_configs)):
            mem = self.container_configs[i]["request_mem"]
            self.memory_per_stage_replica[i] = int(mem[:mem.index("G")]) # FIXME: Other units
            self.horizontal_cmax[i] = int(os.getenv("HORIZONTAL_CMAX", max(1, self.max_cores[i] // 2)))
        
        self.prometheus_session = None
        self.redis = redis.Redis(os.environ["REDIS_HOST"], os.environ["REDIS_PORT"])
        
        self.__thread_executor = ThreadPoolExecutor(max_workers=5 * len(self.latency_models))
        
        # self.lstm_model = saving.load_model(os.environ["LSTM_MODEL"])
        
        self.redis_key = f"hvscale-{self.pipeline_name}"
        
        # TwoScale: {stage: {"batch", "timeout", "instances": [{"name", "cores"}]}}
        # FA2, SPONGE+, and InferLine keep [cpu, replicas, batch, timeout].
        self.current_state = self.get_state_from_redis()
        
        self.ro_list = {} # Only for IL
        self.tp_list = {} # Only for IL
        
        self.dns_server = os.environ["DNS_SERVER_IP"]
        self.dns_port = int(os.environ["DNS_SERVER_PORT"])
        
        self.logger = logging.getLogger()
        logging.basicConfig(level = logging.INFO)
        
    
    async def initialize(self, data: dict):
        self.prometheus_session = ClientSession(base_url=os.environ["PROMETHEUS_ENDPOINT"])
        tasks = []
        current_state = {}
        hv = os.getenv("ADAPTER_TYPE", "hv") == "hv"
        
        for idx, state in data["pipeline_state"].items():
            stage = int(idx)
            c = int(state[0])
            r = int(state[1])
            b = int(state[2])
            bt = int(state[3])
            for _ in range(r):
                tasks.append((stage, c, asyncio.create_task(self.create_pod(stage, c, b, bt))))
            
            self.tp_list[stage] = int(1000 * b / get_latency(c, b, *self.latency_models[stage]))
            self.ro_list[stage] = data["inferline_base_arrival"] / (r * self.tp_list[stage])
            
            # create_service(
            #     self.base_pod_names[int(idx)],
            #     target_port=self.pod_ports[int(idx)],
            #     port=self.pod_ports[int(idx)],
            #     selector=self.pod_labels[int(idx)],
            #     cluster_ip="None",
            #     namespace=self.k8s_namespace
            # )
            
            if hv:
                current_state[stage] = {"batch": b, "timeout": bt, "instances": []}
            else:
                current_state[stage] = [c, r, b, bt]
        
        created = await asyncio.gather(*(task for _, _, task in tasks))
        if hv:
            for (stage, cores, _), pod in zip(tasks, created):
                current_state[stage]["instances"].append({"name": pod["name"], "cores": cores})
        
        self.logger.info(f"datetime={str(datetime.now())}, inferline_base_arrival={data['inferline_base_arrival']}, ro_list={json.dumps(self.ro_list)}")
        self.commit_state(current_state)
    
    # async def get_stage_replicas(self):
    #     stage_replicas = {}
    #     for i in range(len(self.current_state)):
    #         ips = await self.get_endpoints(i)
    #         stage_replicas[i] = ips
    #     return stage_replicas

    async def get_endpoints(self, stage) -> list:
        # The DNS has delay
        domain = f"{self.base_pod_names[stage]}.{self.k8s_namespace}.svc.cluster.local"
        return socket.gethostbyname_ex(domain)[2]  # This method had delay
        
        # packet = b'\x12\x34'   
        # packet += b'\x01\x00' 
        # packet += b'\x00\x01'
        # packet += b'\x00\x00'
        # packet += b'\x00\x00'
        # packet += b'\x00\x00'

        # # Encode QNAME (domain name in DNS format)
        # for part in domain.split('.'):
        #     packet += bytes([len(part)])
        #     packet += part.encode()
        # packet += b'\x00'  # End of QNAME

        # packet += b'\x00\x01'  # QTYPE = A
        # packet += b'\x00\x01'  # QCLASS = IN
        
        # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # sock.sendto(packet, (self.dns_server, self.dns_port))

        # # Receive response
        # data, _ = sock.recvfrom(512)
        # response = dns.message.from_wire(data)
        # return [rr.to_text() for rr in response.answer[0] if rr.rdtype == dns.rdatatype.A]
        
    def extract_stage_replicas(self, pods):
        sr = {}
        for i in range(len(self.current_state)):
            if sr.get(i) is None:
                sr[i] = []
            for r in pods:
                if r["name"].startswith(self.base_pod_names[i]) and r["name"] not in self.pending_pod_names:
                    sr[i].append(r)
        return sr
    
    def commit_state(self, new_state):
        self.current_state = new_state
        self.redis.set(self.redis_key, json.dumps(new_state))

    def get_state_from_redis(self):
        state = json.loads(self.redis.get(self.redis_key) or "{}")
        current_state = {}
        for k in state:
            current_state[int(k)] = state[k]
        return current_state

    async def adapt(self):
        async with self.adapt_lock:
            return await self._adapt()

    async def _adapt(self):
        current_state = self.current_state or self.get_state_from_redis()
        if not current_state:
            self.logger.info(f"datetime={str(datetime.now())}, No current state found, skipping adaptation")
            return
        starting_time = time.perf_counter()
        horizontal_pending = self.horizontal_task is not None
        pods = get_pods(self.k8s_namespace)["pods"]
        pods = list(filter(lambda x: x["phase"] == "Running" and x["terminating"] is False, pods))
        stage_replicas = self.extract_stage_replicas(pods)

        for i in range(len(current_state)):
            running = stage_replicas[i]
            running_names = {pod["name"] for pod in running}
            saved = current_state[i]["instances"]
            alive = [inst for inst in saved if inst["name"] in running_names]
            if saved and not running:
                self.logger.info(
                    f"datetime={str(datetime.now())}, Pods have crashed in stage {i}, expected={len(saved)}, current=0",
                )
                cores = [inst["cores"] for inst in saved]
                cores.append(min(cores))
                created = await asyncio.gather(*[
                    self.create_pod(i, core, current_state[i]["batch"], current_state[i]["timeout"])
                    for core in cores
                ])
                current_state[i]["instances"] = [
                    {"name": pod["name"], "cores": core} for pod, core in zip(created, cores)
                ]
                self.commit_state(current_state)
                pods = get_pods(self.k8s_namespace)["pods"]
                pods = list(filter(lambda x: x["phase"] == "Running" and x["terminating"] is False, pods))
                stage_replicas = self.extract_stage_replicas(pods)
                continue
            if len(alive) != len(saved):
                self.logger.info(
                    f"datetime={str(datetime.now())}, Pods have crashed in stage {i}, expected={len(saved)}, current={len(alive)}",
                )
            known = {inst["name"] for inst in alive}
            extras = [pod for pod in running if pod["name"] not in known]
            if extras:
                self.logger.info(
                    f"datetime={str(datetime.now())}, Mismatch in pod count in stage {i}, expected={len(alive)}, current={len(running)}",
                )
                floor = min((inst["cores"] for inst in alive), default=1)
                alive.extend({"name": pod["name"], "cores": floor} for pod in extras)
            current_state[i]["instances"] = alive
        
        current_rps = await self.get_current_rps()
        current_throughput = math.inf
        for i in range(len(current_state)):
            stage = current_state[i]
            total = cores_throughput(
                [inst["cores"] for inst in stage["instances"]],
                stage["batch"],
                self.latency_models[i],
            )
            if total < current_throughput:
                current_throughput = total
        over_provisioned = current_rps < current_throughput
        
        no_room_for_vertical = False
        for i in range(len(current_state)):
            instances = current_state[i]["instances"]
            if instances and all(inst["cores"] >= self.max_cores[i] for inst in instances):
                no_room_for_vertical = True
                break
        
        should_apply_horizontal = over_provisioned or no_room_for_vertical
        should_apply_horizontal = should_apply_horizontal and not horizontal_pending
        
        self.logger.info(
            f"datetime={str(datetime.now())}, {current_rps=}, {current_throughput=}, stabilization_counter={self.horizontal_stabilization_counter}, {should_apply_horizontal=}, {horizontal_pending=}", 
        )
        if should_apply_horizontal:
            new_horizontal_config = horizontal_2d(
                self.max_batch_sizes,
                self.horizontal_cmax, 
                self.latency_slo, 
                self.latency_models,
                self.memory_per_stage_replica,
                current_rps
            )
            scaling_in = False
            for i in range(len(new_horizontal_config)):
                if new_horizontal_config[i][0] < len(current_state[i]["instances"]):
                    scaling_in = True
            if scaling_in and over_provisioned and self.horizontal_stabilization_counter < self.horizontal_stabilization:
                prev_cost = 0
                new_cost = 0
                for i in range(len(new_horizontal_config)):
                    memory = self.memory_per_stage_replica[i]
                    prev_cost += sum(memory + inst["cores"] * core_to_G_RAM for inst in current_state[i]["instances"])
                    new_cost += new_horizontal_config[i][0] * (memory + new_horizontal_config[i][1] * core_to_G_RAM)
                if (prev_cost - new_cost) / prev_cost < 0.1:
                    self.logger.info(
                        f"datetime={str(datetime.now())}, scaling in ignored, {prev_cost=}, {new_cost=}", 
                    )
                    return
                self.horizontal_stabilization_counter += 1
                return
                
        self.horizontal_stabilization_counter = 0
        
        if should_apply_horizontal:
            # Horizontal optimizer
            creates = []
            for i in range(len(new_horizontal_config)):
                for _ in range(new_horizontal_config[i][0] - len(current_state[i]["instances"])):
                    creates.append((i, self.generate_pod_name(i)))
            if creates:
                self.pending_pod_names.update(name for _, name in creates)
                self.horizontal_task = asyncio.create_task(self._create_pods_and_reconfigure(new_horizontal_config, creates))
                self.logger.info(
                    f"datetime={str(datetime.now())}, horizontal transition started, target={json.dumps(new_horizontal_config)}, creating={json.dumps(creates)}"
                )
            else:
                new_state, extra_pods = await self._reconfigure_existing_pods(new_horizontal_config, current_state)
                await self._delete_pods(extra_pods)
                self.commit_state(new_state)
            self.logger.info(f"datetime={str(datetime.now())}, reconfiguration_took {time.perf_counter() - starting_time:.3f}\n")
            return
        
        # Vertical optimizer.
        update_tasks = []
        t_dp = time.perf_counter()
        specs = []
        for i in range(len(current_state)):
            stage = current_state[i]
            specs.append({
                "ids": [inst["name"] for inst in stage["instances"]],
                "cores": [inst["cores"] for inst in stage["instances"]],
                "batch": stage["batch"],
                "timeout": stage["timeout"],
                "bmax": self.max_batch_sizes[i],
                "cmax": self.max_cores[i],
                "model": self.latency_models[i],
            })
        plans = twoscale_vertical(specs, current_rps, self.latency_slo)
        self.logger.info(
            f"datetime={str(datetime.now())}, twoscale_vertical_took={time.perf_counter() - t_dp:.4f}, {current_rps=}"
        )
        before_vertical_apply_time = time.perf_counter()
        new_state = {}
        for i in range(len(plans)):
            batch = plans[i]["batch"]
            timeout = plans[i]["timeout"]
            cpu_by_name = plans[i]["cpu_by_id"]
            pods_by_name = {pod["name"]: pod for pod in stage_replicas[i]}
            stored = []
            for inst in current_state[i]["instances"]:
                pod = pods_by_name.get(inst["name"])
                if pod is None:
                    stored.append({"name": inst["name"], "cores": inst["cores"]})
                    continue
                new_cpu = max(cpu_by_name.get(inst["name"], inst["cores"]), inst["cores"])
                if new_cpu != inst["cores"]:
                    update_tasks.append(
                        asyncio.create_task(self.update_pod(
                            pod, i, inst["cores"], new_cpu, batch, timeout
                        ))
                    )
                stored.append({"name": inst["name"], "cores": new_cpu})
            new_state[i] = {"batch": batch, "timeout": timeout, "instances": stored}
        pod_status_check_tasks = await asyncio.gather(*update_tasks)
        self.logger.info(
            f"datetime={str(datetime.now())}, vertical_config={json.dumps(new_state)}, vertical_crud_took={time.perf_counter() - before_vertical_apply_time:.3f}"
        )
        
        update_batch_tasks = self.batch_update_coros(current_state, new_state, stage_replicas)
        
        await asyncio.gather(*update_batch_tasks)
        replacements = await asyncio.gather(*[asyncio.create_task(f) for f in pod_status_check_tasks])
        for replacement in replacements:
            if not replacement:
                continue
            old_name, new_name = replacement
            for stage in new_state.values():
                for inst in stage["instances"]:
                    if inst["name"] == old_name:
                        inst["name"] = new_name
        self.logger.info(f"datetime={str(datetime.now())}, Prev state={json.dumps(current_state)}, New state={json.dumps(new_state)}, stage_replicas={json.dumps({k: [sr['pod_ip'] for sr in v] for k, v in stage_replicas.items()})}")
        self.commit_state(new_state)
        self.logger.info(f"datetime={str(datetime.now())}, reconfiguration_took {time.perf_counter() - starting_time:.3f}\n")

    async def _create_pending_pod(self, stage_idx, new_horizontal_config, name):
        try:
            await asyncio.wait_for(
                self.create_pod(
                    stage_idx,
                    new_horizontal_config[stage_idx][1],
                    new_horizontal_config[stage_idx][2],
                    new_horizontal_config[stage_idx][3],
                    name=name
                ),
                timeout=self.pod_creation_timeout
            )
            async with self.adapt_lock:
                self.pending_pod_names.discard(name)
                new_state = copy.deepcopy(self.current_state)
                new_state[stage_idx]["instances"].append({
                    "name": name,
                    "cores": new_horizontal_config[stage_idx][1],
                })
                self.commit_state(new_state)
            return True
        except Exception:
            self.logger.exception(f"datetime={str(datetime.now())}, creating pod {name} for stage {stage_idx} failed, deleting it")
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self.__thread_executor,
                    lambda: delete_pod(name, namespace=self.k8s_namespace)
                )
            except Exception:
                self.logger.exception(f"datetime={str(datetime.now())}, deleting failed pod {name} failed")
            async with self.adapt_lock:
                self.pending_pod_names.discard(name)
            return False

    async def _create_pods_and_reconfigure(self, new_horizontal_config, creates):
        names = [name for _, name in creates]
        try:
            t = time.perf_counter()
            results = await asyncio.gather(*[
                self._create_pending_pod(i, new_horizontal_config, name) for i, name in creates
            ])
            planned = {}
            succeeded = {}
            for (i, _), ok in zip(creates, results):
                planned[i] = planned.get(i, 0) + 1
                succeeded[i] = succeeded.get(i, 0) + int(ok)
            self.logger.info(
                f"datetime={str(datetime.now())}, horizontal_creates_took={time.perf_counter() - t:.3f}, {planned=}, {succeeded=}"
            )
            async with self.adapt_lock:
                config = {i: list(c) for i, c in new_horizontal_config.items()}
                for i in planned:
                    if succeeded[i] == 0:
                        stage = self.current_state[i]
                        # None keeps each replica's current core count.
                        config[i] = [len(stage["instances"]), None, stage["batch"], stage["timeout"]]
                    else:
                        config[i][0] -= planned[i] - succeeded[i]
                new_state, extra_pods = await self._reconfigure_existing_pods(config, self.current_state)
                await self._delete_pods(extra_pods)
                self.commit_state(new_state)
        except Exception:
            self.logger.exception(f"datetime={str(datetime.now())}, horizontal transition failed, pods={json.dumps(names)}")
        finally:
            self.pending_pod_names.difference_update(names)
            self.horizontal_task = None

    async def _reconfigure_existing_pods(self, new_horizontal_config, current_state):
        pods = get_pods(self.k8s_namespace)["pods"]
        pods = list(filter(lambda x: x["phase"] == "Running" and x["terminating"] is False, pods))
        stage_replicas = self.extract_stage_replicas(pods)
        extra_pods = {i: [] for i in range(len(current_state))}
        futures = []
        tasks = []
        new_state = {}
        before_horizontal_apply_time = time.perf_counter()
        for i in range(len(new_horizontal_config)):
            target_cpu = new_horizontal_config[i][1]
            desired = new_horizontal_config[i][0]
            batch = new_horizontal_config[i][2]
            timeout = new_horizontal_config[i][3]
            stage = current_state[i]
            by_name = {inst["name"]: inst["cores"] for inst in stage["instances"]}
            kept = list(stage_replicas[i])
            if desired < len(stage["instances"]):
                extra_pods_count = len(stage["instances"]) - desired
                extra_pods[i] = kept[:extra_pods_count]
                kept = kept[extra_pods_count:]
                stage_replicas[i] = kept
                self.logger.info(f"datetime={str(datetime.now())}, pods_to_delete_stage{i}={json.dumps(extra_pods[i])}")
            kept_names = {pod["name"] for pod in kept}
            ordered = [inst["name"] for inst in stage["instances"] if inst["name"] in kept_names]
            seen = set(ordered)
            for pod in kept:
                if pod["name"] not in seen:
                    ordered.append(pod["name"])
                    seen.add(pod["name"])
            pods_by_name = {pod["name"]: pod for pod in kept}
            if target_cpu is None:
                floor = min(by_name.values()) if by_name else 1
                new_state[i] = {
                    "batch": batch,
                    "timeout": timeout,
                    "instances": [
                        {"name": name, "cores": by_name.get(name, floor)}
                        for name in ordered
                    ],
                }
                continue
            new_state[i] = {
                "batch": batch,
                "timeout": timeout,
                "instances": [{"name": name, "cores": target_cpu} for name in ordered],
            }
            new_cores = {inst["name"]: inst["cores"] for inst in new_state[i]["instances"]}
            if stage["batch"] == batch and stage["timeout"] == timeout and by_name == new_cores:
                continue

            update = futures if desired >= len(stage["instances"]) else tasks
            for name in ordered:
                update.append(self.update_pod(
                    pods_by_name[name], i, by_name.get(name, target_cpu), target_cpu, batch, timeout, check_status=False
                ))
        
        await asyncio.gather(*tasks)
        
        update_batch_tasks = self.batch_update_coros(current_state, new_state, stage_replicas)
        
        self.logger.info(f"datetime={str(datetime.now())}, Prev state={json.dumps(current_state)}, New state={json.dumps(new_state)}, stage_replicas={json.dumps({k: [sr['pod_ip'] for sr in v] for k, v in stage_replicas.items()})}")
        
        await asyncio.gather(*update_batch_tasks)

        if len(futures) > 0:
            self.logger.info(f"datetime={str(datetime.now())}, applying futures (N={len(futures)}) one by one.")
        for f in futures:
            await asyncio.create_task(f)
        self.logger.info(f"datetime={str(datetime.now())}, reconfigure_existing_pods_took={time.perf_counter() - before_horizontal_apply_time:.3f}")
        return new_state, extra_pods

    async def _delete_pods(self, pods_by_stage):
        loop = asyncio.get_event_loop()
        for pods in pods_by_stage.values():
            for r in pods:
                await loop.run_in_executor(
                    self.__thread_executor,
                    lambda: delete_pod(r["name"], namespace=self.k8s_namespace)
                )
                self.logger.info(f"datetime={str(datetime.now())}, deleted_replica={r['name']}")
                
    
    async def adapt_ho(self):
        starting_time = time.perf_counter()
        current_state = self.current_state or json.loads(self.redis.get(self.redis_key))
        pods = get_pods(self.k8s_namespace)["pods"]
        pods = list(filter(lambda x: x["phase"] == "Running" and x["terminating"] is False, pods))
        stage_replicas = self.extract_stage_replicas(pods)
        extra_pods = {}
        rt = time.perf_counter() - starting_time
        current_rps = await self.get_current_rps()
        horizontal_config = horizontal_2d(
            self.max_batch_sizes,
            [1] * len(current_state),
            self.latency_slo,
            self.latency_models, 
            self.memory_per_stage_replica,
            current_rps
        )
        is_scaling_in = False
        for i in range(len(current_state)):
            if current_state[i][1] > horizontal_config[i][0]:
                is_scaling_in = True
                break
        if is_scaling_in:
            if self.horizontal_stabilization_counter < self.horizontal_stabilization:
                self.horizontal_stabilization_counter += 1
                return
        
        new_state = {}
        self.horizontal_stabilization_counter = 0
        tasks = []
        loop = asyncio.get_event_loop()
        for i in range(len(horizontal_config)):
            extra_pods[i] = []
            new_state[i] = [1, horizontal_config[i][0], horizontal_config[i][2], horizontal_config[i][3]]
            self.logger.info(f"datetime={str(datetime.now())}, Prev state {i}={json.dumps(current_state[i])} | New state {i}={json.dumps(new_state[i])}")
            if new_state[i] == current_state[i]:
                continue
            
            if horizontal_config[i][0] > current_state[i][1]:
                for _ in range(horizontal_config[i][0] - current_state[i][1]):
                    tasks.append(
                        asyncio.create_task(self.create_pod(i, 1, horizontal_config[i][2], horizontal_config[i][3]))
                    )
            else:
                extra_pods_count = current_state[i][1] - horizontal_config[i][0]
                extra_pods[i] = stage_replicas[i][:extra_pods_count]
                stage_replicas[i] = stage_replicas[i][extra_pods_count:]
                self.logger.info(f"datetime={str(datetime.now())}, pods_to_delete_stage{i}={json.dumps(extra_pods[i])}")
                self.logger.info(f"datetime={str(datetime.now())}, stage_replicas={json.dumps(stage_replicas)}, current_state={json.dumps(self.current_state)}")
                
        await asyncio.gather(*tasks)
        
        update_batch_tasks = self.batch_update_coros(current_state, new_state, stage_replicas)
        
        self.logger.info(f"datetime={str(datetime.now())}, Prev state={json.dumps(current_state)} | New state={json.dumps(new_state)}, redis_get_took={rt:.3f}")
        
        rt = time.perf_counter()
        self.commit_state(new_state)
        rt = time.perf_counter() - rt
        await asyncio.gather(*update_batch_tasks)
        for i in range(len(horizontal_config)):
            for r in extra_pods[i]:
                delete_response = await loop.run_in_executor(
                    self.__thread_executor,
                    lambda: delete_pod(r["name"], namespace=self.k8s_namespace)
                )
                self.logger.info(f"datetime={str(datetime.now())}, deleted_replica={r['name']}")
        self.logger.info(f"datetime={str(datetime.now())}, reconfiguration_took {time.perf_counter() - starting_time:.3f}, redis_took={rt:.3f}\n")
    
    
    async def adapt_vo(self):
        starting_time = time.perf_counter()
        current_state = self.current_state or json.loads(self.redis.get(self.redis_key))
        pods = get_pods(self.k8s_namespace)["pods"]
        pods = list(filter(lambda x: x["phase"] == "Running" and x["terminating"] is False, pods))
        stage_replicas = self.extract_stage_replicas(pods)
        current_rps = await self.get_current_rps()
        new_vertical_config = sponge_optimizer(self.max_batch_sizes, self.max_cores, self.latency_slo, self.latency_models, current_state, current_rps)
        
        is_scaling_down = all(current_state[i][0] > new_vertical_config[i][0] for i in range(len(current_state)))
        self.logger.info(f"datetime={str(datetime.now())}, current_rps={current_rps}, is_scaling_down={is_scaling_down} reconfiguration_took {time.perf_counter() - starting_time:.3f}")
        if is_scaling_down:
            if self.horizontal_stabilization_counter < self.horizontal_stabilization:
                self.horizontal_stabilization_counter += 1
                return
        
        self.horizontal_stabilization_counter = 0
        new_state = {}
        update_tasks = []
        
        for i in range(len(new_vertical_config)):
            new_state[i] = [new_vertical_config[i][0], current_state[i][1], new_vertical_config[i][1], new_vertical_config[i][2]]
            if new_vertical_config[i][0] != current_state[i][0]:
                for r in stage_replicas[i]:
                    update_tasks.append(
                        asyncio.create_task(
                            self.update_pod(
                                r, i, current_state[i][0], new_vertical_config[i][0], new_vertical_config[i][1], new_vertical_config[i][2]
                            )
                        )
                    )
        
        pod_status_check_tasks = await asyncio.gather(*update_tasks)
        
        update_batch_tasks = self.batch_update_coros(current_state, new_state, stage_replicas)
                
        await asyncio.gather(*update_batch_tasks)
        await asyncio.gather(*[asyncio.create_task(f) for f in pod_status_check_tasks])
        self.commit_state(new_state)
        self.logger.info(f"datetime={str(datetime.now())}, Prev state={json.dumps(current_state)} | New state={json.dumps(new_state)}, reconfiguration_took {time.perf_counter() - starting_time:.3f}\n")
    
    async def adapt_il(self):
        starting_time = time.perf_counter()
        current_state = self.current_state or json.loads(self.redis.get(self.redis_key))
        pods = get_pods(self.k8s_namespace)["pods"]
        pods = list(filter(lambda x: x["phase"] == "Running" and x["terminating"] is False, pods))
        stage_replicas = self.extract_stage_replicas(pods)
        extra_pods = {}
        current_rps = await self.get_rps_history(30)
        current_rps = max(current_rps)

        replicas = []
        for i in range(len(current_state)):
            extra_pods[i] = []
            stage_replicas_count = max(1, math.ceil(current_rps / (self.tp_list[i] * self.ro_list[i])))
            if stage_replicas_count < current_state[i][1]:
                stage_replicas_count = max(1, math.ceil(current_rps / (self.tp_list[i] * min(self.ro_list.values()))))
                if stage_replicas_count > current_state[i][1]:
                    stage_replicas_count = current_state[i][1]
            replicas.append(stage_replicas_count)
            
        is_scaling_in = False
        for i in range(len(current_state)):
            if current_state[i][1] > replicas[i]:
                is_scaling_in = True
                break
        self.logger.info(
            f"datetime={str(datetime.now())}, {current_rps=}, stabilization_counter={self.horizontal_stabilization_counter}, {is_scaling_in=}, Replicas={json.dumps(replicas)}"
        )
        if is_scaling_in:
            if self.horizontal_stabilization_counter < self.horizontal_stabilization:
                self.horizontal_stabilization_counter += 1
                return
        
        new_state = {}
        self.horizontal_stabilization_counter = 0
        tasks = []
        loop = asyncio.get_event_loop()
        for i in range(len(replicas)):
            new_state[i] = [current_state[i][0], replicas[i], current_state[i][2], current_state[i][3]]
            self.logger.info(f"datetime={str(datetime.now())}, Prev state {i}={json.dumps(current_state[i])} | New state {i}={json.dumps(new_state[i])}")
            if new_state[i] == current_state[i]:
                continue
            
            if replicas[i] > current_state[i][1]:
                for _ in range(replicas[i] - current_state[i][1]):
                    tasks.append(
                        asyncio.create_task(self.create_pod(i, current_state[i][0], current_state[i][2], current_state[i][3]))
                    )
            else:
                extra_pods_count = current_state[i][1] - replicas[i]
                extra_pods[i] = stage_replicas[i][:extra_pods_count]
                stage_replicas[i] = stage_replicas[i][extra_pods_count:]
                self.logger.info(f"datetime={str(datetime.now())}, pods_to_delete_stage{i}={json.dumps(extra_pods[i])}")
                self.logger.info(f"datetime={str(datetime.now())}, stage_replicas={json.dumps(stage_replicas)}, current_state={json.dumps(current_state)}")
                
        await asyncio.gather(*tasks)
        for i in range(len(replicas)):
            for r in extra_pods[i]:
                delete_response = await loop.run_in_executor(
                    self.__thread_executor,
                    lambda: delete_pod(r["name"], namespace=self.k8s_namespace)
                )
                self.logger.info(f"datetime={str(datetime.now())}, deleted_replica={r['name']}")
        
        self.logger.info(f"datetime={str(datetime.now())}, Inferline, {current_rps=}, Prev state={json.dumps(current_state)} | New state={json.dumps(new_state)}")
        
        self.commit_state(new_state)
        
        self.logger.info(f"datetime={str(datetime.now())}, reconfiguration_took {time.perf_counter() - starting_time:.3f}\n")
        
    
    async def get_current_rps(self):
        t = time.perf_counter()
        
        current_rps = await self.get_rps_history(10)
        current_rps = max(current_rps)
        self.logger.info(f"datetime={str(datetime.now())}, get_current_rps_took: {time.perf_counter() - t:.3f}")
        return math.ceil(current_rps)
        # async with self.prometheus_session.post(
        #     f"/api/v1/query",
        #     params={
        #         "query": 'rate(queue_requests_total{stage="stage-0"}[2s])',
        #     }
        # ) as response:
        #     response = await response.json()
        #     self.logger.info(f"datetime={str(datetime.now())}, get_current_rps_took: {time.perf_counter() - t:.3f}")
        #     rps = float(response["data"]["result"][0]["value"][1])
        #     return math.ceil(rps)
    
    # async def is_wl_increasing_future(self, target: int):
    #     t = time.perf_counter()
    #     history_rps = await self.get_rps_history(self.history_seconds)
    #     get_history_rps_took = round(time.perf_counter() - t, 3)
    #     t = time.perf_counter()
    #     inp = []
    #     for i in range(0, len(history_rps), self.step_seconds):
    #         inp.append(max(history_rps[i:i+self.step_seconds]))
    #     history_rps = inp
    #     if len(history_rps) < (self.history_seconds // self.step_seconds):
    #         history_rps = ((self.history_seconds // self.step_seconds) - len(history_rps)) * [history_rps[0]] + history_rps
        
    #     model_input = []
    #     EPSILON = 1e-6
    #     for i in range(1, len(history_rps)):
    #         model_input.append(np.log(history_rps[i] + EPSILON) - np.log(history_rps[i-1] + EPSILON))

    #     pred = self.predict(model_input)
    #     predicted_rps = history_rps[-1] * math.e ** (pred)
    #     self.logger.info(f"datetime={str(datetime.now())}, {get_history_rps_took=}, LSTM_prediction_took: {time.perf_counter() - t:.3f}")
    #     return predicted_rps > target
    
    async def get_rps_history(self, seconds):
        now = datetime.now().timestamp()
        if self.prometheus_session is None:
            self.prometheus_session = ClientSession(base_url=os.environ["PROMETHEUS_ENDPOINT"])
        async with self.prometheus_session.post(
            f"/api/v1/query_range",
            params={
                "query": 'queue_requests_total{stage="stage-0"}',
                "start": now - seconds,
                "end": now,
                "step": 1
            }
        ) as response:
            response = await response.json()
            history_rps = response["data"]["result"][0].get("values")
            
        history_rps = list(map(lambda x: int(x[1]), history_rps))
        history_rps = [history_rps[i] - history_rps[i-1] for i in range(1, len(history_rps))]
        return history_rps
        
        
    def generate_pod_name(self, stage_idx):
        return f"{self.base_pod_names[stage_idx]}-{''.join(random.choices(string.ascii_lowercase + string.digits, k=5))}"

    async def create_pod(self, stage_idx, cpu: int, b: int, bt: int, name=None):
        new_pod_name = name or self.generate_pod_name(stage_idx)
        loop = asyncio.get_event_loop()
        t = time.perf_counter()
        await loop.run_in_executor(
            self.__thread_executor,
            lambda: create_pod(
                new_pod_name,
                [{
                    **self.container_configs[stage_idx],
                    "request_cpu": cpu,
                    "limit_cpu": cpu
                }],
                namespace=self.k8s_namespace,
                labels=self.pod_labels[stage_idx]
            )
        )
        pod_creation_time = time.perf_counter() - t
        t = time.perf_counter()
        while True:
            await asyncio.sleep(0.05)
            pod = await loop.run_in_executor(self.__thread_executor, lambda: get_pod(new_pod_name, namespace=self.k8s_namespace))
            if pod["pod_ip"] and pod["pod_ip"].lower() != "none":
                break
        pod_wait_for_ip_time = time.perf_counter() - t
        t = time.perf_counter()
        while True:
            await asyncio.sleep(0.05)
            try:
                await self.initialize_pod(f"{pod['pod_ip']}:{self.pod_ports[stage_idx]}", cpu, b, bt)
                break
            except Exception:
                pass
        pod_initialization_time = time.perf_counter() - t
        self.logger.info(f"datetime={str(datetime.now())}, New pod for stage {stage_idx} created. pod name: {pod['name']}, cpu: {cpu}, creation={pod_creation_time:.3f}, ip={pod_wait_for_ip_time:.3f}, init={pod_initialization_time:.3f}")
        return pod
        
    
    async def initialize_pod(self, pod_endpoint, cpu, b, bt):
        async with ClientSession() as session:
            async with session.post(
                f"http://{pod_endpoint}/initialize", json={"threads": cpu, "batch_size": b, "batch_timeout": bt}
            ) as response:
                return await response.json()

    async def update_pod(self, pod_info: dict, stage_idx, current_cpu, new_cpu, new_batch, new_batch_timeout, check_status=True):
        loop = asyncio.get_event_loop()
        if new_cpu == 1:
            await self.update_threading(f"{pod_info['pod_ip']}:{self.pod_ports[stage_idx]}", new_cpu)
        await loop.run_in_executor(
            self.__thread_executor,
            lambda: update_pod(
                pod_info["name"],
                [{
                    **self.container_configs[stage_idx],
                    "request_cpu": new_cpu,
                    "limit_cpu": new_cpu
                }],
                namespace=self.k8s_namespace
            )
        )
        if new_cpu > 1:
            await self.update_threading(f"{pod_info['pod_ip']}:{self.pod_ports[stage_idx]}", new_cpu)
        
        if check_status:
            return self.react_to_update_status(pod_info, stage_idx, current_cpu, new_cpu, new_batch, new_batch_timeout)
    
    async def react_to_update_status(self, pod_info, stage_idx, current_cpu, new_cpu, new_batch, new_batch_timeout):
        loop = asyncio.get_event_loop()
        status, reason = await self.check_update_status(pod_info["name"])
        
        if status == "PodResizePending":
            self.logger.info(f"datetime={str(datetime.now())}, pod {pod_info['name']} update pending reason={reason}")
            await self.update_threading(f"{pod_info['pod_ip']}:{self.pod_ports[stage_idx]}", current_cpu)
            new_pod = await self.create_pod(stage_idx, new_cpu, new_batch, new_batch_timeout)
            await loop.run_in_executor(
                self.__thread_executor,
                lambda: delete_pod(pod_info['name'], namespace=self.k8s_namespace)
            )
            return pod_info["name"], new_pod["name"]
        
    def resize_condition(self, pod):
        by_type = {
            condition["type"]: condition
            for condition in (pod.get("conditions") or [])
            if condition["type"] in ("PodResizePending", "PodResizeInProgress")
        }
        return by_type.get("PodResizePending") or by_type.get("PodResizeInProgress")

    
    async def check_update_status(self, pod_name, rep=0):
        response = get_pod(pod_name, namespace=self.k8s_namespace)
        condition = self.resize_condition(response)
        status = condition["type"] if condition else None
        if status not in ("PodResizeInProgress", "PodResizePending"):
            await asyncio.sleep(0.04)
            if rep < 5:
                return await self.check_update_status(pod_name, rep+1)
        reason = condition.get("reason") if condition else None
        self.logger.info(
            f"datetime={str(datetime.now())}, pod {pod_name} update status {status or 'completed'} reason={reason} rep={rep}"
        )
        return status, reason

    async def update_threading(self, pod_endpoint, threads: int):
        async with ClientSession() as session:
            async with session.post(
                f"http://{pod_endpoint}/update-threads", json={"threads": threads}
            ) as response:
                return await response.json()
            
     
    def batch_update_coros(self, current_state, new_state, stage_replicas):
        coros = []
        for i in range(len(current_state)):
            if isinstance(current_state[i], dict):
                prev_batch_size, prev_batch_timeout = current_state[i]["batch"], current_state[i]["timeout"]
                new_batch_size, new_batch_timeout = new_state[i]["batch"], new_state[i]["timeout"]
            else:
                prev_batch_size, prev_batch_timeout = current_state[i][2], current_state[i][3]
                new_batch_size, new_batch_timeout = new_state[i][2], new_state[i][3]
            if prev_batch_size != new_batch_size or prev_batch_timeout != new_batch_timeout:
                for r in stage_replicas[i]:
                    coros.append(
                        self.update_batch(f"{r['pod_ip']}:{self.pod_ports[i]}", new_batch_size, new_batch_timeout)
                    )
        return coros

    async def update_batch(self, pod_endpoint, new_batch: int, new_batch_timeout: int):
        async with ClientSession() as session:
            async with session.post(
                f"http://{pod_endpoint}/update-batch", json={"batch_size": new_batch, "batch_timeout": new_batch_timeout}
            ) as response:
                return await response.json()
                
        
    # def predict(self, history_rps_log_changes):
    #     history = tf.convert_to_tensor([history_rps_log_changes], dtype=tf.float32)
    #     preds = self.lstm_model.predict(history, verbose=0)
    #     return preds[0][0]


adapter = Adapter()


async def initialize(request):
    data = await request.json()
    if not adapter.current_state:
        await adapter.initialize(data)
        return web.json_response({"success": True})
    return web.json_response({"success": False, "message": "Already initialized."})


async def export_cost(request):
    content = "# HELP hvscale_cost Number of cpu cores * replicas.\n"
    content += "# TYPE hvscale_cost gauge\n"
    for stage_idx in range(len(adapter.current_state)):
        stage = adapter.current_state[stage_idx]
        memory = adapter.memory_per_stage_replica[stage_idx]
        if isinstance(stage, dict):
            cost = sum(memory + inst["cores"] * core_to_G_RAM for inst in stage["instances"])
        else:
            cost = stage[1] * (memory + stage[0] * core_to_G_RAM)
        content += f'hvscale_cost{{stage="{stage_idx}"}} {cost}\n'
    return web.Response(body=content)


async def decide(request):
    return web.json_response(await adapter.adapt())

async def decide_ho(request):
    return web.json_response(await adapter.adapt_ho())

async def decide_vo(request):
    return web.json_response(await adapter.adapt_vo())

async def decide_il(request):
    return web.json_response(await adapter.adapt_il())


app = web.Application()
app.add_routes(
    [
        web.post("/initialize", initialize),
        web.get("/metrics", export_cost),
        web.post("/decide", decide),
        web.post("/decide-ho", decide_ho),
        web.post("/decide-vo", decide_vo),
        web.post("/decide-il", decide_il),
    ]
)
if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=8000, access_log=None)
