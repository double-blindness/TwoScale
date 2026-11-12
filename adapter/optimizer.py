import math
import os
from scipy.stats import gamma

core_to_G_RAM = 18

std_addition = float(os.getenv("STD_ADDITION", "0"))

def latency(core, batch, alpha, beta, gamma, zeta):
    zeta += (alpha + beta + gamma + zeta) / 10
    mlt = float(os.getenv("LATENCY_MODEL_MULTIPLIER", "1"))
    mltb = float(os.getenv("LATENCY_MODEL_BATCH_MULTIPLIER", "1"))
    return math.ceil(mlt * (alpha * batch / core + mltb * (beta * batch) + gamma / core + math.ceil(zeta)))


def get_throughput(state: dict, models: dict):
    current_tp = math.inf
    for s in range(len(state)):
        b = state[s][2]
        replicas = state[s][1]
        cores = state[s][0]
        current_latency = latency(cores, b, *models[s])
        tp = replicas * int(1000 * b / current_latency)
        if tp < current_tp:
            current_tp = tp
    return current_tp

def get_queuing(b, workload):
    if b == 1:
        return 0
    queueing_dist = os.getenv("QUEUEING_DIST", "gamma")
    if queueing_dist == "gamma":
        return math.ceil(1000 * gamma.ppf(0.99, a=b-1, scale=1/workload))
    elif queueing_dist == "exponential":
        return math.ceil((b-1) * (-math.log(0.01, math.e) * 1000) / workload)
    elif queueing_dist == "constant":
        return math.ceil((b-1) / workload)
    else:
        raise Exception("Invalid distribution")


def horizontal_2d(b_max: list, c_max: list, slo, models, stage_memory_requests_G, workload):
    if workload == 0:
        return {s: [1, 1, 1, 0] for s in range(len(models))}
    dp = []
    best = []
    for i in range(len(models)):
        dp.append([False] * (slo + 1))
        best.append([(math.inf, math.inf, math.inf, math.inf)] * (slo + 1))
    best[0][0] = (0, 0, 0, 0)
    dp[0][0] = True
    for s in range(len(models)):
        for i in range(slo, -1, -1):
            if (s == 0 and dp[s][i]) or s != 0 and dp[s - 1][i]:
                for c in range(1, c_max[s] + 1):
                    for b in range(1, b_max[s] + 1):
                        curr_latency = latency(c, b, models[s][0], models[s][1], models[s][2], models[s][3])
                        throughput = int(1000 * b / curr_latency)
                        queuing_delay = get_queuing(b, workload)
                        curr_latency += queuing_delay
                        
                        if i + curr_latency > slo:
                            continue
                        
                        needed_instances = math.ceil(round(workload +  std_addition * (workload ** 0.5)) / throughput)
                        cost =  needed_instances * (stage_memory_requests_G[s] + c * core_to_G_RAM)
                        # The first model
                        if s == 0:
                            if dp[s][i + curr_latency] is False:
                                dp[s][i + curr_latency] = True
                                best[s][i + curr_latency] = (cost, needed_instances, c, b, queuing_delay)
                            elif cost < best[s][i + curr_latency][0]:
                                best[s][i + curr_latency] = (cost, needed_instances, c, b, queuing_delay)
                        elif dp[s - 1][i] and i > 0:
                            if dp[s][i + curr_latency] is False:
                                dp[s][i + curr_latency] = True
                                best[s][i + curr_latency] = (best[s - 1][i][0] + cost, needed_instances, c, b, queuing_delay)
                            elif best[s - 1][i][0] + cost < best[s][i + curr_latency][0]:
                                best[s][i + curr_latency] = (best[s - 1][i][0] + cost, needed_instances, c, b, queuing_delay)
    least_n = math.inf
    ind = -1
    for i in range(slo):
        if dp[len(models) - 1][i]:
            if ind == -1:
                ind = i
                least_n = best[len(models) - 1][i][0]
            elif best[len(models) - 1][i][0] < least_n:
                ind = i
                least_n = best[len(models) - 1][i][0]
    res = {}
    if ind == -1:
        raise Exception("No solution found in horizontal_2d.")
        
    stage = len(models) - 1
    while stage >= 0:
        res[stage] = [best[stage][ind][1], best[stage][ind][2], best[stage][ind][3], best[stage][ind][4]]
        ind -= (
            latency(
                best[stage][ind][2],
                best[stage][ind][3],
                *models[stage],
            ) + get_queuing(best[stage][ind][3], workload)
        )
        stage -= 1
    return res


def sponge_optimizer(b_max: list, c_max: list, slo, models, current_instance, workload):
    if workload == 0:
        return {s: [1, 1, 0] for s in range(len(models))}
    dp = []
    best = []
    for i in range(len(models)):
        dp.append([False] * (slo + 1))
        best.append([(math.inf, math.inf, math.inf)] * (slo + 1))
    best[0][0] = (0, 0, 0)
    dp[0][0] = True
    for s in range(len(models)):
        for i in range(slo, -1, -1):
            if (s == 0 and dp[s][i]) or s != 0 and dp[s - 1][i]:
                for c in range(1, c_max[s] + 1):
                    for b in range(1, b_max[s] + 1):
                        curr_latency = latency(c, b, models[s][0], models[s][1], models[s][2], models[s][3])
                        throughput = int(1000 * b / curr_latency)
                        queuing_delay = get_queuing(b, workload)
                        curr_latency += queuing_delay
                        
                        if i + curr_latency > slo:
                            continue
                        
                        if throughput * current_instance[s][1] < round(workload + std_addition * (workload ** 0.5)):
                            continue
                        # The first model
                        if s == 0:
                            if dp[s][i + curr_latency] is False:
                                dp[s][i + curr_latency] = True
                                best[s][i + curr_latency] = (c, c, b, queuing_delay)
                        # Not the first model
                        elif dp[s - 1][i] and i > 0:
                            # if TOTAL_CORE < best[s - 1][i][0] + c:
                            #     continue
                            if dp[s][i + curr_latency] is False:
                                dp[s][i + curr_latency] = True
                                best[s][i + curr_latency] = (best[s - 1][i][0] + c, c, b, queuing_delay)
                            elif best[s - 1][i][0] + c < best[s][i + curr_latency][0]:
                                best[s][i + curr_latency] = (best[s - 1][i][0] + c, c, b, queuing_delay)
    least_c = math.inf
    ind = -1
    for i in range(slo):
        if dp[len(models) - 1][i] and best[len(models) - 1][i][0] < least_c:
            ind = i
            least_c = best[len(models) - 1][i][0]
    res = {}
    if ind == -1:
        required = round(workload + std_addition * (workload ** 0.5))
        for x in range(len(models)):
            c, replicas, b, bt = current_instance[x][:4]
            if int(1000 * b / latency(c, b, *models[x])) * replicas < required:
                c = c_max[x]
            res[x] = [c, b, bt]
    else:
        counter = len(models) - 1
        while counter >= 0:
            res[counter] = [best[counter][ind][1], best[counter][ind][2], best[counter][ind][3]]
            ind -= (
                latency(best[counter][ind][1], best[counter][ind][2], *models[counter]) +
                get_queuing(best[counter][ind][2], workload)
            )
            counter -= 1
    return res


vertical_2d = sponge_optimizer


def cores_throughput(cores, batch, model, latency_fn=latency):
    total = 0
    for cpu in cores:
        service = latency_fn(cpu, batch, *model)
        total += int(1000 * batch / service)
    return total


def spread_cores(cores, batch, model, cmax, demand, latency_fn=latency):
    """Add one core at a time to the replica with the fewest cores.

    Ties go to the earlier replica. A replica already at cmax is skipped.
    Stops when throughput covers demand, or every replica is at cmax.
    """
    cores = list(cores)
    while cores and cores_throughput(cores, batch, model, latency_fn) < demand:
        pick = None
        for i, cpu in enumerate(cores):
            if cpu >= cmax:
                continue
            if pick is None or (cpu, i) < (cores[pick], pick):
                pick = i
        if pick is None:
            break
        cores[pick] += 1
    return cores


def twoscale_vertical(stage_specs, rps, slo_ms):
    """One TwoScale vertical decision.

    Each stage keeps one batch. Cores stay on individual replicas and only
    increase. Stage latency is the slowest replica. The batch and the spread
    are chosen together so end-to-end latency fits the SLO and each stage
    covers workload + margin with as few cores as possible.
    """
    if rps == 0:
        return [
            {
                "cpu_by_id": {replica_id: cpu for replica_id, cpu in zip(spec["ids"], spec["cores"])},
                "batch": 1,
                "timeout": 0,
            }
            for spec in stage_specs
        ]

    demand = round(rps + std_addition * (rps ** 0.5))
    options = []
    for spec in stage_specs:
        found = {}
        for batch in range(1, spec["bmax"] + 1):
            cores = spread_cores(spec["cores"], batch, spec["model"], spec["cmax"], demand)
            if cores_throughput(cores, batch, spec["model"]) < demand:
                continue
            slowest = min(cores)
            service = latency(slowest, batch, *spec["model"])
            queuing = get_queuing(batch, rps)
            latency_ms = service + queuing
            if latency_ms > slo_ms:
                continue
            found[batch] = (tuple(cores), latency_ms, sum(cores), queuing)
        options.append(found)

    n = len(stage_specs)
    choice = [[None] * (slo_ms + 1) for _ in range(n)]
    for s in range(n):
        for used in range(slo_ms, -1, -1):
            if s == 0:
                if used != 0:
                    continue
                base_cost = 0
            else:
                if used == 0 or choice[s - 1][used] is None:
                    continue
                base_cost = choice[s - 1][used][0]
            for batch, (cores, latency_ms, cost, queuing) in options[s].items():
                nxt = used + latency_ms
                if nxt > slo_ms:
                    continue
                total = base_cost + cost
                if choice[s][nxt] is None or total < choice[s][nxt][0]:
                    choice[s][nxt] = (total, batch, cores, queuing, latency_ms)

    best_at = None
    best_cost = math.inf
    for used in range(slo_ms):
        picked = choice[n - 1][used]
        if picked is not None and picked[0] < best_cost:
            best_cost = picked[0]
            best_at = used

    if best_at is None:
        planned = []
        for spec in stage_specs:
            cores = spread_cores(spec["cores"], spec["batch"], spec["model"], spec["cmax"], demand)
            planned.append({
                "cpu_by_id": {replica_id: cpu for replica_id, cpu in zip(spec["ids"], cores)},
                "batch": spec["batch"],
                "timeout": spec["timeout"],
            })
        return planned

    planned = [None] * n
    used = best_at
    for s in range(n - 1, -1, -1):
        _total, batch, cores, queuing, latency_ms = choice[s][used]
        spec = stage_specs[s]
        planned[s] = {
            "cpu_by_id": {replica_id: cpu for replica_id, cpu in zip(spec["ids"], cores)},
            "batch": batch,
            "timeout": queuing,
        }
        used -= latency_ms
    return planned


if __name__ == "__main__":
    batch_max = 4
    core_max = 4
    slo_max = 300
    memory_stages = {0: 2, 1: 1}
    models_set = {0: [39.90889958924582, 7.784141716703453, 1.8333345492230113, 0.4507066167132669],
                1: [28.084275010842124, 2.626952412159743, 6.807172001409088, 0.4588339579256442]}

    current_workload = 150
    config_current = {0: [1, 1, 1, 0], 1: [1, 1, 1, 0]}
    config_vertical = vertical_2d([batch_max, batch_max], [core_max, core_max], slo_max,
                                  models_set, config_current, current_workload)
    config_horizontal = horizontal_2d([batch_max, batch_max], [core_max, core_max], slo_max, models_set, memory_stages, current_workload)
    print('Current Config: {MODEL: CORE, INSTANCE, BATCH, TIMEOUT}')
    print(config_current)
    print('Horizontal Config: {MODEL: [INSTANCE, BATCH]}')
    print(config_horizontal)
    print('Vertical Config: {MODEL: [CORE, BATCH]}')
    print(config_vertical)
