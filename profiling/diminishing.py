
import os
import json



model_to_title = {
    "detector": "Object Detection", 
    "classifier": "Object Classifier", 
    # "languageidentification": "Lanaguage Identification", 
    # "translation": "Translation"
}

def batch_cost_latency_model(cpu_batch_tuple, alpha, beta, gamma, zeta):
        cpu, batch = cpu_batch_tuple
        return alpha * batch / cpu + beta * batch + gamma / cpu + zeta


model_params = {}

for model in model_to_title:
    with open(f"{os.path.dirname(__file__)}/models/{model}/parameters-{99}.json") as f:
        model_params[model] = json.load(f)


model_max_cpu = {}
for model in model_params:
    for cpu in range(1, 8):
        l1 = batch_cost_latency_model((cpu, 1), **model_params[model])
        l2 = batch_cost_latency_model((cpu + 1, 1), **model_params[model])
        diff = 100 * (l1 - l2) / l1
        
        if diff < 15:
            model_max_cpu[model] = cpu
            break
print(model_max_cpu)
