# HVScale

## Instructions
1. Create a Kubernetes cluster (on all cluster nodes)
    1. Install Microk8s: [Get started](https://microk8s.io/docs/getting-started) (for channel, use `--channel=1.34/beta`)
    2. microk8s enable dns

2. on Master node run: `microk8s add-node`
3. Copy the join command (with --worker) and run it on worker nodes
4. Activate a Python3 virtualenv
5. Install requirements: `pip install -r requirements.txt`
6. Run an experiment: ` python starter.py video hv manual 1`
