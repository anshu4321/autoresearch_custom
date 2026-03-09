# Dashboard

Live UI for monitoring retrieval autoresearch runs.

## Run locally

```bash
cd /path/to/repo
python3 dashboard/server.py --repo-root . --port 8787
```

Then open: `http://127.0.0.1:8787`

## Run on Lambda and tunnel to your laptop

On Lambda VM:

```bash
cd ~/autoresearch_custom
nohup python3 dashboard/server.py --repo-root . --host 0.0.0.0 --port 8787 > dashboard.log 2>&1 &
```

On your laptop:

```bash
ssh -i /path/to/First_one.pem -L 8787:127.0.0.1:8787 ubuntu@<INSTANCE_IP>
```

Then open: `http://127.0.0.1:8787`

## Data sources

- `results_retrieval.tsv`
- `run_retrieval.log`
