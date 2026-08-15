uv sync --dev
uv run run_benchmark.py llh
uv run run_benchmark.py cond
uv run run_benchmark.py pred
uv sync --group cuda
uv run run_benchmark.py llh --gpu
uv run run_benchmark.py cond --gpu
uv run run_benchmark.py pred --gpu
