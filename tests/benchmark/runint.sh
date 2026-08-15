uv sync --dev
uv run run_benchmark.py llh --int
uv run run_benchmark.py cond --int
uv run run_benchmark.py pred --int
uv sync --group cuda
uv run run_benchmark.py llh --int --gpu
uv run run_benchmark.py cond --int --gpu
uv run run_benchmark.py pred --int --gpu
