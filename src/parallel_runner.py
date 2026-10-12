"""Runs SimuHome episodes several at a time, in this process.

Approach 1 ran one episode at a time (SimuHome's evaluator runs a model's episodes
sequentially, one simulator per model). With ~45 s per hosted-LLM call that used
under 2 of the 40 requests/minute a free NVIDIA key allows. Here each worker
thread owns one simulator server (its own port) and runs SimuHome's own
`run_single_config` (the same agent, scorer and judge panel). All LLM calls
still pass through the notebook's request shim, whose global throttle keeps the
total under the key's rate limit however many threads run.

Results are written exactly where and how SimuHome's evaluator writes them
(`<output_root>/<run_id>/<safe model name>/<episode file name>`, same record and
error-record format), so SimuHome's aggregator reads them unchanged.

Imports SimuHome lazily (call after SimuHome is on sys.path, with cwd = SimuHome).
"""

from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_simulators(n: int, simuhome_dir: str = ".", wait_seconds: float = 60.0, log_dir: Optional[str] = None):
    """Start n SimuHome simulator servers; returns [(port, Popen)]."""
    import requests

    servers = []
    for i in range(n):
        port = _free_port()
        env = dict(os.environ, SERVER_PORT=str(port))
        log = open(os.path.join(log_dir, f"sim_{port}.log"), "w") if log_dir else subprocess.DEVNULL
        proc = subprocess.Popen([sys.executable, "-m", "src.simulator.api.app"], cwd=simuhome_dir, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        servers.append((port, proc))
    deadline = time.time() + wait_seconds
    for port, proc in servers:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"simulator on port {port} exited early (code {proc.returncode})")
            try:
                if requests.get(f"http://127.0.0.1:{port}/api/__health__", timeout=1).ok:
                    break
            except Exception:
                pass
            if time.time() > deadline:
                raise RuntimeError(f"simulator on port {port} did not come up in {wait_seconds:.0f} s")
            time.sleep(0.5)
    return servers


def stop_simulators(servers) -> None:
    for _, proc in servers:
        if proc.poll() is None:
            proc.terminate()
    for _, proc in servers:
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def episode_paths(benchmark_dir: str, qt: str, case: str, seeds) -> list[Path]:
    out = []
    for s in seeds:
        p = Path(benchmark_dir) / f"{qt}_{case}_seed_{s}.json"
        if p.exists():
            out.append(p)
    return out


def run_episodes(
    cfg_paths: list[Path],
    *,
    servers,
    run_id: str,
    model: str,
    make_main_llm: Callable[[], object],
    make_judge_llms: Callable[[], list],
    output_root: str = "experiments",
    max_steps: int = 20,
    timeout: float = 120.0,
    on_episode_start: Optional[Callable[[str], None]] = None,
    skip_done: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    """Run every episode in `cfg_paths` with len(servers) workers. Returns counts."""
    from src.cli.episode_evaluator import (
        _build_error_record,
        _classify_runtime_failure,
        _is_valid_existing_result,
        _output_path_for_config,
        _write_json_atomic,
    )
    from src.pipelines.episode_evaluation.runner import run_single_config

    ports: "queue.Queue[int]" = queue.Queue()
    for port, _ in servers:
        ports.put(port)
    counts = {"total": len(cfg_paths), "skipped": 0, "success": 0, "failed": 0}
    lock = threading.Lock()
    t_start = time.time()

    def one(cfg_path: Path) -> None:
        out_path = _output_path_for_config(cfg_path, output_root, run_id, model)
        if skip_done and _is_valid_existing_result(out_path):
            with lock:
                counts["skipped"] += 1
            return
        port = ports.get()
        try:
            if on_episode_start:
                on_episode_start(f"{run_id}/{cfg_path.stem}")
            t0 = time.time()
            try:
                record = run_single_config(
                    cfg_path=str(cfg_path), base_url=f"http://127.0.0.1:{port}/api", timeout=float(timeout),
                    max_steps=int(max_steps), agent_strategy="react",
                    main_llm=make_main_llm(), judge_llms=make_judge_llms(),
                )
                _write_json_atomic(out_path, record)
                score = (record.get("evaluation_result") or {}).get("score")
                with lock:
                    counts["success"] += 1
                log(f"  {cfg_path.stem}: score={score} ({time.time() - t0:.0f} s)")
            except Exception as exc:
                score, error_type, category, attempted = _classify_runtime_failure(exc)
                _write_json_atomic(out_path, _build_error_record(
                    cfg_path=cfg_path, model=model, strategy="react", error_type=error_type, message=str(exc),
                    score=score, error_category=category, attempted=attempted))
                with lock:
                    counts["failed"] += 1
                log(f"  {cfg_path.stem}: FAILED {error_type}: {str(exc)[:160]}")
        finally:
            ports.put(port)

    with ThreadPoolExecutor(max_workers=max(1, len(servers))) as pool:
        list(pool.map(one, cfg_paths))

    counts["seconds"] = round(time.time() - t_start, 1)
    summary_path = Path(output_root) / run_id / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"run_id": run_id, "model": model, "workers": len(servers), **counts}, indent=2))
    return counts
