from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np


APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

MPL_CONFIG_DIR = APP_ROOT / "tmp-checkpoint" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))


GESTURE_WINDOW_SIZE = 200
SWIPE_WINDOW_SIZE = 60
PINCH_WINDOW_SIZE = 60
TAP_WINDOW_SIZE = 20
WHISPER_WINDOW_SIZE = 200


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    checkpoint: str
    status: str
    input_desc: str
    total_ms: float | None = None
    avg_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    error: str = ""


@dataclass(frozen=True)
class BenchmarkJob:
    name: str
    checkpoint: str
    input_desc: str
    load_fn: Callable[[], Callable[[], object]]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    kind: Literal["swipe", "openvino", "tap", "inception_pth", "motion", "whisper"]
    checkpoint: str
    frame_count: int
    seed: int


@dataclass(frozen=True)
class TotalBenchmarkStats:
    serial_total_ms: float
    parallel_total_ms: float
    process_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark average inference latency for models under checkpoint/."
    )
    parser.add_argument("--iterations", type=int, default=50, help="Timed iterations per model.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per model.")
    parser.add_argument(
        "--whisper-frames",
        type=int,
        default=WHISPER_WINDOW_SIZE,
        help="Input frame count for whisper models.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for dummy inputs.")
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[BenchmarkSpec]:
    return [
        BenchmarkSpec(
            name="swipe-cnn",
            kind="swipe",
            checkpoint="checkpoint/swipe/cls",
            frame_count=SWIPE_WINDOW_SIZE,
            seed=args.seed + 0,
        ),
        BenchmarkSpec(
            name="pinch-cnn",
            kind="swipe",
            checkpoint="checkpoint/swipe/cls-pinch",
            frame_count=PINCH_WINDOW_SIZE,
            seed=args.seed + 1,
        ),
        BenchmarkSpec(
            name="it-pinch",
            kind="openvino",
            checkpoint="checkpoint/it-pinch/it-pinch.xml",
            frame_count=GESTURE_WINDOW_SIZE,
            seed=args.seed + 2,
        ),
        BenchmarkSpec(
            name="it-snap",
            kind="openvino",
            checkpoint="checkpoint/it-snap/it-snap.xml",
            frame_count=GESTURE_WINDOW_SIZE,
            seed=args.seed + 3,
        ),
        BenchmarkSpec(
            name="it-pinch-pth",
            kind="inception_pth",
            checkpoint="checkpoint/it-pinch.pth",
            frame_count=GESTURE_WINDOW_SIZE,
            seed=args.seed + 4,
        ),
        BenchmarkSpec(
            name="it-snap-pth",
            kind="inception_pth",
            checkpoint="checkpoint/it-snap.pth",
            frame_count=GESTURE_WINDOW_SIZE,
            seed=args.seed + 5,
        ),
        BenchmarkSpec(
            name="tap-94",
            kind="tap",
            checkpoint="checkpoint/tap-94.pth",
            frame_count=TAP_WINDOW_SIZE,
            seed=args.seed + 6,
        ),
        BenchmarkSpec(
            name="tcnlstm",
            kind="motion",
            checkpoint="checkpoint/tcnlstm.pt",
            frame_count=1,
            seed=args.seed + 7,
        ),
        BenchmarkSpec(
            name="whisper-word",
            kind="whisper",
            checkpoint="checkpoint/whisper-word",
            frame_count=args.whisper_frames,
            seed=args.seed + 8,
        ),
        BenchmarkSpec(
            name="whisper-number",
            kind="whisper",
            checkpoint="checkpoint/whisper-number",
            frame_count=args.whisper_frames,
            seed=args.seed + 9,
        ),
    ]


def benchmark_runner(
    *,
    job: BenchmarkJob,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    try:
        runner = job.load_fn()
    except Exception as exc:
        return BenchmarkResult(
            name=job.name,
            checkpoint=job.checkpoint,
            status="ERROR",
            input_desc=job.input_desc,
            error=str(exc),
        )

    try:
        for _ in range(max(warmup, 0)):
            runner()

        timings_ms: list[float] = []
        total_started = time.perf_counter()
        for _ in range(max(iterations, 1)):
            started = time.perf_counter()
            runner()
            timings_ms.append((time.perf_counter() - started) * 1000.0)
        total_ms = (time.perf_counter() - total_started) * 1000.0
    except Exception as exc:
        return BenchmarkResult(
            name=job.name,
            checkpoint=job.checkpoint,
            status="ERROR",
            input_desc=job.input_desc,
            error=str(exc),
        )

    timings = np.asarray(timings_ms, dtype=np.float64)
    return BenchmarkResult(
        name=job.name,
        checkpoint=job.checkpoint,
        status="OK",
        input_desc=job.input_desc,
        total_ms=float(total_ms),
        avg_ms=float(np.mean(timings)),
        p50_ms=float(np.percentile(timings, 50)),
        p95_ms=float(np.percentile(timings, 95)),
        min_ms=float(np.min(timings)),
        max_ms=float(np.max(timings)),
    )


def build_openvino_job(
    *,
    name: str,
    checkpoint: Path,
    rng: np.random.Generator,
) -> BenchmarkJob:
    input_desc = f"1x6x{GESTURE_WINDOW_SIZE}"

    def load_fn() -> Callable[[], object]:
        from openvino import Core

        core = Core()
        compiled = core.compile_model(model=str(checkpoint), device_name="CPU")
        input_layer = compiled.input(0)
        output_layer = compiled.output(0)
        sample = rng.normal(size=(1, 6, GESTURE_WINDOW_SIZE)).astype(np.float32)

        def runner() -> object:
            return compiled({input_layer: sample})[output_layer]

        return runner

    return BenchmarkJob(
        name=name,
        checkpoint=str(checkpoint.relative_to(APP_ROOT)),
        input_desc=input_desc,
        load_fn=load_fn,
    )


def build_swipe_like_job(
    *,
    name: str,
    checkpoint_dir: Path,
    recognizer_cls: type,
    frame_count: int,
    rng: np.random.Generator,
) -> BenchmarkJob:
    input_desc = f"{frame_count}x6"

    def load_fn() -> Callable[[], object]:
        recognizer = recognizer_cls(checkpoint_dir, device="cpu")
        sample = rng.normal(size=(frame_count, recognizer.input_dim)).astype(np.float32)

        def runner() -> object:
            return recognizer.predict(sample)

        return runner

    return BenchmarkJob(
        name=name,
        checkpoint=str(checkpoint_dir.relative_to(APP_ROOT)),
        input_desc=input_desc,
        load_fn=load_fn,
    )


def build_tap_job(
    *,
    checkpoint: Path,
    rng: np.random.Generator,
) -> BenchmarkJob:
    input_desc = f"1x6x{TAP_WINDOW_SIZE}"

    def load_fn() -> Callable[[], object]:
        import torch
        from tsai.models.InceptionTime import InceptionTime

        checkpoint_obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint_obj, dict) or "model" not in checkpoint_obj:
            raise RuntimeError(f"invalid checkpoint: {checkpoint}")

        model = InceptionTime(6, 4, nf=16)
        model.load_state_dict(checkpoint_obj["model"])
        model.eval()
        sample = torch.from_numpy(rng.normal(size=(1, 6, TAP_WINDOW_SIZE)).astype(np.float32))

        def runner() -> object:
            with torch.inference_mode():
                return model(sample)

        return runner

    return BenchmarkJob(
        name="tap-94",
        checkpoint=str(checkpoint.relative_to(APP_ROOT)),
        input_desc=input_desc,
        load_fn=load_fn,
    )


def build_inception_pth_job(
    *,
    name: str,
    checkpoint: Path,
    frame_count: int,
    rng: np.random.Generator,
) -> BenchmarkJob:
    input_desc = f"1x6x{frame_count}"

    def load_fn() -> Callable[[], object]:
        import torch
        from tsai.models.InceptionTime import InceptionTime

        checkpoint_obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint_obj, dict) or "model" not in checkpoint_obj:
            raise RuntimeError(f"invalid checkpoint: {checkpoint}")

        state_dict = checkpoint_obj["model"]
        bottleneck_weight = state_dict.get("inceptionblock.inception.0.bottleneck.weight")
        fc_weight = state_dict.get("fc.weight")
        if bottleneck_weight is None or fc_weight is None:
            raise RuntimeError(f"unexpected inception checkpoint layout: {checkpoint}")

        input_channels = int(bottleneck_weight.shape[1])
        nf = int(bottleneck_weight.shape[0])
        output_dim = int(fc_weight.shape[0])

        model = InceptionTime(input_channels, output_dim, nf=nf)
        model.load_state_dict(state_dict)
        model.eval()
        sample = torch.from_numpy(rng.normal(size=(1, input_channels, frame_count)).astype(np.float32))

        def runner() -> object:
            with torch.inference_mode():
                return model(sample)

        return runner

    return BenchmarkJob(
        name=name,
        checkpoint=str(checkpoint.relative_to(APP_ROOT)),
        input_desc=input_desc,
        load_fn=load_fn,
    )


def build_motion_job(
    *,
    checkpoint: Path,
    rng: np.random.Generator,
) -> BenchmarkJob:
    input_desc = "1x1x6 (streaming step)"

    def load_fn() -> Callable[[], object]:
        import torch

        from models.tcn_lstm import TCNLSTMStreaming

        model = TCNLSTMStreaming(6, 128, 2, [64, 128], 3, 0.2)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        sample = torch.from_numpy(rng.normal(size=(1, 1, 6)).astype(np.float32))

        with torch.inference_mode():
            for _ in range(model.receptive_field):
                model(sample)

        def runner() -> object:
            with torch.inference_mode():
                return model(sample)

        return runner

    return BenchmarkJob(
        name="tcnlstm",
        checkpoint=str(checkpoint.relative_to(APP_ROOT)),
        input_desc=input_desc,
        load_fn=load_fn,
    )


def build_whisper_job(
    *,
    name: str,
    checkpoint_dir: Path,
    recognizer_cls: type,
    frame_count: int,
    rng: np.random.Generator,
) -> BenchmarkJob:
    input_desc = f"{frame_count}x6"

    def load_fn() -> Callable[[], object]:
        recognizer = recognizer_cls(checkpoint_dir, device="cpu")
        effective_frame_count = min(frame_count, recognizer.max_input_length)
        sample = rng.normal(size=(effective_frame_count, 6)).astype(np.float32)

        def runner() -> object:
            return recognizer.predict(sample)

        return runner

    return BenchmarkJob(
        name=name,
        checkpoint=str(checkpoint_dir.relative_to(APP_ROOT)),
        input_desc=input_desc,
        load_fn=load_fn,
    )


def build_job(spec: BenchmarkSpec) -> BenchmarkJob:
    rng = np.random.default_rng(spec.seed)
    checkpoint_path = APP_ROOT / spec.checkpoint

    if spec.kind == "swipe":
        from models.pinch_cnn_recognizer import PinchCNNRecognizer
        from models.swipe_cnn_recognizer import SwipeCNNRecognizer

        recognizer_cls = SwipeCNNRecognizer if spec.name == "swipe-cnn" else PinchCNNRecognizer
        return build_swipe_like_job(
            name=spec.name,
            checkpoint_dir=checkpoint_path,
            recognizer_cls=recognizer_cls,
            frame_count=spec.frame_count,
            rng=rng,
        )

    if spec.kind == "openvino":
        return build_openvino_job(
            name=spec.name,
            checkpoint=checkpoint_path,
            rng=rng,
        )

    if spec.kind == "tap":
        return build_tap_job(
            checkpoint=checkpoint_path,
            rng=rng,
        )

    if spec.kind == "inception_pth":
        return build_inception_pth_job(
            name=spec.name,
            checkpoint=checkpoint_path,
            frame_count=spec.frame_count,
            rng=rng,
        )

    if spec.kind == "motion":
        return build_motion_job(
            checkpoint=checkpoint_path,
            rng=rng,
        )

    if spec.kind == "whisper":
        from whisper.whisper_number_recognizer import WhisperNumberRecognizer
        from whisper.whisper_word_recognizer import WhisperWordRecognizer

        recognizer_cls = WhisperWordRecognizer if spec.name == "whisper-word" else WhisperNumberRecognizer
        return build_whisper_job(
            name=spec.name,
            checkpoint_dir=checkpoint_path,
            recognizer_cls=recognizer_cls,
            frame_count=spec.frame_count,
            rng=rng,
        )

    raise ValueError(f"unsupported benchmark kind: {spec.kind}")


def run_spec(spec: BenchmarkSpec, args: argparse.Namespace) -> BenchmarkResult:
    return benchmark_runner(job=build_job(spec), warmup=args.warmup, iterations=args.iterations)


def build_results(args: argparse.Namespace, specs: list[BenchmarkSpec]) -> list[BenchmarkResult]:
    return [run_spec(spec, args) for spec in specs]


def benchmark_in_subprocess(
    spec: BenchmarkSpec,
    args: argparse.Namespace,
    ready_queue: mp.Queue,
    result_queue: mp.Queue,
    start_event: mp.synchronize.Event,
) -> None:
    try:
        job = build_job(spec)
        runner = job.load_fn()
        for _ in range(max(args.warmup, 0)):
            runner()
    except Exception as exc:
        ready_queue.put(("error", spec.name, str(exc)))
        return

    ready_queue.put(("ready", spec.name, ""))
    start_event.wait()

    try:
        started = time.perf_counter()
        for _ in range(max(args.iterations, 1)):
            runner()
        result_queue.put(("done", spec.name, (time.perf_counter() - started) * 1000.0))
    except Exception as exc:
        result_queue.put(("error", spec.name, str(exc)))


def measure_parallel_total_runtime(args: argparse.Namespace, specs: list[BenchmarkSpec]) -> float:
    ctx = mp.get_context("spawn")
    ready_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    start_event = ctx.Event()
    processes: list[mp.Process] = []

    for spec in specs:
        proc = ctx.Process(target=benchmark_in_subprocess, args=(spec, args, ready_queue, result_queue, start_event))
        proc.start()
        processes.append(proc)

    ready_ok = 0
    for _ in specs:
        status, _name, _payload = ready_queue.get()
        if status == "ready":
            ready_ok += 1

    parallel_started = time.perf_counter()
    start_event.set()

    for _ in range(ready_ok):
        result_queue.get()

    for proc in processes:
        proc.join()

    parallel_total_ms = (time.perf_counter() - parallel_started) * 1000.0
    ready_queue.close()
    ready_queue.join_thread()
    result_queue.close()
    result_queue.join_thread()

    return parallel_total_ms


def format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def print_results(
    results: list[BenchmarkResult],
    args: argparse.Namespace,
    totals: TotalBenchmarkStats,
) -> None:
    print(f"checkpoint latency benchmark (warmup={args.warmup}, iterations={args.iterations})")
    print(
        f"{'model':<16} {'status':<8} {'input':<20} {'total_ms':>10} "
        f"{'avg_ms':>10} {'p50_ms':>10} {'p95_ms':>10} {'min_ms':>10} {'max_ms':>10}"
    )
    print("-" * 112)
    for result in results:
        print(
            f"{result.name:<16} {result.status:<8} {result.input_desc:<20} "
            f"{format_metric(result.total_ms):>10} {format_metric(result.avg_ms):>10} "
            f"{format_metric(result.p50_ms):>10} {format_metric(result.p95_ms):>10} "
            f"{format_metric(result.min_ms):>10} {format_metric(result.max_ms):>10}"
        )
        print(f"  checkpoint: {result.checkpoint}")
        if result.error:
            print(f"  error: {result.error}")

    ok_count = sum(1 for result in results if result.status == "OK")
    print("-" * 112)
    print(f"completed: {ok_count}/{len(results)} models")
    print(
        f"serial total: {totals.serial_total_ms:.3f} ms | "
        f"parallel total ({totals.process_count} processes): {totals.parallel_total_ms:.3f} ms"
    )
    if totals.parallel_total_ms > 0:
        print(f"serial/parallel ratio: {totals.serial_total_ms / totals.parallel_total_ms:.3f}x")
    print(
        "notes: swipe/pinch use 60-frame windows, OpenVINO gestures use 200 frames, "
        "tap uses 20 frames, tcnlstm measures steady-state 1-frame streaming latency."
    )


def main() -> int:
    args = parse_args()
    specs = build_specs(args)
    results = build_results(args, specs)
    serial_total_ms = float(sum(result.total_ms or 0.0 for result in results))
    parallel_total_ms = measure_parallel_total_runtime(args, specs)
    totals = TotalBenchmarkStats(
        serial_total_ms=serial_total_ms,
        parallel_total_ms=parallel_total_ms,
        process_count=len(specs),
    )
    print_results(results, args, totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
