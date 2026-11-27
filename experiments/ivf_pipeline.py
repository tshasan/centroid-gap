"""Shared explicit state machines for IVF benchmark artifact staging and downloads."""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from pathlib import Path
from typing import Optional


class Stage(str, Enum):
    """Explicit stages of the file-backed IVF workload benchmark pipeline."""

    DATA = "data"
    INDEX = "index"
    BASELINE = "baseline"
    FITTED = "fitted"
    SHIFT_CONTROL = "shift_control"
    PQ_REPLAY = "pq_replay"


UPSTREAM = {
    "ranks.npz": ("flat.faiss",),
    "flat-baseline.json": ("flat.faiss",),
    "flat-fitted.json": ("flat.faiss",),
    "pq.json": ("flat.faiss", "router-at1.npy", "router-at10.npy"),
    "shift.json": ("flat-baseline.json", "flat-fitted.json"),
}


def _required_files(stage: Stage, workload: str) -> tuple[str, ...]:
    splits = ("val", "test", "fit", "image") if workload == "t2i" else ("val", "test", "fit")
    if stage is Stage.DATA:
        return (f"{workload}-base.npy", f"{workload}-workload.json",
                *(f"{workload}-{s}.npy" for s in splits))
    if stage is Stage.INDEX:
        return ("owner.npy", "sizes.npy", "centers.npy", "flat.faiss", "index.json",
                *(f"{s}.npz" for s in splits))
    if stage is Stage.BASELINE:
        return ("flat-baseline.json", "ranks.npz")
    if stage is Stage.FITTED:
        return ("flat-fitted.json", "router-at1.npy", "router-at10.npy",
                "challenger-at1.npy", "challenger-at10.npy")
    if stage is Stage.SHIFT_CONTROL:
        return ("shift.json",)
    if stage is Stage.PQ_REPLAY:
        return ("pq.json",)
    raise ValueError(f"unknown stage: {stage}")


def require_stage(directory: Path, workload: str, stage: Stage) -> None:
    """Fail fast with a clear stage error if prerequisite artifacts are missing or stale."""
    directory = Path(directory)
    files = _required_files(stage, workload)
    missing = [name for name in files if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"pipeline stage '{stage.value}' incomplete for workload '{workload}' in {directory}; "
            f"missing: {', '.join(missing)}"
        )
    check_stage_order(directory, files)


def check_stage_order(directory: Path, names) -> None:
    """Verify every stage artifact is newer than its declared upstream artifacts."""
    directory = Path(directory)
    for name in names:
        for source in UPSTREAM.get(name, ()):
            if (directory / name).stat().st_mtime_ns < (directory / source).stat().st_mtime_ns:
                raise ValueError(f"{directory / name} predates {source}; rerun the stage that writes it")


@dataclass
class StreamDownloadState:
    """Explicit state machine for exact-length, SHA-256-verified chunked HTTP downloads."""

    label: str
    expected_bytes: int
    expected_sha256: Optional[str] = None
    chunk_size: int = 1 << 20
    received: int = 0
    _hasher: object = field(default_factory=hashlib.sha256, repr=False)

    def feed_prefix(self, data: bytes, output) -> None:
        if data:
            self._hasher.update(data)
            output.write(data)
            self.received += len(data)

    @property
    def remaining(self) -> int:
        return self.expected_bytes - self.received

    def step(self, response, output) -> bool:
        if self.remaining <= 0:
            return False
        block = response.read(min(self.remaining, self.chunk_size))
        if not block:
            raise ValueError(f"{self.label}: truncated download after {self.received}/{self.expected_bytes} bytes")
        self._hasher.update(block)
        output.write(block)
        self.received += len(block)
        return self.remaining > 0

    def finish(self) -> str:
        if self.received != self.expected_bytes:
            raise ValueError(f"{self.label}: expected {self.expected_bytes} bytes, got {self.received}")
        digest = self._hasher.hexdigest()
        if self.expected_sha256 is not None and digest != self.expected_sha256:
            raise ValueError(f"{self.label}: SHA-256 mismatch")
        return digest

    def stream_all(self, response, output) -> str:
        while self.step(response, output):
            pass
        return self.finish()
