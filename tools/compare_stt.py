"""Compare STT models and speech-enhancement backends on one scene of a video.

The tool takes a fixed window of a source video, produces several *audio conditions*
(the untouched audio plus the output of each enhancement backend) and transcribes
every condition with every requested faster-whisper model.  It then reports how much
the transcripts agree, so a backend can be judged on its effect on the words the STT
engine actually receives.

Usage::

    python tools/compare_stt.py VIDEO [--start S] [--duration D]
                                      [--models tiny,base,small,medium,large-v3]
                                      [--conditions raw,gtcrn,dpdfnet2,dpdfnet8,noisereduce]
                                      [--out DIR]

Nothing here is imported by the ``autoclip`` package: it is a development aid, so the
project keeps no runtime dependency on a metric library.  The character error rate is
computed with a small edit distance over Arabic-normalised text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoclip.media import decode_audio

F32 = NDArray[np.float32]
SR = 16000  # what faster-whisper expects

#: Enhancement backends and the file each needs (sherpa-onnx release assets).
MODELS_DIR = Path("models")
SHERPA_MODELS: dict[str, str] = {
    "gtcrn": "gtcrn_simple",
    "dpdfnet_baseline": "dpdfnet_baseline",
    "dpdfnet2": "dpdfnet2",
    "dpdfnet4": "dpdfnet4",
    "dpdfnet8": "dpdfnet8",
}
SHERPA_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/{name}.onnx"

#: Arabic letters that are written differently but pronounced the same way.
_ARABIC_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})  # noqa: RUF001
_DIACRITICS = re.compile(r"[\u064b-\u0652\u0640]")
_PUNCT = re.compile(r"[^\w\u0600-\u06ff]+", re.UNICODE)


# --------------------------------------------------------------------------------------
# text metrics
# --------------------------------------------------------------------------------------
def normalise(text: str) -> str:
    """Fold a transcript to comparable characters (no spaces, no punctuation, folded Arabic)."""
    text = unicodedata.normalize("NFC", text)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_ARABIC_FOLD)
    text = _PUNCT.sub(" ", text)
    return "".join(text.split())


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (iterative, single row)."""
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Character error rate of ``hypothesis`` against ``reference`` (1.0 = no match)."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def script_ratio(text: str) -> tuple[int, int]:
    """How many Arabic and Latin letters the text contains."""
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    latin = sum(1 for c in text if ("a" <= c.lower() <= "z"))
    return arabic, latin


# --------------------------------------------------------------------------------------
# audio conditions
# --------------------------------------------------------------------------------------
def sherpa_denoiser(model: str, model_path: Path, threads: int, attenuation_db: float) -> Callable[[F32], F32]:
    """A callable that enhances 16 kHz mono audio with a sherpa-onnx denoiser model."""
    import sherpa_onnx

    gtcrn = sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig()
    dpdfnet = sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig()
    if model.startswith("dpdfnet"):
        dpdfnet.model = str(model_path)
        dpdfnet.attenuation_limit_db = attenuation_db
    else:
        gtcrn.model = str(model_path)
    config = sherpa_onnx.OfflineSpeechDenoiserConfig(
        model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(gtcrn, dpdfnet, threads, False, "cpu")
    )
    if not config.validate():
        raise RuntimeError(f"invalid sherpa-onnx denoiser config for {model}")
    denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)

    def enhance(audio: F32) -> F32:
        out = denoiser.run(audio, SR)
        return np.asarray(out.samples, dtype=np.float32)

    return enhance


def noisereduce_denoiser(strength: float = 0.75) -> Callable[[F32], F32]:
    """A callable that enhances 16 kHz mono audio with spectral gating (noisereduce)."""
    import noisereduce

    def enhance(audio: F32) -> F32:
        return np.asarray(
            noisereduce.reduce_noise(y=audio, sr=SR, stationary=False, n_jobs=4, prop_decrease=strength),
            dtype=np.float32,
        )

    return enhance


def build_condition(name: str, models_dir: Path, threads: int, attenuation_db: float) -> Callable[[F32], F32] | None:
    """The enhancement callable for a condition name (``raw`` returns ``None``)."""
    if name == "raw":
        return None
    if name == "noisereduce":
        return noisereduce_denoiser()
    if name not in SHERPA_MODELS:
        raise SystemExit(f"unknown condition {name!r}")
    model_path = models_dir / f"{SHERPA_MODELS[name]}.onnx"
    if not model_path.is_file():
        raise SystemExit(f"missing model {model_path} (run with --models-dir to point at the download)")
    return sherpa_denoiser(name, model_path, threads, attenuation_db)


# --------------------------------------------------------------------------------------
# transcription
# --------------------------------------------------------------------------------------
@dataclass
class Run:
    """One transcription of one audio condition."""

    condition: str
    model: str
    text: str
    language: str
    language_probability: float
    avg_logprob: float
    no_speech_prob: float
    segments: int
    seconds: float
    lines: list[str] = field(default_factory=list)

    @property
    def arabic_chars(self) -> int:
        return script_ratio(self.text)[0]

    @property
    def latin_chars(self) -> int:
        return script_ratio(self.text)[1]


def transcribe(audio: F32, model_name: str, device: str = "cuda", beam_size: int = 5) -> Run:
    """Transcribe 16 kHz audio exactly like ``autoclip.transcript.transcribe`` does."""
    from faster_whisper import WhisperModel

    whisper = WhisperModel(
        model_name,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
        cpu_threads=8,
    )
    started = time.perf_counter()
    stream, info = whisper.transcribe(
        audio,
        vad_filter=True,
        word_timestamps=True,
        language=None,  # auto-detect, like the production config
        condition_on_previous_text=False,
        beam_size=beam_size,
    )
    segments = list(stream)
    seconds = time.perf_counter() - started
    texts = [s.text.strip() for s in segments]
    logprobs = [s.avg_logprob for s in segments]
    speech = [s.no_speech_prob for s in segments]
    return Run(
        condition="",
        model=model_name,
        text=" ".join(t for t in texts if t),
        language=info.language,
        language_probability=info.language_probability,
        avg_logprob=sum(logprobs) / len(logprobs) if logprobs else 0.0,
        no_speech_prob=sum(speech) / len(speech) if speech else 0.0,
        segments=len(segments),
        seconds=seconds,
        lines=texts,
    )


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]  # keep a background run observable
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, default=4956.0, help="window start in seconds")
    parser.add_argument("--duration", type=float, default=120.0, help="window length in seconds")
    parser.add_argument("--models", default="tiny,base,small,medium,large-v3")
    parser.add_argument("--conditions", default="raw,gtcrn,dpdfnet2,dpdfnet8,noisereduce")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--out", type=Path, default=Path("runs/stt-compare"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attenuation-db", type=float, default=12.0)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"decoding {args.video.name} {args.start:.0f}s..{args.start + args.duration:.0f}s")
    audio = decode_audio(args.video, args.start, args.duration, SR)
    print(f"  {len(audio)} samples @ {SR} Hz")

    runs: list[Run] = []
    for condition in conditions:
        enhance = build_condition(condition, args.models_dir, 4, args.attenuation_db)
        if enhance is None:
            conditioned = audio
            ctime = 0.0
        else:
            t0 = time.perf_counter()
            conditioned = enhance(audio)
            ctime = time.perf_counter() - t0
            print(f"  condition {condition}: enhanced in {ctime:.1f}s (RTF {ctime / args.duration:.3f})")
        for model_name in models:
            run = transcribe(conditioned, model_name, device=args.device)
            run.condition = condition
            runs.append(run)
            print(
                f"    {condition:12} {model_name:9} lang={run.language}({run.language_probability:.2f}) "
                f"seg={run.segments:3} logp={run.avg_logprob:+.3f} "
                f"ar={run.arabic_chars:5} la={run.latin_chars:4} {run.seconds:6.1f}s"
            )

    reference = max(runs, key=lambda r: (r.model == "large-v3", r.model))
    print(f"\nreference: {reference.condition}/{reference.model}\n")
    header = f"{'condition':12} {'model':9} {'CER':>7} {'lang':>5} {'p':>5} {'logp':>7} {'seg':>4} {'ar%':>5} {'s':>6}"
    print(header)
    print("-" * len(header))
    for run in runs:
        cer = char_error_rate(reference.text, run.text)
        ratio = 100 * run.arabic_chars / max(1, run.arabic_chars + run.latin_chars)
        print(
            f"{run.condition:12} {run.model:9} {cer:7.3f} {run.language:>5} "
            f"{run.language_probability:5.2f} {run.avg_logprob:+7.3f} {run.segments:4d} {ratio:5.1f} {run.seconds:6.1f}"
        )

    payload = [
        {
            "condition": r.condition,
            "model": r.model,
            "cer_vs_reference": char_error_rate(reference.text, r.text),
            "language": r.language,
            "language_probability": r.language_probability,
            "avg_logprob": r.avg_logprob,
            "no_speech_prob": r.no_speech_prob,
            "segments": r.segments,
            "arabic_chars": r.arabic_chars,
            "latin_chars": r.latin_chars,
            "seconds": r.seconds,
            "text": r.text,
            "lines": r.lines,
        }
        for r in runs
    ]
    (args.out / "results.json").write_text(
        json.dumps({"reference": f"{reference.condition}/{reference.model}", "runs": payload}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    for run in runs:
        (args.out / f"{run.condition}__{run.model}.txt").write_text("\n".join(run.lines), encoding="utf-8")

    print("\nlargest disagreements with the reference (for a manual audit):")
    for run in sorted(runs, key=lambda r: -char_error_rate(reference.text, r.text))[:6]:
        if run is reference:
            continue
        cer = char_error_rate(reference.text, run.text)
        ref_lines, hyp_lines = reference.lines, run.lines
        print(f"\n  {run.condition}/{run.model}  CER {cer:.3f}")
        for i in range(min(len(ref_lines), len(hyp_lines))):
            if normalise(ref_lines[i]) != normalise(hyp_lines[i]):
                print(f"    ref : {ref_lines[i][:90]}")
                print(f"    hyp : {hyp_lines[i][:90]}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
