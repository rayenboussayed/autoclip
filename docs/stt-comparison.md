# STT models x speech enhancement - measured on one scene

> Generated with `tools/compare_stt.py`; the raw numbers are in `runs/stt-compare/results.json`.
> Reproduce: `python tools/compare_stt.py VIDEO --start 4956 --duration 120`


**Scene:** `jutilities-package.mp4`, 4956-5076 s (120 s), Moroccan-dialect Arabic tech talk.
The window was chosen because the first full run showed a language drift ("falando about",
Portuguese) inside it.

**Reference:** `raw` audio + `large-v3` (the best model on this content).

## 1. Why `sherpa-onnx` instead of `clearvoice`

| package | numpy | other pins | py3.14 | verdict |
| --- | --- | --- | --- | --- |
| `clearvoice` 0.1.2 | `<2` | `scenedetect==0.6.6`, `opencv-python==4.10` | ok | would downgrade the venv (numpy 2.5.4→1.26) and break `scene[0].seconds` (PySceneDetect ≥0.7) |
| `deepfilternet` 0.5.6 | `<2` | torch + a Rust toolchain | ok | numpy downgrade; 48 kHz only |
| `resemble-enhance` 0.0.1 | `==1.26.2` | `torch==2.1.1` | ok | ancient pins, unmaintained since 2024-12 |
| `espnet` | `>=2.0` | torch 2.11 | **needs <3.14** | not installable on this Python |
| `asteroid` | ok | torch + lightning + torchmetrics | ok | research toolkit, not a drop-in |
| `speechbrain` 1.1.2 | ok | torch ≥2.7.1 | ok | works, but drags ~2.5 GB of torch into the venv |
| **`sherpa-onnx` 1.13.8** | ok (onnxruntime) | one dep (`sherpa-onnx-core`) | **cp314 wheel** | **chosen** |

`sherpa-onnx` needs no torch, does not touch numpy, keeps 16 kHz (Whisper's rate) and its DPDFNet
models are documented for "16 kHz downstream ASR". The models are 0.5-15 MB and come from the
sherpa-onnx GitHub release or from `Ceva-IP/DPDFNet` on Hugging Face.

## 2. The matrix (5 audio conditions × 5 STT models)

CER is measured against `raw`/`large-v3`; `ar%` is the share of Arabic characters (a language-drift
detector); `logp` is Whisper's own average log-probability (higher = more confident).

| condition | model | CER | lang | p | logp | seg | ar% | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | tiny | 0.768 | ar | 0.38 | -3.236 | 25 | 92.1 | 25.2 |
| raw | base | 0.972 | **pt** | 0.30 | -2.370 | 13 | **0.0** | 25.2 |
| raw | small | 0.785 | ar | 0.99 | -1.856 | 13 | 85.0 | 30.7 |
| raw | medium | 0.686 | ar | 0.96 | -1.437 | 57 | 90.6 | 46.6 |
| raw | large-v3 | 0.000 | ar | 0.99 | **-0.489** | 14 | 93.6 | 16.2 |
| gtcrn | large-v3 | 0.589 | ar | 0.99 | -0.522 | 30 | **73.7** | 20.7 |
| dpdfnet2 | large-v3 | 0.427 | ar | 0.99 | -0.482 | 16 | 95.0 | 12.2 |
| dpdfnet8 | large-v3 | 0.391 | ar | 0.99 | -0.524 | 13 | 94.4 | 19.3 |
| noisereduce | large-v3 | 0.476 | ar | 1.00 | -0.515 | 25 | **68.1** | 29.1 |

## 3. What the data says

**`base` is broken on this content.** It detects Portuguese, Welsh or Latin on *every* condition and
transcribes 0 % Arabic - the language detection, not the noise, is the problem. Every other model
detects Arabic with 0.96-1.00.

**`large-v3` is the best model**: the highest confidence (-0.489), the highest Arabic share, the
fewest hallucinations. On the whole video it also produces the cleanest English: 1.6 % English-word
segments versus 2.7 % for `small`, and the English it keeps is coherent ("clean up dot for each
function") while `small` produces mixed junk ("و نضع cela ح cowork it andclick it").

**Denoising clean audio makes the transcript worse.** The scene has a signal-to-noise ratio of
**81.8 dB** - there is no noise to remove, so a denoiser can only take signal away. The effect:

* `gtcrn` and `noisereduce` (aggressive suppression) turn the second half of the Arabic transcript
  into **English** - the Arabic share drops from 93.6 % to 73.7 % / 68.1 %.
* The DPDFNet models keep the Arabic (94-95 %) but change the wording and add trailing
  hallucinations ("أقوم بإغلاق المدفع" = "I close the cannon", "تصنيع الوثائق").
* No attenuation setting fixes it: with the limit from 6 dB to 100 dB the CER stays between 0.28 and
  0.50. Only the DPDFNet models support the limit at all, and a *larger* value allows *more*
  suppression.
* For scale: transcribing the **same** raw audio twice gives CER ≈ 0.1 (Whisper on the GPU is not
  bit-reproducible), so the enhancement's effect is 3-5× the measurement noise.

**Conclusion:** the enhancement is not free, and on a clean recording it is harmful. It must only
run where there is noise to remove.

## 4. What was built

* `transcript.enhance` is **enabled by default** (`sherpa` / `gtcrn`), but every piece of audio is
  measured first (`enhance.snr_threshold_db`, default 20 dB) and left alone when it is already
  clean. On this video all 23 pieces score 53-106 dB, so nothing is enhanced and the transcript
  keeps the quality of the un-enhanced run.
* The transcription always runs in pieces of `processing.chunk_minutes` (5 minutes) - the denoiser
  and Whisper never hold more than that, and the language is detected once and reused.
* `transcript.model` is now `large-v3`.
* The enhancement settings are part of the transcript cache key, so changing them re-transcribes.
* `compute.device` ("cpu" / "gpu" / "auto") decides whether the GPU is used at all: with "cpu" the
  transcriber runs on the CPU and the clips are encoded with libx264.

## 5. Cost of the denoisers (CPU only - the sherpa-onnx wheel has no CUDA provider)

| model | size | RTF | 112 min video |
| --- | --- | --- | --- |
| `gtcrn` | 0.5 MB | 0.21 | 22 min |
| `dpdfnet2` | 10 MB | 0.47 | 53 min |
| `dpdfnet8` | 15 MB | 1.20 | 135 min |

More threads make it *slower* (0.21 at 4 threads, 0.30 at 16) - the models are too small to scale.
`gtcrn` is the default for that reason.

## 6. Cost of the transcriber

Measured on this machine (16 cores, RTX 3070):

| model | GPU (fp16) | CPU (int8) |
| --- | --- | --- |
| `base` | 25 s / 120 s (4.8x realtime) | 176 s / 90 s (**0.51x realtime**) |
| `large-v3` | 16 s / 120 s (7.4x realtime) | ~0.1x realtime (extrapolated: ~13 h for 112 min) |

With `compute.device: "cpu"` a 112-minute video therefore needs a much smaller
`transcript.model` - `base` still takes about 3.7 hours there.  The GPU is what makes
`large-v3` practical.
