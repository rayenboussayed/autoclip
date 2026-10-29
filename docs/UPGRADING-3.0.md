# Upgrading from 2.x to 3.0

- Python floor is now 3.11 (`requires-python = ">=3.11"`). Python 3.10 is no longer supported.
- The default transcription model is `large-v3` on a CUDA GPU, `small` otherwise. Set
  `transcript.model` explicitly if you relied on the old `large-v3` default everywhere.
- CLI is now subcommand based: `autoclip run cfg.json` (the old `autoclip cfg.json` still works
  and prints a one-line deprecation note). `--version` is now supported. `autoclip serve`
  starts the HTTP service. `autoclip doctor` / `autoclip models` / `autoclip download` are new.
- New config keys: `transcript.model_dir`, `transcript.offline`. Both are optional and default to
  the previous behaviour.
- `tkinterdnd2` moved to the `gui` extra; core install no longer requires Tk.
- A release now ships `install.sh` / `install.ps1`: install with
  `curl -fsSL https://github.com/rayenboussayed/autoclip/releases/latest/download/install.sh | sh`.
- Outputs are written to `<name>.part` and renamed on success, so a killed job leaves no
  half-written `.mp4`.