# Logging pipeline

`autoclip` uses Python's logging package as a structured runtime log, not just as a console
styling layer.

## What is logged

Every production function is decorated with `autoclip.core.log.traced`.

`traced` supports both forms:

```python
@traced
def run() -> None: ...


@traced("render_job", logging.INFO)
def render_job() -> None: ...
```

It provides:

- `[START] name` and `[END] name - Elapsed time: ...` for the interactive output;
- structured `START` / `END` (and `FAIL`) lines with wall time, free RAM, CPU and memory pressure;
- `RESOURCES.after_operation()` to shrink caches and free memory after operations;
- per-subtype labels (`probe`, `render_job`, `process_video`, ...);
- nested calls are not re-logged: one call stack logs its outermost frame only, so helpers do not
  bury the real signal under thousands of lines.

`traced` does not log from the logger internals (file handler, `print` tee, resource probe and
setup path) to avoid infinite recursion.

## Files and durability

Logs live at `{output_dir}/logs/autoclip_<timestamp>.log`. Logging is always on; the minimum level is `DEBUG` in debug mode (`AUTOCLIP_DEBUG=1`) and `INFO` otherwise. There is no `OFF` level and no verbosity flag.

Both stdout print lines and logging records use the same file handler. The file handler:

1. writes the record,
2. flushes the Python buffer,
3. calls `os.fsync()` on the file descriptor.

That happens for every `.log` entry, so a completed line is durable before the call that logged it
returns. A crash or a killed worker should not lose prior lines. On non-critical filesystems the
`fsync` failure is deliberately ignored, but the line was still written and flushed first.

`sys.stdout` is replaced with a tee when a log file is configured. The tee mirrors complete lines
back into the logger, so console output and `.log` contain the same lines.

## Sensitive data

No secrets are intentionally logged:

- URLs are passed through `redact_url` before joining the log;
- local user home paths are folded to `~/`;
- system health profiles include OS, CPU count, memory capacity, load envelope and
  selected output/log/cache roots only;
- command arguments from child processes are not dumped raw.

## Operational checks

A user may verify a run log in two ways:

```bash
grep -E '^\[(START|END)\]' /path/to/logs/autoclip_<timestamp>.log | head
grep -E 'FAIL +[^ ]+' /path/to/logs/autoclip_<timestamp>.log
```

Look for `FAIL` followed by an exception type; a missing final `END` means the worker was killed
or crashed before it returned.
