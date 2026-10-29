"""The desktop app: a native window that drives the same pipeline as the command line.

The window is built around three things: the form (every option of the config file, with an
explanation), the details panel (what / why / when for the option you are looking at) and the log
(the output of the run). The run itself is a child process of ``python -m autoclip``, so it can be
stopped at any time and the app stays responsive.
"""

from __future__ import annotations

import contextlib
import os
import platform
import queue
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from types import TracebackType
from typing import Any

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    _TkinterDnD = TkinterDnD
    _DND_FILES = DND_FILES
except Exception:  # pragma: no cover - depends on the installed extras
    TkinterDnD = None
    DND_FILES = None
    _TkinterDnD = None
    _DND_FILES = None

import threading

from ..core.errors import ConfigError
from .fields import SECTIONS, Option
from .form import Form, ScrollFrame, unflatten
from .session import (
    PipelineHandle,
    cancel_pipeline,
    check_config,
    default_config_path,
    default_values,
    open_folder,
    read_config,
    run_pipeline,
    write_config,
)


def system_theme() -> str:
    """The theme that looks native on this platform."""
    style = ttk.Style()
    names = style.theme_names()
    wanted = {
        "win32": ("vista", "winnative", "clam"),
        "darwin": ("aqua", "clam"),
    }.get(platform.system(), ("clam", "adwaita"))
    for name in wanted:
        if name in names:
            return name
    return style.theme_use()


def debug_enabled(env: dict[str, str] | None = None) -> bool:
    """Is the app in debug mode? ``AUTOCLIP_DEBUG=1`` (or true/yes/on) turns it on.

    In debug mode an error in a Tk callback prints its traceback, closes the window and is raised
    again, so a test fails instead of passing with the error only in the console.
    """
    if env is None:
        env = dict(os.environ)
    return env.get("AUTOCLIP_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


class AutoclipApp(tk.Tk):
    """The application window."""

    def __init__(self, config_path: Path | None = None, *, debug: bool | None = None) -> None:
        super().__init__()
        self._dnd = False
        dnd = globals().get("_TkinterDnD")
        if dnd is not None:
            try:
                dnd.require(self)
                self._dnd = True
            except Exception:
                self._dnd = False
        self.dnd = self._dnd
        self._debug = debug_enabled() if debug is None else debug
        self.title("autoclip - cut long videos into clips")
        self.geometry("1060x760")
        self.minsize(900, 620)
        self._configure_style()

        self.config_path = Path(config_path) if config_path is not None else default_config_path()
        self.values: dict[str, Any] = default_values()
        self._forms: list[Form] = []
        self._handle: PipelineHandle | None = None
        self._lines: queue.Queue[str] = queue.Queue()

        self._build_header()
        self._build_body()
        self._build_footer()
        self._startup()
        self._load_into_form()
        self._after_id = self.after(100, self._drain_log)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout ------------------------------------------------------------------------------

    def _configure_style(self) -> None:
        """Use the platform theme and a smaller, quieter font for the help lines."""
        style = ttk.Style(self)
        with contextlib.suppress(tk.TclError):  # the platform theme is always there
            style.theme_use(system_theme())
        style.configure("Help.TLabel", foreground="#666666")
        style.configure("Title.TLabel", font=("TkDefaultFont", 13, "bold"))
        style.configure("Sub.TLabel", foreground="#666666")
        style.configure("Status.TLabel", foreground="#333333")

    def _build_header(self) -> None:
        """Title, config file and the shortcut to the output folder."""
        header = ttk.Frame(self, padding=(14, 10, 14, 6))
        header.pack(fill="x")
        ttk.Label(header, text="autoclip", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="cut long videos into the best short and long clips", style="Sub.TLabel").pack(side="left", padx=(10, 0))
        ttk.Button(header, text="Open output folder", command=self._open_output).pack(side="right")

        row = ttk.Frame(self, padding=(14, 0, 14, 8))
        row.pack(fill="x")
        ttk.Label(row, text="Config file", width=12, anchor="w").pack(side="left")
        self._path_var = tk.StringVar(value=str(self.config_path))
        entry = ttk.Entry(row, textvariable=self._path_var)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        entry.bind("<FocusOut>", lambda _e: self._remember_path(self._path_var.get()))
        ttk.Button(row, text="Browse...", command=self._choose_config).pack(side="left")

    def _build_body(self) -> None:
        """The notebook with the option groups and the details panel."""
        body = ttk.PanedWindow(self, orient="vertical")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 8))

        self.notebook = ttk.Notebook(body)
        for section in SECTIONS:
            scroll = ScrollFrame(self.notebook)
            form = Form(scroll.inner, section.options, self.values, self._on_change, wheel=scroll.scroll_by)
            form.pack(fill="both", expand=True)
            self._forms.append(form)
            self.notebook.add(scroll, text=section.title)
        body.add(self.notebook, weight=4)

        details = ttk.LabelFrame(body, text="What this option does", padding=(12, 8, 12, 10))
        details.columnconfigure(1, weight=1)
        ttk.Label(details, text="What", width=8, anchor="nw", style="Sub.TLabel").grid(row=0, column=0, sticky="nw")
        self.what_label = ttk.Label(
            details, text="Click an option to read what it does.", style="Sub.TLabel", wraplength=880, justify="left"
        )
        self.what_label.grid(row=0, column=1, sticky="w")
        ttk.Label(details, text="Why", width=8, anchor="nw", style="Sub.TLabel").grid(row=1, column=0, sticky="nw", pady=(4, 0))
        self.why_label = ttk.Label(details, text="", wraplength=880, justify="left")
        self.why_label.grid(row=1, column=1, sticky="w", pady=(4, 0))
        ttk.Label(details, text="When", width=8, anchor="nw", style="Sub.TLabel").grid(row=2, column=0, sticky="nw", pady=(4, 0))
        self.when_label = ttk.Label(details, text="", wraplength=880, justify="left")
        self.when_label.grid(row=2, column=1, sticky="w", pady=(4, 0))
        body.add(details, weight=0)

        log_pane = ttk.LabelFrame(body, text="Log", padding=(6, 4, 6, 6))
        log_pane.rowconfigure(0, weight=1)
        log_pane.columnconfigure(0, weight=1)
        self.log = tk.Text(log_pane, height=9, wrap="word", font="TkFixedFont", state="disabled")
        log_scroll = ttk.Scrollbar(log_pane, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")
        body.add(log_pane, weight=1)

    def _build_footer(self) -> None:
        """Buttons, log window and status line."""
        buttons = ttk.Frame(self, padding=(14, 0, 14, 6))
        buttons.pack(fill="x")
        self.run_button = ttk.Button(buttons, text="Run", command=self._run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(6, 18))
        ttk.Button(buttons, text="Save config", command=self._save).pack(side="left")
        ttk.Button(buttons, text="Save as...", command=self._save_as).pack(side="left", padx=6)
        ttk.Button(buttons, text="Load config...", command=self._load).pack(side="left")
        ttk.Button(buttons, text="Reset", command=self._reset).pack(side="left", padx=6)
        ttk.Button(buttons, text="All options...", command=self._show_reference).pack(side="left")
        self.status = ttk.Label(buttons, text="Ready.", style="Status.TLabel")
        self.status.pack(side="right")

        button_row = ttk.Frame(self, padding=(14, 0, 14, 10))
        button_row.pack(fill="x")
        ttk.Button(button_row, text="Copy log", command=self._copy_log).pack(side="right")
        ttk.Button(button_row, text="Copy error", command=self._copy_error).pack(side="right", padx=(0, 6))

    # -- startup -----------------------------------------------------------------------------

    def _startup(self) -> None:
        """Load the config file, or keep the defaults in memory when there is none.

        A missing file is not an error and opens no dialog: the defaults are the config until the
        user saves one (Save / Save as) or runs the pipeline, which writes the file first.
        """
        path = self.config_path
        if path.is_file():
            self._load_at_startup(path)
            return
        self._set_status("No config file - the defaults are loaded.")
        self._append(f"No config file at {path}: the defaults are loaded in memory. Save config to create one.")

    def _load_at_startup(self, path: Path) -> None:
        """Use a valid file as it is; a file that is not a valid config falls back to the defaults."""
        try:
            values = read_config(path)
            check_config(unflatten({**self.values, **values}))
        except (OSError, ValueError, ConfigError) as e:
            self.values = default_values()
            self._append(f"error: {path} is not a valid config: {e}")
            self._append("Reset to the defaults. Save config (or Run) to replace the file.")
            self._set_status("The config file was not valid - reset to the defaults.")
            return
        self.values.update(values)
        self._set_status(f"Loaded {path.name}")

    # -- values ------------------------------------------------------------------------------

    def _on_change(self, path: str, value: Any) -> None:
        """A value changed: remember it and update the options that depend on it."""
        self.values[path] = value
        for form in self._forms:
            form.refresh_visibility()

    def _load_into_form(self) -> None:
        """Put the current values into the controls."""
        for form in self._forms:
            for path, field in form.fields.items():
                field.set(self.values.get(path))

    def _remember_path(self, text: str) -> None:
        """The config file the user typed."""
        if text.strip():
            self.config_path = Path(text.strip()).expanduser()

    def _choose_config(self) -> None:
        """Pick the config file."""
        picked = filedialog.askopenfilename(
            title="Choose a config file",
            initialdir=str(self.config_path.parent),
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")],
        )
        if picked:
            self.config_path = Path(picked)
            self._path_var.set(str(self.config_path))
            self._read_config(self.config_path)

    def _read_config(self, path: Path) -> None:
        """Read a config file into the form."""
        try:
            values = read_config(path)
        except (OSError, ValueError, ConfigError) as e:
            messagebox.showerror("autoclip", f"Cannot read the config file:\n{e}")
            return
        self.values.update(values)
        self._load_into_form()
        self._set_status(f"Loaded {path.name}")

    def _config_data(self) -> dict[str, Any]:
        """The current form as a nested config dict."""
        return unflatten({path: field.value() for form in self._forms for path, field in form.fields.items()})

    def _save_to(self, path: Path) -> bool:
        """Write the form to a config file; return whether it was written."""
        try:
            write_config(path, self._config_data())
        except OSError as e:
            messagebox.showerror("autoclip", f"Cannot write the config file:\n{e}")
            return False
        self.config_path = path
        self._path_var.set(str(path))
        self._set_status(f"Saved {path}")
        return True

    def _save(self) -> None:
        """Save the config file."""
        self._save_to(self.config_path)

    def _save_as(self) -> None:
        """Save the config file under a new name."""
        picked = filedialog.asksaveasfilename(
            title="Save the config file",
            initialdir=str(self.config_path.parent),
            defaultextension=".json",
            filetypes=[("JSON config", "*.json")],
        )
        if picked:
            self._save_to(Path(picked))

    def _load(self) -> None:
        """Load a config file."""
        picked = filedialog.askopenfilename(
            title="Load a config file",
            initialdir=str(self.config_path.parent),
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")],
        )
        if picked:
            self.config_path = Path(picked)
            self._path_var.set(str(self.config_path))
            self._read_config(self.config_path)

    def _reset(self) -> None:
        """Put the defaults back into the form."""
        self.values = default_values()
        self._load_into_form()
        self._set_status("Reset to the defaults")

    # -- running -----------------------------------------------------------------------------

    def _run(self) -> None:
        """Validate the form, write the config and start the pipeline."""
        if self._process is not None:
            return
        data = self._config_data()
        try:
            check_config(data)
        except ConfigError as e:
            self._append(f"error: {e}")
            self._set_status("The configuration is not valid - see the log.")
            messagebox.showerror("autoclip", f"The configuration is not valid:\n\n{e}")
            return
        if not self._save_to(self.config_path):
            self._append(f"error: cannot write {self.config_path}; the run was not started.")
            self._set_status("The config file could not be written - the run was not started.")
            return
        self._append(f"$ autoclip {self.config_path}")
        from .session import CancelToken as _CancelToken

        cancel = _CancelToken()

        def on_line(line: str) -> None:
            self._lines.put(line)

        worker = threading.Thread(
            target=self._run_worker,
            kwargs={"config_path": self.config_path, "cancel": cancel, "on_line": on_line},
            daemon=True,
        )
        handle = PipelineHandle(cancel=cancel, thread=worker)
        handle.thread.start()
        self._handle = handle
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._set_status("Running...")

    def _read_output(self) -> None:
        """Forward the output of the pipeline to the log window."""
        # Output is produced by the worker thread through self._lines directly; nothing to read here.
        return

    def _run_worker(self, config_path: Any, cancel: Any, on_line: Any) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            pass
        try:
            run_pipeline(config_path, cancel=cancel, on_line=on_line)
        except Exception as exc:  # pragma: no cover - defensive
            on_line(f"error: {type(exc).__name__}: {exc}")
            on_line("")

    @property
    def _process(self) -> PipelineHandle | None:
        return self._handle

    @_process.setter
    def _process(self, value: PipelineHandle | None) -> None:
        self._handle = value

    def _drain_log(self) -> None:
        """Move the lines of the pipeline into the log window (on the main thread)."""
        try:
            if not self.winfo_exists():
                return
            while True:
                line = self._lines.get_nowait()
                if line == "":
                    self._handle = None
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self._set_status("Finished.")
                else:
                    self._append(line)
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

    def _stop(self) -> None:
        """Stop the running pipeline."""
        handle = self._handle
        if handle is None:
            return
        self._append("stopping...")
        cancel_pipeline(handle)
        self._set_status("Stopped.")

    def _open_output(self) -> None:
        """Open the output folder in the file manager."""
        target = Path(self.values.get("output_dir") or "clips")
        if not target.is_absolute():
            target = (self.config_path.parent / target).resolve()
        try:
            open_folder(target)
        except OSError as e:
            messagebox.showerror("autoclip", f"Cannot open the output folder:\n{e}")

    # -- log and status -----------------------------------------------------------------------

    def _append(self, line: str) -> None:
        """Add a line to the log window."""
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        """Show a message in the status bar."""
        self.status.configure(text=text)

    def _log_text(self) -> str:
        """Everything the log window shows (without the trailing newline Tk keeps)."""
        return self.log.get("1.0", "end-1c").removesuffix("\n")

    def _copy(self, text: str, message: str) -> None:
        """Put ``text`` on the clipboard."""
        self.clipboard_clear()
        self.clipboard_append(text)
        self._set_status(message)

    def _copy_error(self) -> None:
        """Put the last error line on the clipboard (a disabled log cannot be selected)."""
        errors = [line for line in self._log_text().splitlines() if line.strip() and "error" in line.lower()]
        if not errors:
            self._set_status("There is no error in the log to copy.")
            return
        self._copy(errors[-1], "Copied the error to the clipboard.")

    def _copy_log(self) -> None:
        """Put the whole log on the clipboard."""
        text = self._log_text()
        if not text.strip():
            self._set_status("The log is empty.")
            return
        self._copy(text, "Copied the log to the clipboard.")

    def _on_close(self) -> None:
        """Stop a running pipeline and close the window."""
        if self._process is not None:
            self._stop()
        self.destroy()

    def report_callback_exception(self, exc: type[BaseException], value: BaseException, tb: TracebackType | None) -> None:
        """Handle an error in a Tk callback.

        Normally Tk prints the traceback and keeps running (the default). In debug mode the
        traceback is printed, the window is closed and the error is raised again, so a test that
        triggered the callback fails instead of passing with the error only in the console.
        """
        if not self._debug:
            super().report_callback_exception(exc, value, tb)
            return
        traceback.print_exception(exc, value, tb)
        with contextlib.suppress(tk.TclError):
            self.destroy()
        raise value.with_traceback(tb)

    def destroy(self) -> None:
        """Close the window (the pending timer is cancelled first)."""
        with contextlib.suppress(tk.TclError):
            self.after_cancel(self._after_id)
        super().destroy()

    # -- help --------------------------------------------------------------------------------

    def show_details(self, option: Option) -> None:
        """Show the full explanation of an option."""
        self.what_label.configure(text=option.what)
        self.why_label.configure(text=option.why)
        self.when_label.configure(text=option.when)

    def _show_reference(self) -> None:
        """A window with every option and its explanation."""
        ReferenceWindow(self)


class ReferenceWindow(tk.Toplevel):
    """A searchable list of all options with their what / why / when."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.title("autoclip - all options")
        self.geometry("900x620")
        ttk.Label(self, text="Every option of the config file, with what it does, why you would change it and when.", padding=(12, 8)).pack(
            fill="x"
        )
        search = ttk.Frame(self, padding=(12, 0))
        search.pack(fill="x")
        ttk.Label(search, text="Search", width=8, anchor="w").pack(side="left")
        self._query = tk.StringVar()
        ttk.Entry(search, textvariable=self._query).pack(side="left", fill="x", expand=True)
        self._query.trace_add("write", self._filter)

        pane = ttk.PanedWindow(self, orient="vertical")
        pane.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.tree = ttk.Treeview(pane, columns=("label", "path"), show="headings", height=18)
        self.tree.heading("label", text="Option")
        self.tree.heading("path", text="Config path")
        self.tree.column("label", width=320, anchor="w")
        self.tree.column("path", width=280, anchor="w")
        self.tree.bind("<<TreeviewSelect>>", self._show)
        pane.add(self.tree, weight=1)
        self.text = tk.Text(pane, height=12, wrap="word", state="disabled")
        pane.add(self.text, weight=1)
        self.options = {o.path: o for s in SECTIONS for o in s.options}
        self._filter("")

    def _filter(self, *_args: Any) -> None:
        """Show the options that match the search."""
        needle = self._query.get().lower().strip()
        for item in self.tree.get_children():
            self.tree.delete(item)
        for path, option in self.options.items():
            haystack = f"{path} {option.label} {option.what}".lower()
            if needle and needle not in haystack:
                continue
            self.tree.insert("", "end", iid=path, values=(option.label, path))

    def _show(self, _event: tk.Event | None = None) -> None:
        """Show the explanation of the selected option."""
        selection = self.tree.selection()
        if not selection:
            return
        option = self.options.get(selection[0])
        if option is None:
            return
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", f"{option.label}  ({option.path})\n\nWHAT\n{option.what}\n\nWHY\n{option.why}\n\nWHEN\n{option.when}\n")
        self.text.configure(state="disabled")


def launch(config_path: Path | None = None) -> int:
    """Open the desktop app."""
    app = AutoclipApp(config_path)
    app.mainloop()
    return 0
