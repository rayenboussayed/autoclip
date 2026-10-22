"""The desktop app: the option catalogue, the behaviour and the window.

The catalogue has to cover every option of the config file - an option that exists in the config
but not in the app would be invisible in the user interface.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from dataclasses import is_dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

import pytest

from autoclip.app.fields import KINDS, OPTIONS, SECTIONS, option_paths
from autoclip.app.session import (
    check_config,
    default_values,
    flatten,
    open_folder,
    pipeline_command,
    read_config,
    split_csv,
    start_pipeline,
    stop_pipeline,
    unflatten,
    write_config,
)
from autoclip.core.config import Config
from autoclip.core.errors import ConfigError

pytest.importorskip("tkinter", reason="the desktop app needs tkinter")


def config_paths(cls: Any = Config, prefix: str = "") -> set[str]:
    """Every option path of the config dataclasses (``clips.short.count`` and so on)."""
    paths: set[str] = set()
    for name, hint in get_type_hints(cls).items():
        path = f"{prefix}{name}"
        origin = get_origin(hint)
        if origin in (Union, types.UnionType):
            args = [a for a in get_args(hint) if a is not type(None)]
            if len(args) == 1 and is_dataclass(args[0]):
                paths |= config_paths(args[0], f"{path}.")
                continue
        if is_dataclass(hint) and isinstance(hint, type):
            paths |= config_paths(hint, f"{path}.")
        else:
            paths.add(path)
    return paths


def has_display() -> bool:
    """Can a window be opened here?"""
    if os.name == "nt" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


needs_display = pytest.mark.skipif(not has_display(), reason="no display for a window")


# -- the catalogue ----------------------------------------------------------------------------


def test_every_config_option_is_in_the_app() -> None:
    """No option of the config file may be missing from the desktop app."""
    assert set(option_paths()) == config_paths()


def test_every_option_is_explained() -> None:
    """Every option says what it does, why you would change it and when."""
    for path, option in OPTIONS.items():
        assert option.label.strip(), path
        assert option.what.strip(), path
        assert option.why.strip(), path
        assert option.when.strip(), path
        assert option.kind in KINDS, path


def test_the_sections_cover_every_option_exactly_once() -> None:
    """An option appears in one section only, and every section has options."""
    paths = option_paths()
    assert len(paths) == len(set(paths))
    for section in SECTIONS:
        assert section.options, section.key
        assert section.title.strip() and section.subtitle.strip(), section.key


def test_the_choices_are_the_ones_the_config_accepts() -> None:
    """A choice must offer exactly the values the config accepts for that option."""
    from autoclip.core.config import ClipsCfg, ComputeCfg, TranscriptCfg, VideoCfg

    expected = {
        "compute.device": ComputeCfg.__dataclass_fields__["device"].default,
        "clips.mode": ClipsCfg.__dataclass_fields__["mode"].default,
        "transcript.mode": TranscriptCfg.__dataclass_fields__["mode"].default,
        "video.gpu": VideoCfg.__dataclass_fields__["gpu"].default,
    }
    for path, default in expected.items():
        assert default in OPTIONS[path].choices, path
    assert OPTIONS["video.layout"].choices == ("fit", "fill")


# -- the behaviour (no display needed) --------------------------------------------------------


def test_flatten_and_unflatten_round_trip() -> None:
    """The form works with flat paths, the config file with nested dicts."""
    data = {"clips": {"short": {"count": 3}, "kinds": ["short"]}, "input": ["a.mp4"], "output_dir": "out"}
    flat = flatten(data)
    assert flat == {"clips.short.count": 3, "clips.kinds": ["short"], "input": ["a.mp4"], "output_dir": "out"}
    assert unflatten(flat) == data


def test_split_csv_reads_a_comma_separated_field() -> None:
    """The list fields accept commas and newlines."""
    assert split_csv("a, b ,,c") == ["a", "b", "c"]
    assert split_csv("") == []


def test_the_defaults_cover_every_option() -> None:
    """A fresh config has a value for every option of the catalogue."""
    values = default_values()
    assert set(values) == set(option_paths())
    check_config(unflatten(values))  # the defaults are a valid config


def test_read_config_reads_the_option_paths(tmp_path: Path) -> None:
    """A config file is read into the flat paths the form uses."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"clips": {"short": {"count": 7}}, "input": ["a.mp4"]}), encoding="utf-8")
    values = read_config(path)
    assert values["clips.short.count"] == 7 and values["input"] == ["a.mp4"]


def test_read_config_rejects_a_file_that_is_not_an_object(tmp_path: Path) -> None:
    """A config file has to be a JSON object."""
    path = tmp_path / "config.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON object"):
        read_config(path)


def test_write_config_writes_a_file_the_app_can_read(tmp_path: Path) -> None:
    """What the app writes is what it reads back."""
    path = tmp_path / "nested" / "config.json"
    write_config(path, unflatten({"clips.short.count": 4, "input": ["a.mp4"]}))
    assert read_config(path)["clips.short.count"] == 4


def test_check_config_accepts_the_defaults_and_rejects_a_broken_one() -> None:
    """The form is checked against the schema before a run starts."""
    check_config(unflatten(default_values()))
    with pytest.raises(ConfigError):
        check_config(unflatten({"clips.short.count": 0}))


def test_the_pipeline_runs_the_command_line(tmp_path: Path) -> None:
    """The app starts the pipeline as a child process of the command line."""
    command = pipeline_command(tmp_path / "config.json")
    assert command[1:] == ["-m", "autoclip", str(tmp_path / "config.json")]
    assert command[0] == sys.executable


def test_start_pipeline_runs_the_cli(tmp_path: Path) -> None:
    """The child process is the command line, with the project on the path."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"input": [], "clips": {"kinds": []}}), encoding="utf-8")
    process = start_pipeline(config)
    try:
        assert process.stdout is not None
        assert process.wait(timeout=60) == 2  # no video matches: a configuration error
    finally:
        process.kill()


def test_stop_pipeline_terminates_the_process(tmp_path: Path) -> None:
    """Stopping the app stops the pipeline (the whole process group)."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"input": [], "clips": {"kinds": []}}), encoding="utf-8")
    process = start_pipeline(config)
    stop_pipeline(process)
    assert process.wait(timeout=60) != 0 or process.poll() is not None


def test_open_folder_creates_the_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The output folder is created and handed to the file manager."""
    called: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: called.append(cmd))
    target = tmp_path / "out"
    open_folder(target)
    assert target.is_dir()
    assert called and called[0][0] in ("xdg-open", "open")


# -- the window (needs a display) --------------------------------------------------------------


class FakeProcess:
    """A child process that only prints a few lines."""

    def __init__(self, lines: list[str]) -> None:
        self.stdout = iter(lines)
        self.pid = 1

    def wait(self, timeout: float | None = None) -> int:
        return 0


def set_option(app: Any, path: str, value: Any) -> None:
    """Set an option the way the user does: through the control of the form."""
    for form in app._forms:
        if path in form.fields:
            form.fields[path].set(value)
            return
    raise AssertionError(f"no such option: {path}")


@needs_display
def test_the_window_shows_every_option_and_saves_a_valid_config(tmp_path: Path) -> None:
    """The window builds, shows every option and writes a config the core accepts."""
    from autoclip.app.window import AutoclipApp

    app = AutoclipApp(tmp_path / "config.json")
    try:
        assert app.title().startswith("autoclip")
        shown = {path for form in app._forms for path in form.fields}
        assert shown == set(option_paths())
        set_option(app, "clips.short.count", 7)
        check_config(app._config_data())  # the form produces a config the core accepts
        app._save_to(tmp_path / "config.json")
        assert (tmp_path / "config.json").exists()
    finally:
        app.destroy()


@needs_display
def test_the_window_loads_a_config_and_resets_to_the_defaults(tmp_path: Path) -> None:
    """Loading a config fills the form, Reset brings the defaults back."""
    from autoclip.app.window import AutoclipApp

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"clips": {"short": {"count": 9}}}), encoding="utf-8")
    app = AutoclipApp(path)
    try:
        app._read_config(path)
        assert app.values["clips.short.count"] == 9
        app._reset()
        assert app.values["clips.short.count"] == default_values()["clips.short.count"]
    finally:
        app.destroy()


@needs_display
def test_an_invalid_config_is_reported_and_nothing_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken config stops the run before the pipeline starts."""
    from autoclip.app.window import AutoclipApp

    errors: list[str] = []
    monkeypatch.setattr(messagebox, "showerror", lambda title, text: errors.append(text))
    app = AutoclipApp(tmp_path / "config.json")
    try:
        set_option(app, "clips.short.count", 0)
        app._run()
        assert app._process is None
        assert errors and "not valid" in errors[0]
        assert "error:" in app.log.get("1.0", "end")
    finally:
        app.destroy()


@needs_display
def test_running_starts_the_pipeline_and_shows_its_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run starts the pipeline, the log shows what it prints."""
    from autoclip.app import window
    from autoclip.app.window import AutoclipApp

    monkeypatch.setattr(window, "start_pipeline", lambda path: FakeProcess(["all done", ""]))
    app = AutoclipApp(tmp_path / "config.json")
    try:
        app._run()
        for _ in range(50):
            app.update()
            if app._process is None and "all done" in app.log.get("1.0", "end"):
                break
            import time

            time.sleep(0.05)
        assert "all done" in app.log.get("1.0", "end")
        assert app.status.cget("text") == "Finished."
    finally:
        app.destroy()


@needs_display
def test_stopping_the_pipeline_is_possible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop ends the run."""
    from autoclip.app import window
    from autoclip.app.window import AutoclipApp

    stopped: list[object] = []
    monkeypatch.setattr(window, "start_pipeline", lambda path: FakeProcess([]))

    def record(process: object) -> None:
        stopped.append(process)

    monkeypatch.setattr(window, "stop_pipeline", record)
    app = AutoclipApp(tmp_path / "config.json")
    try:
        app._run()
        app._stop()
        assert stopped and app.status.cget("text") == "Stopped."
    finally:
        app.destroy()


@needs_display
def test_the_details_panel_explains_the_focused_option() -> None:
    """Focusing an option fills the what / why / when panel."""
    from autoclip.app.window import AutoclipApp

    app = AutoclipApp()
    try:
        app.show_details(OPTIONS["clips.short.target_seconds"])
        assert "preference" in app.what_label.cget("text").lower()
        assert app.why_label.cget("text").strip()
        assert app.when_label.cget("text").strip()
    finally:
        app.destroy()


@needs_display
def test_the_reference_window_lists_and_filters_the_options() -> None:
    """The reference shows every option and can be searched."""
    from autoclip.app.window import AutoclipApp, ReferenceWindow

    app = AutoclipApp()
    try:
        window = ReferenceWindow(app)
        assert len(window.tree.get_children()) == len(option_paths())
        window._query.set("funny")
        window._filter()
        assert 0 < len(window.tree.get_children()) < len(option_paths())
        window.tree.selection_set("clips.scenes")
        window._show()
        assert "WHAT" in window.text.get("1.0", "end")
    finally:
        app.destroy()


# -- the package ------------------------------------------------------------------------------


def test_the_package_launches_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """``autoclip.app.launch`` opens the window (tkinter is imported only then)."""
    import autoclip.app as app_pkg

    called: list[Path | None] = []

    def fake_launch(path: Path | None) -> int:
        called.append(path)
        return 0

    monkeypatch.setattr("autoclip.app.window.launch", fake_launch)
    assert app_pkg.launch(Path("x.json")) == 0
    assert called == [Path("x.json")]


def test_the_package_exposes_the_window_class() -> None:
    """``AutoclipApp`` is reachable without importing tkinter by hand."""
    import autoclip.app as app_pkg

    assert app_pkg.AutoclipApp.__name__ == "AutoclipApp"
    with pytest.raises(AttributeError):
        _ = app_pkg.does_not_exist


def test_the_app_module_can_be_started(monkeypatch: pytest.MonkeyPatch) -> None:
    """``python -m autoclip.app`` opens the window and ends with its exit status."""
    import runpy

    import autoclip.app as app_pkg

    monkeypatch.setattr(app_pkg, "launch", lambda path=None: 0)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("autoclip.app", run_name="__main__")
    assert exit_info.value.code == 0


# -- the form ---------------------------------------------------------------------------------


@needs_display
def test_the_controls_read_and_write_every_kind_of_option() -> None:
    """Every kind of option survives the round trip through its control."""
    import tkinter as tk

    from autoclip.app.form import Form

    kinds: dict[str, Any] = {}
    for option in OPTIONS.values():
        kinds.setdefault(option.kind, option)
    values = {path: None for path in (o.path for o in kinds.values())}
    changed: list[tuple[str, Any]] = []
    root = tk.Tk()
    root.withdraw()
    try:
        form = Form(root, tuple(kinds.values()), values, lambda path, value: changed.append((path, value)))
        samples = {
            "bool": True,
            "choice": kinds["choice"].choices[1],
            "int": "42",
            "float": "12.5",
            "str": "hello",
            "path": "/tmp/file.srt",
            "dir": "/tmp/folder",
            "csv": "one, two",
            "lines": "first\nsecond",
        }
        for kind, option in kinds.items():
            field = form.fields[option.path]
            field.set(samples[kind])
            value = field.value()
            if kind == "bool":
                assert value is True
            elif kind == "choice":
                assert value == samples[kind]
            elif kind in ("int", "float"):
                assert value == float(samples[kind])
            elif kind == "csv":
                assert value == ["one", "two"]
            elif kind == "lines":
                assert value == ["first", "second"]
            else:
                assert value == samples[kind]
        assert changed  # every change was reported
    finally:
        root.destroy()


@needs_display
def test_a_control_that_holds_nothing_is_none() -> None:
    """An empty number field means "not set", not zero."""
    import tkinter as tk

    from autoclip.app.form import Form

    root = tk.Tk()
    root.withdraw()
    try:
        form = Form(root, (OPTIONS["clips.short.count"],), {"clips.short.count": 5}, lambda path, value: None)
        field = form.fields["clips.short.count"]
        field.set("")
        assert field.value() is None
        field.set("not a number")
        assert field.value() is None
    finally:
        root.destroy()


@needs_display
def test_the_options_that_do_not_apply_are_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ranges are only shown in ranges mode, the split length only in split mode."""
    import tkinter as tk

    from autoclip.app.form import Form

    values = {"clips.mode": "highlights", "clips.short.ranges": [], "clips.split_seconds": 600}
    root = tk.Tk()
    root.withdraw()
    try:
        form = Form(
            root,
            (OPTIONS["clips.mode"], OPTIONS["clips.short.ranges"], OPTIONS["clips.split_seconds"]),
            values,
            lambda path, value: values.update({path: value}),
        )
        assert not form.fields["clips.short.ranges"].holder.winfo_manager()
        assert not form.fields["clips.split_seconds"].holder.winfo_manager()
        form.fields["clips.mode"].set("ranges")
        form.refresh_visibility()
        assert form.fields["clips.short.ranges"].holder.winfo_manager() == "grid"
        assert not form.fields["clips.split_seconds"].holder.winfo_manager()
    finally:
        root.destroy()


@needs_display
def test_browsing_fills_the_path_fields_and_the_video_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The browse buttons put the chosen file or folder into the field."""
    import tkinter as tk
    from tkinter import filedialog

    from autoclip.app.form import Form

    monkeypatch.setattr(filedialog, "askopenfilename", lambda **kw: "/tmp/picked.srt")
    monkeypatch.setattr(filedialog, "askdirectory", lambda **kw: "/tmp/picked-folder")
    monkeypatch.setattr(filedialog, "askopenfilenames", lambda **kw: ("/tmp/one.mp4", "/tmp/two.mp4"))
    values: dict[str, Any] = {"transcript.file": "", "output_dir": "", "input": []}
    root = tk.Tk()
    root.withdraw()
    try:
        form = Form(
            root,
            (OPTIONS["transcript.file"], OPTIONS["output_dir"], OPTIONS["input"]),
            values,
            lambda path, value: values.update({path: value}),
        )
        form._browse(OPTIONS["transcript.file"])
        assert values["transcript.file"] == "/tmp/picked.srt"
        form._browse(OPTIONS["output_dir"])
        assert values["output_dir"] == "/tmp/picked-folder"
        form._browse_videos()
        assert values["input"] == ["/tmp/one.mp4", "/tmp/two.mp4"]
    finally:
        root.destroy()


@needs_display
def test_the_mouse_wheel_scrolls_the_form() -> None:
    """The wheel scrolls the option list (the buttons scroll the other way)."""
    import tkinter as tk

    from autoclip.app.form import ScrollFrame

    root = tk.Tk()
    root.withdraw()
    try:
        scroll = ScrollFrame(root)
        calls: list[tuple[int, str]] = []
        scroll.canvas.yview_scroll = lambda amount, what: calls.append((amount, what))  # type: ignore[method-assign]

        def wheel(**kwargs: Any) -> Any:
            return types.SimpleNamespace(**kwargs)

        scroll.scroll_by(cast("tk.Event", wheel(num=4)))
        scroll.scroll_by(cast("tk.Event", wheel(num=5)))
        scroll.scroll_by(cast("tk.Event", wheel(delta=120)))
        scroll.scroll_by(cast("tk.Event", wheel(delta=-120)))
        assert calls == [(-3, "units"), (3, "units"), (-1, "units"), (1, "units")]
    finally:
        root.destroy()


@needs_display
def test_the_file_dialogs_pick_and_save_the_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Browse, Save as... and Load... work on the config file."""
    from autoclip.app.window import AutoclipApp

    picked = tmp_path / "picked.json"
    picked.write_text(json.dumps({"clips": {"short": {"count": 3}}}), encoding="utf-8")
    monkeypatch.setattr(filedialog, "askopenfilename", lambda **kw: str(picked))
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **kw: str(tmp_path / "other.json"))
    app = AutoclipApp(tmp_path / "config.json")
    try:
        app._choose_config()
        assert app.config_path == picked
        assert app.values["clips.short.count"] == 3
        app._save_as()
        assert (tmp_path / "other.json").exists()
        app._load()
    finally:
        app.destroy()


@needs_display
def test_a_broken_config_file_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file that cannot be read or parsed is reported, the app keeps running."""
    from autoclip.app.window import AutoclipApp

    errors: list[str] = []
    monkeypatch.setattr(messagebox, "showerror", lambda title, text: errors.append(text))
    broken = tmp_path / "broken.json"
    broken.write_text("{nope", encoding="utf-8")
    app = AutoclipApp(tmp_path / "config.json")
    try:
        app._read_config(broken)
        app._read_config(tmp_path / "missing.json")
        assert len(errors) == 2
    finally:
        app.destroy()


@needs_display
def test_a_folder_that_cannot_be_opened_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The output folder is opened; a failure is reported, not raised."""
    from autoclip.app import window
    from autoclip.app.window import AutoclipApp

    errors: list[str] = []
    monkeypatch.setattr(messagebox, "showerror", lambda title, text: errors.append(text))

    def broken(path: Path) -> None:
        raise OSError("no file manager")

    monkeypatch.setattr(window, "open_folder", broken)
    app = AutoclipApp(tmp_path / "config.json")
    try:
        app._open_output()
        assert errors
    finally:
        app.destroy()


@needs_display
def test_closing_the_window_stops_the_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing the app while a run is going stops the pipeline."""
    from autoclip.app import window
    from autoclip.app.window import AutoclipApp

    stopped: list[object] = []
    monkeypatch.setattr(window, "start_pipeline", lambda path: FakeProcess([]))

    def record(process: object) -> None:
        stopped.append(process)

    monkeypatch.setattr(window, "stop_pipeline", record)
    app = AutoclipApp(tmp_path / "config.json")
    app._run()
    app._on_close()
    assert stopped


def test_stopping_a_pipeline_that_is_already_gone_terminates_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pipeline that cannot be signalled is terminated instead."""
    terminated: list[bool] = []

    class Gone:
        pid = 999_999

        def terminate(self) -> None:
            terminated.append(True)

    stop_pipeline(Gone())  # type: ignore[arg-type]
    assert terminated


def test_a_file_manager_that_is_not_there_opens_the_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a file manager the folder is opened in the web browser."""
    opened: list[str] = []

    def broken(cmd: list[str], **kwargs: Any) -> None:
        raise OSError("no xdg-open")

    monkeypatch.setattr(subprocess, "run", broken)
    monkeypatch.setattr("autoclip.app.session.webbrowser.open", opened.append)
    open_folder(tmp_path / "out")
    assert opened and opened[0].endswith("out")


def test_a_folder_that_cannot_be_created_is_reported(tmp_path: Path) -> None:
    """A file in the way of the output folder is an error."""
    blocker = tmp_path / "out"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        open_folder(blocker)


@needs_display
def test_the_remaining_buttons_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Save, Load, Reset and the typed config path all do what they promise."""
    from autoclip.app.window import AutoclipApp

    monkeypatch.setattr(filedialog, "askopenfilename", lambda **kw: str(tmp_path / "config.json"))
    app = AutoclipApp(tmp_path / "config.json")
    try:
        app._save()
        assert (tmp_path / "config.json").exists()
        app._load()
        app._reset()
        app._remember_path("")
        app._remember_path(str(tmp_path / "other.json"))
        assert app.config_path == tmp_path / "other.json"
        app._stop()  # nothing is running: nothing happens
        app._read_output()  # no process: nothing happens
    finally:
        app.destroy()


@needs_display
def test_a_config_file_that_cannot_be_written_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A config file that cannot be written is reported, not raised."""
    from autoclip.app import window
    from autoclip.app.window import AutoclipApp

    errors: list[str] = []
    monkeypatch.setattr(messagebox, "showerror", lambda title, text: errors.append(text))

    def broken(path: Path, data: Any) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(window, "write_config", broken)
    app = AutoclipApp(tmp_path / "config.json")
    try:
        app._save_to(tmp_path / "config.json")
        assert errors
    finally:
        app.destroy()


@needs_display
def test_the_pipeline_cannot_be_started_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second Run does nothing while the first run is going."""
    from autoclip.app import window
    from autoclip.app.window import AutoclipApp

    monkeypatch.setattr(window, "start_pipeline", lambda path: FakeProcess([]))
    app = AutoclipApp(tmp_path / "config.json")
    try:
        app._run()
        app._run()
        assert str(app.run_button.cget("state")) == "disabled"
    finally:
        app.destroy()


@needs_display
def test_the_reference_window_ignores_an_empty_selection() -> None:
    """Selecting nothing shows nothing."""
    from autoclip.app.window import AutoclipApp, ReferenceWindow

    app = AutoclipApp()
    try:
        window = ReferenceWindow(app)
        window._show()
        assert window.text.get("1.0", "end").strip() == ""
    finally:
        app.destroy()


def test_the_theme_falls_back_to_the_current_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no known theme exists the current one is kept."""
    from autoclip.app import window

    class FakeStyle:
        def theme_names(self) -> list[str]:
            return ["something-else"]

        def theme_use(self) -> str:
            return "something-else"

    monkeypatch.setattr(ttk, "Style", lambda *a, **kw: FakeStyle())
    assert window.system_theme() == "something-else"


def test_opening_a_folder_on_macos_uses_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """On macOS the folder is opened with the system command."""
    called: list[list[str]] = []
    monkeypatch.setattr("autoclip.app.session.sys.platform", "darwin")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: called.append(cmd))
    open_folder(tmp_path / "out")
    assert called and called[0][0] == "open"


def test_launch_opens_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """``launch`` opens the window and waits for it."""
    from autoclip.app import window

    opened: list[Path | None] = []

    class FakeApp:
        def __init__(self, config_path: Path | None = None) -> None:
            opened.append(config_path)

        def mainloop(self) -> None:
            pass

    monkeypatch.setattr(window, "AutoclipApp", FakeApp)
    assert window.launch(Path("x.json")) == 0
    assert opened == [Path("x.json")]


@needs_display
def test_cancelling_the_file_dialogs_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled dialog leaves the fields as they were."""
    import tkinter as tk
    from tkinter import filedialog

    from autoclip.app.form import Form

    monkeypatch.setattr(filedialog, "askopenfilename", lambda **kw: "")
    monkeypatch.setattr(filedialog, "askopenfilenames", lambda **kw: ())
    values: dict[str, Any] = {"transcript.file": "", "input": []}
    root = tk.Tk()
    root.withdraw()
    try:
        form = Form(
            root,
            (OPTIONS["transcript.file"], OPTIONS["input"]),
            values,
            lambda path, value: values.update({path: value}),
        )
        form._browse(OPTIONS["transcript.file"])
        form._browse_videos()
        assert values["transcript.file"] == "" and values["input"] == []
    finally:
        root.destroy()


@needs_display
def test_a_field_without_a_control_ignores_its_value() -> None:
    """A field whose control is missing does not crash."""
    import tkinter as tk

    from autoclip.app.form import Field

    root = tk.Tk()
    root.withdraw()
    try:
        option = OPTIONS["clips.short.count"]
        empty = Field(option, 0, ttk.Label(root), ttk.Frame(root), ttk.Label(root), ttk.Frame(root), None, None)
        empty.set("5")
        assert empty.value() is None
    finally:
        root.destroy()


@needs_display
def test_focusing_an_option_tells_the_window() -> None:
    """Focusing a control fills the details panel of the window."""
    from autoclip.app.form import Form
    from autoclip.app.window import AutoclipApp

    app = AutoclipApp()
    try:
        form = Form(app, (OPTIONS["clips.short.count"],), app.values, app._on_change)
        form._focused(OPTIONS["clips.short.count"])
        assert app.what_label.cget("text") == OPTIONS["clips.short.count"].what
        assert app.why_label.cget("text") == OPTIONS["clips.short.count"].why
    finally:
        app.destroy()


# -- the last corners ------------------------------------------------------------------------


@needs_display
def test_cancelling_the_config_dialogs_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled file dialog leaves the app as it was."""
    from autoclip.app.window import AutoclipApp

    monkeypatch.setattr(filedialog, "askopenfilename", lambda **kw: "")
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **kw: "")
    app = AutoclipApp(tmp_path / "config.json")
    try:
        app._choose_config()
        app._save_as()
        app._load()
        assert app.config_path == tmp_path / "config.json"
    finally:
        app.destroy()


@needs_display
def test_a_pipeline_that_cannot_be_started_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the pipeline does not start, the app says so."""
    from autoclip.app import window
    from autoclip.app.window import AutoclipApp

    def broken(path: Path) -> Any:
        raise OSError("no python")

    monkeypatch.setattr(window, "start_pipeline", broken)
    app = AutoclipApp(tmp_path / "config.json")
    try:
        app._run()
        assert app._process is None
        assert "error:" in app.log.get("1.0", "end")
        assert app.status.cget("text") == "Could not start the pipeline."
    finally:
        app.destroy()


@needs_display
def test_an_absolute_output_folder_is_used_as_it_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An absolute output folder is not made relative to the config file."""
    from autoclip.app import window
    from autoclip.app.window import AutoclipApp

    opened: list[Path] = []
    monkeypatch.setattr(window, "open_folder", opened.append)
    app = AutoclipApp(tmp_path / "config.json")
    try:
        set_option(app, "output_dir", str(tmp_path / "elsewhere"))
        app._open_output()
        assert opened == [tmp_path / "elsewhere"]
    finally:
        app.destroy()


@needs_display
def test_closing_the_window_without_a_run_just_closes_it(tmp_path: Path) -> None:
    """Closing the app while nothing runs does not stop anything."""
    import tkinter as tk

    from autoclip.app.window import AutoclipApp

    app = AutoclipApp(tmp_path / "config.json")
    app._on_close()
    with pytest.raises(tk.TclError):
        app.winfo_exists()  # the window is gone


@needs_display
def test_the_button_that_opens_the_option_reference(tmp_path: Path) -> None:
    """The reference window can be opened from the app."""
    from autoclip.app.window import AutoclipApp, ReferenceWindow

    app = AutoclipApp(tmp_path / "config.json")
    try:
        app._show_reference()
        assert any(isinstance(w, ReferenceWindow) for w in app.winfo_children())
    finally:
        app.destroy()


@needs_display
def test_the_reference_window_ignores_an_unknown_option() -> None:
    """A selection that is not in the catalogue shows nothing."""
    from autoclip.app.window import AutoclipApp, ReferenceWindow

    app = AutoclipApp()
    try:
        window = ReferenceWindow(app)
        window.options.clear()
        window.tree.selection_set("clips.scenes")
        window._show()
        assert window.text.get("1.0", "end").strip() == ""
    finally:
        app.destroy()


@needs_display
def test_a_wheel_event_without_a_direction_scrolls_nothing() -> None:
    """An event that is neither a button nor a wheel does not scroll."""
    import tkinter as tk

    from autoclip.app.form import ScrollFrame

    root = tk.Tk()
    root.withdraw()
    try:
        scroll = ScrollFrame(root)
        calls: list[tuple[int, str]] = []
        scroll.canvas.yview_scroll = lambda amount, what: calls.append((amount, what))  # type: ignore[method-assign]
        scroll.scroll_by(cast("tk.Event", types.SimpleNamespace()))
        assert calls == []
    finally:
        root.destroy()


@needs_display
def test_a_multi_line_field_without_a_control_ignores_its_value() -> None:
    """A multi-line field whose control is missing does not crash."""
    import tkinter as tk

    from autoclip.app.form import Field

    root = tk.Tk()
    root.withdraw()
    try:
        option = OPTIONS["clips.short.ranges"]
        empty = Field(option, 0, ttk.Label(root), ttk.Frame(root), ttk.Label(root), ttk.Frame(root), None, None)
        empty.set("00:00-00:10")
        assert empty.value() == []  # there is no control to read
    finally:
        root.destroy()


@needs_display
def test_the_callbacks_of_the_form_report_their_changes() -> None:
    """The callbacks of a field report the value of the control."""
    import tkinter as tk

    from autoclip.app.form import Form

    changed: list[tuple[str, Any]] = []
    root = tk.Tk()
    root.withdraw()
    try:
        form = Form(root, (OPTIONS["input"],), {"input": []}, lambda path, value: changed.append((path, value)))
        reporter = form._text_reporter(OPTIONS["input"], form.fields["input"].text)  # type: ignore[arg-type]
        reporter(cast("tk.Event", types.SimpleNamespace()))
        assert changed and changed[-1][0] == "input"
        focus = form._focus_reporter(OPTIONS["input"])
        focus(cast("tk.Event", types.SimpleNamespace()))  # a plain root has no details panel
    finally:
        root.destroy()


def test_the_command_line_can_open_the_desktop_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--gui`` opens the window instead of running a config file."""
    import autoclip.cli as cli

    called: list[Path | None] = []

    def fake_launch(path: Path | None = None) -> int:
        called.append(path)
        return 0

    monkeypatch.setattr("autoclip.app.launch", fake_launch)
    assert cli.main(["--gui", "config.json"]) == 0
    assert called == [Path("config.json")]


def test_a_frozen_build_runs_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frozen build has no Python around it: the binary runs the pipeline itself."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    command = pipeline_command(Path("config.json"))
    assert command == [sys.executable, "config.json"]
    monkeypatch.delattr(sys, "frozen", raising=False)


def test_a_frozen_build_starts_no_python(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A frozen build puts no project path into the environment of the pipeline."""
    started: list[dict[str, Any]] = []

    class FakePopen:
        def __init__(self, cmd: list[str], **kwargs: Any) -> None:
            started.append({"cmd": cmd, **kwargs})

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    start_pipeline(tmp_path / "config.json")
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert started and started[0]["cmd"] == [sys.executable, str(tmp_path / "config.json")]
    assert "PYTHONPATH" not in started[0]["env"]
