"""Building the form: one row per option, with the control that fits its kind.

The form is generated from :mod:`autoclip.app.fields`, so the app and the config file can never
drift apart. Every row shows a one-line summary under the control; the full explanation
(what / why / when) appears in the details panel of the window when the option is focused.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import filedialog, ttk

from ..core.config import JsonValue
from .fields import Option
from .session import split_csv

try:
    from tkinterdnd2 import DND_FILES as _DND_FILES
except Exception:  # pragma: no cover - depends on the installed extras
    _DND_FILES = None

#: A callback that receives the changed option path and its new value.
Changed = Callable[[str, object], None]

__all__ = ["Changed", "Field", "Form", "ScrollFrame", "split_csv", "variable_value"]


def variable_value(var: tk.Variable) -> object:
    """The value of a Tk variable (the type stubs do not describe ``Variable.get``)."""
    return var.get()  # type: ignore[no-untyped-call]


class ScrollFrame(ttk.Frame):
    """A frame that scrolls its content vertically (with the mouse wheel as well)."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        for widget in (self.canvas, self.inner):
            widget.bind("<MouseWheel>", self.scroll_by)
            widget.bind("<Button-4>", self.scroll_by)
            widget.bind("<Button-5>", self.scroll_by)

    def _on_inner(self, _event: tk.Event) -> None:
        """Keep the scroll area as tall as its content."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event: tk.Event) -> None:
        """Let the content use the whole width."""
        self.canvas.itemconfigure(self._window, width=event.width)

    def scroll_by(self, event: tk.Event) -> None:
        """Scroll with the mouse wheel (the direction differs per platform)."""
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(3, "units")
        elif getattr(event, "delta", 0):
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")


@dataclass
class Field:
    """One row of the form: the option, its control and the widget that holds it."""

    option: Option
    row: int
    label: ttk.Label
    control: tk.Widget
    help_label: ttk.Label
    holder: ttk.Frame
    var: tk.Variable | None = None  # the variable behind the control
    text: tk.Text | None = None  # the text widget of a multi-line option

    def value(self) -> object:
        """The current value of the control, in the type the config expects."""
        kind = self.option.kind
        if kind == "csv":
            text = str(variable_value(self.var)) if self.var else ""
            if self.option.csv_is_list:
                return split_csv(text)
            return split_csv(text) if "," in text else text.strip()
        if kind == "lines":
            return [line.strip() for line in self.text.get("1.0", "end").splitlines() if line.strip()] if self.text else []
        text = str(variable_value(self.var)).strip() if self.var else ""
        if not text:
            return None if self.option.empty_is_none else ""  # an empty text field means "not set", unless it is a plain string
        return text

    def set(self, value: JsonValue | str | bool | int | float | None) -> None:
        """Put a value into the control."""
        kind = self.option.kind
        if kind == "lines":
            if self.text is None:
                return
            self.text.delete("1.0", "end")
            self.text.insert(
                "1.0",
                value if isinstance(value, str) else "\n".join(str(v) for v in (value if isinstance(value, (list, tuple)) else [])),
            )
            return
        if self.var is None:
            return
        if kind == "csv":
            if isinstance(value, str):
                self.var.set(value)
            else:
                self.var.set(", ".join(str(v) for v in (value if isinstance(value, (list, tuple)) else [])))
        elif value is None:
            self.var.set("")
        else:
            self.var.set(str(value))


class Form(ttk.Frame):
    """The form of one section: a row per option."""

    def __init__(
        self,
        master: tk.Misc,
        options: tuple[Option, ...],
        values: dict[str, JsonValue],
        changed: Changed,
        wheel: Callable[[tk.Event], None] | None = None,
    ) -> None:
        super().__init__(master, padding=(12, 10, 12, 12))
        self.fields: dict[str, Field] = {}
        self._values = values
        self._changed = changed
        self._wheel = wheel
        self.columnconfigure(1, weight=1)
        row = 0
        for option in options:
            field = self._build_row(row, option)
            self.fields[option.path] = field
            row += 2

    def _build_row(self, row: int, option: Option, master: tk.Misc | None = None) -> Field:
        """Label, control and the one-line summary of one option."""
        parent = master if master is not None else self
        label = ttk.Label(parent, text=option.label, width=26, anchor="w")
        label.grid(row=row, column=0, sticky="nw", padx=(0, 12), pady=(2, 2))
        holder = ttk.Frame(parent)
        holder.grid(row=row, column=1, sticky="ew", pady=(2, 2))
        control, var, text = self._build_control(option, holder)
        control.grid(row=0, column=0, sticky="ew")
        if option.kind == "lines" and option.path == "input":
            ttk.Button(holder, text="Add videos...", command=self._browse_videos).grid(row=0, column=1, sticky="e", padx=(6, 0))
            self._enable_drag_and_drop(holder)
        holder.columnconfigure(0, weight=1)
        help_label = ttk.Label(parent, text=option.what, style="Help.TLabel", wraplength=620, justify="left")
        help_label.grid(row=row + 1, column=1, sticky="w", pady=(0, 6))
        for widget in (label, control, help_label):
            widget.bind("<FocusIn>", self._focus_reporter(option))
            if self._wheel is not None:
                widget.bind("<MouseWheel>", self._wheel)
                widget.bind("<Button-4>", self._wheel)
                widget.bind("<Button-5>", self._wheel)
        return Field(option, row, label, control, help_label, holder, var, text)

    def _build_control(self, option: Option, holder: ttk.Frame) -> tuple[tk.Widget, tk.Variable | None, tk.Text | None]:
        """The widget that fits the kind of the option, and the variable behind it."""
        kind = option.kind
        if kind == "choice":
            choice_var = tk.StringVar(value=str(self._values.get(option.path, option.choices[0] if option.choices else "")))
            box = ttk.Combobox(holder, textvariable=choice_var, values=option.choices, state="readonly", width=28)
            choice_var.trace_add("write", self._reporter(option, choice_var))
            return box, choice_var, None
        if kind == "lines":
            text = tk.Text(holder, height=3, width=52, wrap="none", font="TkFixedFont")
            text.bind("<KeyRelease>", self._text_reporter(option, text))
            return text, None, text
        text_var = tk.StringVar(value=self._as_text(option, self._values.get(option.path)))
        entry = ttk.Entry(holder, width=32, textvariable=text_var)
        text_var.trace_add("write", self._reporter(option, text_var))
        return entry, text_var, None

    def _reporter(self, option: Option, var: tk.Variable) -> Callable[..., None]:
        """A callback that reports a changed value to the window."""

        def report(*_args: object) -> None:
            field = self.fields.get(option.path)
            self._changed(option.path, field.value() if field else variable_value(var))

        return report

    def _text_reporter(self, option: Option, text: tk.Text) -> Callable[[tk.Event], None]:
        """A callback that reports the new content of a multi-line field."""

        def report(_event: tk.Event) -> None:
            field = self.fields.get(option.path)
            self._changed(option.path, field.value() if field else text.get("1.0", "end"))

        return report

    def _as_text(self, option: Option, value: object) -> str:
        """The text an entry should show for a value."""
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    def _browse_videos(self) -> None:
        """Add video files to the input list."""
        picked = filedialog.askopenfilenames(
            title="Choose the videos to process",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"), ("All files", "*.*")],
        )
        if not picked:
            return
        self._add_videos([str(p) for p in picked])

    def _enable_drag_and_drop(self, holder: ttk.Frame) -> None:
        """Accept dropped video files when the platform supports it (no-op otherwise)."""
        register = getattr(holder, "drop_target_register", None)
        bind = getattr(holder, "dnd_bind", None)
        if not callable(register) or not callable(bind) or _DND_FILES is None:
            return
        try:
            register(_DND_FILES)
        except Exception:
            return
        bind("<<Drop>>", self._on_drop_videos)

    def _on_drop_videos(self, event: tk.Event) -> str | None:
        """Add the files from a drag-and-drop operation to the input list (de-duplicated)."""
        data = getattr(event, "data", "")
        try:
            files = tuple(self.splitlist(data))  # type: ignore[attr-defined]
        except Exception:
            files = tuple(str(data).split())
        self._add_videos([str(path) for path in files])
        field = self.fields.get("input")
        if field is not None and field.text is not None:
            field.text.focus_set()
        return "break"

    def _add_videos(self, paths: list[str]) -> None:
        """Merge new video paths into the input list without duplicates."""
        field = self.fields.get("input")
        current = list(self._values.get("input") or [])  # type: ignore[arg-type]
        merged = list(current)
        changed = False
        for path in paths:
            if path and path not in merged:
                merged.append(path)
                changed = True
        if not changed:
            return
        self._values["input"] = merged
        if field is not None:
            field.set(merged)
        self._changed("input", merged)

    def _focused(self, option: Option) -> None:
        """Tell the window which option the user is looking at."""
        parent = self.winfo_toplevel()
        show = getattr(parent, "show_details", None)
        if callable(show):
            show(option)

    def _focus_reporter(self, option: Option) -> Callable[[tk.Event], None]:
        """A callback that tells the window which option the user is looking at."""

        def report(_event: tk.Event) -> None:
            self._focused(option)

        return report
