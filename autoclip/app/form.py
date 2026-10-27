"""Building the form: one row per option, with the control that fits its kind.

The form is generated from :mod:`autoclip.app.fields`, so the app and the config file can never
drift apart. Every row shows a one-line summary under the control; the full explanation
(what / why / when) appears in the details panel of the window when the option is focused.
"""

from __future__ import annotations

import json
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

from .fields import Option
from .session import flatten, split_csv, unflatten

#: A callback that receives the changed option path and its new value.
Changed = Callable[[str, Any], None]

__all__ = ["Changed", "Field", "Form", "ScrollFrame", "flatten", "split_csv", "unflatten", "variable_value"]


def variable_value(var: tk.Variable) -> Any:
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

    def value(self) -> Any:
        """The current value of the control, in the type the config expects."""
        kind = self.option.kind
        if kind == "bool":
            return bool(variable_value(self.var)) if self.var else False
        if kind in ("int", "float"):
            text = str(variable_value(self.var)).strip() if self.var else ""
            if not text:
                return None
            try:
                return int(text) if kind == "int" else float(text)
            except ValueError:
                return None
        if kind == "csv":
            text = str(variable_value(self.var)) if self.var else ""
            if self.option.csv_is_list:
                return split_csv(text)
            return split_csv(text) if "," in text else text.strip()
        if kind == "lines":
            return [line.strip() for line in self.text.get("1.0", "end").splitlines() if line.strip()] if self.text else []
        if kind == "json":
            return self._json_value()
        text = str(variable_value(self.var)).strip() if self.var else ""
        if not text:
            return None  # an empty text field means "not set"; every text option is nullable or required anyway
        if self.option.numeric:
            try:
                return int(text)
            except ValueError:
                try:
                    return float(text)
                except ValueError:
                    return text
        return text

    def _json_value(self) -> Any:
        """The object typed into a JSON field, or the text itself when it does not parse."""
        text = self.text.get("1.0", "end").strip() if self.text else ""
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text  # the config check reports it with the option path, which a silent {} would hide

    def set(self, value: Any) -> None:
        """Put a value into the control."""
        kind = self.option.kind
        if kind in ("lines", "json"):
            if self.text is None:
                return
            self.text.delete("1.0", "end")
            if kind == "json":
                text = json.dumps(value, indent=2, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value or "{}")
                self.text.insert("1.0", text)
            else:
                self.text.insert("1.0", value if isinstance(value, str) else "\n".join(str(v) for v in value or []))
            return
        if self.var is None:
            return
        if kind == "csv":
            if isinstance(value, str):
                self.var.set(value)
            else:
                self.var.set(", ".join(str(v) for v in value or []))
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
        values: dict[str, Any],
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
        self._apply_visibility()

    def _build_row(self, row: int, option: Option) -> Field:
        """Label, control and the one-line summary of one option."""
        label = ttk.Label(self, text=option.label, width=26, anchor="w")
        label.grid(row=row, column=0, sticky="nw", padx=(0, 12), pady=(2, 2))
        holder = ttk.Frame(self)
        holder.grid(row=row, column=1, sticky="ew", pady=(2, 2))
        control, var, text = self._build_control(option, holder)
        control.grid(row=0, column=0, sticky="ew")
        if option.kind in ("path", "dir"):
            ttk.Button(holder, text="Browse...", command=partial(self._browse, option)).grid(row=0, column=1, sticky="e", padx=(6, 0))
        if option.kind == "lines" and option.path == "input":
            ttk.Button(holder, text="Add videos...", command=self._browse_videos).grid(row=0, column=1, sticky="e", padx=(6, 0))
        holder.columnconfigure(0, weight=1)
        help_label = ttk.Label(self, text=option.what, style="Help.TLabel", wraplength=620, justify="left")
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
        if kind == "bool":
            bool_var = tk.BooleanVar(value=bool(self._values.get(option.path, False)))
            control: tk.Widget = ttk.Checkbutton(holder, text="enabled", variable=bool_var)
            bool_var.trace_add("write", self._reporter(option, bool_var))
            return control, bool_var, None
        if kind == "choice":
            choice_var = tk.StringVar(value=str(self._values.get(option.path, option.choices[0] if option.choices else "")))
            box = ttk.Combobox(holder, textvariable=choice_var, values=option.choices, state="readonly", width=28)
            choice_var.trace_add("write", self._reporter(option, choice_var))
            return box, choice_var, None
        if kind == "lines":
            text = tk.Text(holder, height=3, width=52, wrap="none", font="TkFixedFont")
            text.bind("<KeyRelease>", self._text_reporter(option, text))
            return text, None, text
        if kind == "json":
            json_text = tk.Text(holder, height=6, width=52, wrap="none", font="TkFixedFont")
            json_text.bind("<KeyRelease>", self._text_reporter(option, json_text))
            return json_text, None, json_text
        text_var = tk.StringVar(value=self._as_text(option, self._values.get(option.path)))
        entry = ttk.Entry(holder, width=32, textvariable=text_var)
        text_var.trace_add("write", self._reporter(option, text_var))
        return entry, text_var, None

    def _reporter(self, option: Option, var: tk.Variable) -> Callable[..., None]:
        """A callback that reports a changed value to the window."""

        def report(*_args: Any) -> None:
            field = self.fields.get(option.path)
            self._changed(option.path, field.value() if field else variable_value(var))

        return report

    def _text_reporter(self, option: Option, text: tk.Text) -> Callable[[tk.Event], None]:
        """A callback that reports the new content of a multi-line field."""

        def report(_event: tk.Event) -> None:
            field = self.fields.get(option.path)
            self._changed(option.path, field.value() if field else text.get("1.0", "end"))

        return report

    def _as_text(self, option: Option, value: Any) -> str:
        """The text an entry should show for a value."""
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    def _browse(self, option: Option) -> None:
        """Pick a file or a folder for a path option."""
        current = str(self._values.get(option.path) or "")
        if option.kind == "dir":
            picked = filedialog.askdirectory(title=f"Choose a folder for {option.label}", initialdir=current or str(Path.home()))
        else:
            picked = filedialog.askopenfilename(
                title=f"Choose a file for {option.label}", initialdir=str(Path(current).parent) if current else ""
            )
        if picked:
            self._values[option.path] = picked
            self.fields[option.path].set(picked)
            self._changed(option.path, picked)

    def _browse_videos(self) -> None:
        """Add video files to the input list."""
        picked = filedialog.askopenfilenames(
            title="Choose the videos to process",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"), ("All files", "*.*")],
        )
        if not picked:
            return
        current = self._values.get("input") or []
        merged = list(current) + [str(p) for p in picked]
        self._values["input"] = merged
        self.fields["input"].set(merged)
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

    def _apply_visibility(self) -> None:
        """Hide the options whose condition is not met (for example the ranges in highlights mode)."""
        for field in self.fields.values():
            option = field.option
            visible = True
            if option.visible_when:
                other, _, wanted = option.visible_when.partition("=")
                visible = str(self._values.get(other, "")) == wanted
            if visible:
                field.holder.grid()
                field.help_label.grid()
            else:
                field.holder.grid_remove()
                field.help_label.grid_remove()

    def refresh_visibility(self) -> None:
        """Re-evaluate which options are shown (after a value changed)."""
        self._apply_visibility()
