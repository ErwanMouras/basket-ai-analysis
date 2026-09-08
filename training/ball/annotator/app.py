"""A single Tk window: video canvas, controls and autosaved annotations."""

from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .model import Annotation, Store, Viewport
from .video import VideoReader


class App:
    def __init__(self, root: tk.Tk, reader: VideoReader, store: Store) -> None:
        self.root, self.reader, self.store = root, reader, store
        self.frame_idx = 0
        self.viewport = None
        self.photo = None
        self._redraw_pending = None
        self._syncing_slider = False
        self.info = tk.StringVar(root)
        self.status = tk.StringVar(
            root,
            value="Click the centre of the ball. Changes are saved automatically.",
        )
        root.title(f"Ball Annotator — {Path(reader.meta.path).name}")
        root.geometry(
            f"{min(1200, root.winfo_screenwidth() - 80)}"
            f"x{min(820, root.winfo_screenheight() - 100)}"
        )
        root.minsize(720, 480)
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(fill="x")
        buttons = [
            ("Previous", lambda: self.goto(self.frame_idx - 1)),
            ("Next", lambda: self.goto(self.frame_idx + 1)),
            ("Next unannotated", self.next_unannotated),
            ("Radius −", lambda: self.adjust_radius(-1)),
            ("Radius +", lambda: self.adjust_radius(1)),
        ]
        for label, command in buttons:
            ttk.Button(
                toolbar,
                text=label,
                command=lambda fn=command: self.perform(fn),
            ).pack(side="left", padx=2)
        ttk.Label(root, textvariable=self.info, padding=(10, 4)).pack(fill="x")

        self.canvas = tk.Canvas(root, background="#171b23", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.schedule_redraw)
        self.canvas.bind(
            "<Button-1>",
            lambda event: self.perform(lambda: self.place(event.x, event.y)),
        )
        self.canvas.bind("<Motion>", self.preview)
        self.canvas.bind("<Leave>", lambda _: self.canvas.delete("cursor"))

        self.slider = tk.Scale(
            root,
            from_=0,
            to=reader.playable_frame_count - 1,
            orient="horizontal",
            showvalue=False,
            command=self.seek,
        )
        self.slider.pack(fill="x", padx=10)
        actions = ttk.Frame(root, padding=(8, 2))
        actions.pack(fill="x")
        for label, command in [
            ("Toggle visibility (V)", self.toggle_occlusion),
            ("Mark unknown (H)", self.mark_unknown),
            ("Copy previous (C)", self.copy_previous),
            ("Clear (X)", self.delete),
        ]:
            ttk.Button(
                actions,
                text=label,
                command=lambda fn=command: self.perform(fn),
            ).pack(side="left", padx=2)
        ttk.Label(root, textvariable=self.status, padding=(10, 5), wraplength=680).pack(
            fill="x"
        )
        ttk.Label(
            root,
            text="Left / Right or b / n: +/-1 frame | Shift+B / N: +/-10 "
            "| g / G: first / last\n"
            "Up / Down: radius | f: next unannotated | q / Esc: quit",
            padding=(10, 4),
        ).pack(fill="x")
        root.bind("<KeyPress>", self.on_key)
        resume = next(
            (i for i in range(reader.playable_frame_count) if i not in store.annotations),
            0,
        )
        self.goto(resume)

    def perform(self, action) -> None:
        try:
            action()
        except (OSError, ValueError, cv2.error) as exc:
            self.status.set(str(exc))
            self.sync_slider()
            messagebox.showerror("Annotation error", str(exc), parent=self.root)

    def sync_slider(self) -> None:
        self._syncing_slider = True
        try:
            self.slider.set(self.frame_idx)
        finally:
            self._syncing_slider = False

    def seek(self, value: str) -> None:
        if not self._syncing_slider and int(value) != self.frame_idx:
            self.perform(lambda: self.goto(int(value)))

    def goto(self, frame_idx: int) -> None:
        frame_idx = max(0, min(self.reader.playable_frame_count - 1, frame_idx))
        # Decode before changing the active index: a failed seek cannot mislabel
        # the last valid image as a different frame.
        frame = self.reader.read(frame_idx)
        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.frame_idx, self.frame_image = frame_idx, rgb
        self.sync_slider()
        self.redraw()

    def next_unannotated(self) -> None:
        target = next(
            (
                i
                for i in range(self.frame_idx + 1, self.reader.playable_frame_count)
                if i not in self.store.annotations
            ),
            None,
        )
        if target is None:
            self.status.set("There are no unannotated frames after this one.")
        else:
            self.goto(target)

    def edited(self) -> None:
        self.status.set(
            f"Frame {self.frame_idx + 1} saved · {self.store.path.name}"
        )
        self.redraw()

    def place(self, x: float, y: float) -> None:
        coords = self.viewport.to_source(x, y) if self.viewport else None
        if coords is None:
            return
        self.store.update(
            self.frame_idx, Annotation(*coords, self.store.default_radius)
        )
        self.edited()

    def toggle_occlusion(self) -> None:
        ann = self.store.annotations.get(self.frame_idx)
        if ann is None:
            self.store.update(self.frame_idx, Annotation(occluded=True))
        elif not ann.has_position:
            self.status.set(
                "Place a centre point to mark the ball visible, or press X to clear it."
            )
            return
        else:
            self.store.update(self.frame_idx, replace(ann, occluded=not ann.occluded))
        self.edited()

    def mark_unknown(self) -> None:
        self.store.update(self.frame_idx, Annotation(occluded=True))
        self.edited()

    def copy_previous(self) -> None:
        ann = self.store.annotations.get(self.frame_idx - 1)
        if ann is None:
            self.status.set("The immediately previous frame is not annotated.")
            return
        self.store.update(self.frame_idx, ann)
        self.edited()

    def delete(self) -> None:
        self.store.update(self.frame_idx, None)
        self.edited()

    def adjust_radius(self, delta: int) -> None:
        self.store.adjust_radius(self.frame_idx, delta)
        self.edited()

    def schedule_redraw(self, _event=None) -> None:
        if self._redraw_pending is None:
            self._redraw_pending = self.root.after_idle(self.redraw)

    def redraw(self) -> None:
        if self._redraw_pending is not None:
            self.root.after_cancel(self._redraw_pending)
            self._redraw_pending = None
        meta = self.reader.meta
        self.viewport = Viewport.fit(
            meta.width,
            meta.height,
            max(1, self.canvas.winfo_width()),
            max(1, self.canvas.winfo_height()),
        )
        vp = self.viewport
        displayed = self.frame_image.resize(
            (vp.width, vp.height), Image.Resampling.LANCZOS
        )
        self.photo = ImageTk.PhotoImage(displayed, master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(vp.left, vp.top, image=self.photo, anchor="nw")
        ann = self.store.annotations.get(self.frame_idx)
        if ann is None:
            state = "Unannotated"
        elif ann.has_position:
            state = "Occluded, estimated position" if ann.occluded else "Visible"
            self.draw_marker(ann.cx, ann.cy, ann.radius, "#ffc857", dashed=ann.occluded)
        else:
            state = "Occluded / out of frame, position unknown"
        radius = ann.radius if ann and ann.has_position else self.store.default_radius
        self.info.set(
            f"Frame {self.frame_idx + 1} / {self.reader.playable_frame_count} | {state} "
            f"| {len(self.store.annotations)} annotated\n"
            f"Radius {radius:g} px source | display {vp.width}x{vp.height} "
            f"/ source {meta.width}x{meta.height}"
        )

    def draw_marker(
        self,
        cx: float,
        cy: float,
        radius: float,
        color: str,
        *,
        dashed: bool = False,
        tag: str = "annotation",
    ) -> None:
        vp = self.viewport
        x, y = vp.to_display(cx, cy)
        rx, ry = (
            radius * vp.width / vp.source_width,
            radius * vp.height / vp.source_height,
        )
        self.canvas.create_oval(
            x - rx,
            y - ry,
            x + rx,
            y + ry,
            outline=color,
            width=2,
            dash=(4, 3) if dashed else (),
            tags=tag,
        )
        self.canvas.create_line(x - 4, y, x + 4, y, fill=color, tags=tag)
        self.canvas.create_line(x, y - 4, x, y + 4, fill=color, tags=tag)

    def preview(self, event) -> None:
        self.canvas.delete("cursor")
        coords = self.viewport.to_source(event.x, event.y) if self.viewport else None
        if coords is not None:
            self.draw_marker(
                *coords, self.store.default_radius, "#72d6ed", dashed=True, tag="cursor"
            )

    def on_key(self, event):
        key = event.keysym
        char = event.char
        if key == "Escape" or char == "q":
            self.root.destroy()
            return "break"
        navigation = {"Left": -1, "Right": 1, "b": -1, "n": 1, "B": -10, "N": 10}
        if key in navigation:
            self.perform(lambda: self.goto(self.frame_idx + navigation[key]))
        elif key in ("Up", "Down"):
            self.perform(lambda: self.adjust_radius(1 if key == "Up" else -1))
        else:
            commands = {
                "v": self.toggle_occlusion,
                "h": self.mark_unknown,
                "c": self.copy_previous,
                "x": self.delete,
                "f": self.next_unannotated,
                "g": lambda: self.goto(0),
                "G": lambda: self.goto(self.reader.playable_frame_count - 1),
            }
            if char not in commands:
                return None
            self.perform(commands[char])
        return "break"


def run(video: Path) -> None:
    reader = VideoReader(video)
    root = None
    try:
        store = Store(reader.meta)
        root = tk.Tk()
        App(root, reader, store)
        root.mainloop()
    except tk.TclError as exc:
        raise RuntimeError(f"The Tk window could not start: {exc}") from exc
    finally:
        reader.close()
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass
