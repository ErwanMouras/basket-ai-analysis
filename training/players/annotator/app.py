"""Tk editor for multiple source-space player boxes, images and sampled videos."""

import tkinter as tk
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .media import IMAGE_EXTENSIONS, MediaReader, selected_frames
from .model import Store, sidecar_path


@dataclass
class Transform:
    scale: float
    left: float
    top: float
    width: int
    height: int

    def source(self, x, y, *, clamp=False):
        x, y = (x - self.left) / self.scale, (y - self.top) / self.scale
        if not clamp and not (0 <= x <= self.width and 0 <= y <= self.height):
            return None
        return [max(0.0, min(self.width, x)), max(0.0, min(self.height, y))]

    def display(self, bbox):
        x1, y1, x2, y2 = bbox
        return [
            self.left + x1 * self.scale,
            self.top + y1 * self.scale,
            self.left + x2 * self.scale,
            self.top + y2 * self.scale,
        ]


def drag_box(bbox, start, end, width, height, corner=None):
    """Move without changing size, or resize one corner inside image bounds."""
    x1, y1, x2, y2 = bbox
    if corner is None:
        dx = max(-x1, min(width - x2, end[0] - start[0]))
        dy = max(-y1, min(height - y2, end[1] - start[1]))
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
    values = list(bbox)
    x_index, y_index = ((0, 1), (2, 1), (2, 3), (0, 3))[corner]
    values[x_index], values[y_index] = end
    return [
        min(values[0], values[2]),
        min(values[1], values[3]),
        max(values[0], values[2]),
        max(values[1], values[3]),
    ]


class App:
    def __init__(
        self, root, paths, source_root, *, step=30, start=0, stop=None, **identity
    ):
        self.root, self.paths, self.source_root = root, paths, source_root
        self.step, self.start, self.stop, self.identity = step, start, stop, identity
        self.reader = None
        self.store = None
        self.media_index = -1
        self.index = 0
        self.transform = None
        self.selected = None
        self.gesture = None
        self.image = None
        self.photo = None
        self.zoom = 1.0
        self.positions = []
        self.syncing = False
        self.info = tk.StringVar(root)
        self.status = tk.StringVar(
            root,
            value="Dessinez une boîte. Vérifiez tous les joueurs avant de valider la frame.",
        )
        self.new_box = tk.BooleanVar(root, value=False)
        root.title("Annotation joueurs")
        root.geometry("1200x850")
        root.minsize(850, 550)
        root.protocol("WM_DELETE_WINDOW", self.close)
        toolbar = ttk.Frame(root, padding=6)
        toolbar.pack(fill="x")
        for label, action in (
            ("Média précédent", lambda: self.open_media(self.media_index - 1)),
            ("Média suivant", lambda: self.open_media(self.media_index + 1)),
            ("Frame précédente ←", lambda: self.navigate(-1)),
            ("Frame suivante →", lambda: self.navigate(1)),
            ("À vérifier [F]", self.next_unverified),
        ):
            ttk.Button(
                toolbar, text=label, command=lambda a=action: self.perform(a)
            ).pack(side="left", padx=2)
        ttk.Label(root, textvariable=self.info, padding=6).pack(fill="x")
        panel = ttk.Frame(root)
        panel.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(panel, background="#18202a", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)
        for orient, row, col, command, sticky in (
            ("horizontal", 1, 0, self.canvas.xview, "ew"),
            ("vertical", 0, 1, self.canvas.yview, "ns"),
        ):
            bar = ttk.Scrollbar(panel, orient=orient, command=command)
            bar.grid(row=row, column=col, sticky=sticky)
            self.canvas.configure(
                **{f"{'x' if orient == 'horizontal' else 'y'}scrollcommand": bar.set}
            )
        self.canvas.bind("<Configure>", lambda _: self.redraw())
        self.canvas.bind(
            "<ButtonPress-1>", lambda e: self.perform(lambda: self.press(e.x, e.y))
        )
        self.canvas.bind(
            "<B1-Motion>", lambda e: self.perform(lambda: self.motion(e.x, e.y))
        )
        self.canvas.bind(
            "<ButtonRelease-1>", lambda e: self.perform(lambda: self.release(e.x, e.y))
        )
        self.slider = tk.Scale(
            root, from_=0, to=0, orient="horizontal", showvalue=False, command=self.seek
        )
        self.slider.pack(fill="x", padx=8)
        actions = ttk.Frame(root, padding=6)
        actions.pack(fill="x")
        ttk.Checkbutton(actions, text="Nouvelle boîte [N]", variable=self.new_box).pack(
            side="left"
        )
        for label, action in (
            ("Supprimer [Suppr]", self.delete),
            ("Occultation [O]", lambda: self.toggle("occluded")),
            ("Troncature [T]", lambda: self.toggle("truncated")),
            ("Valider la frame [V]", self.verify),
            ("Réouvrir [R]", self.reopen),
            ("Zoom +", lambda: self.set_zoom(self.zoom * 1.5)),
            ("Zoom −", lambda: self.set_zoom(self.zoom / 1.5)),
        ):
            ttk.Button(
                actions, text=label, command=lambda a=action: self.perform(a)
            ).pack(side="left", padx=1)
        ttk.Label(root, textvariable=self.status, padding=6, wraplength=1150).pack(
            fill="x"
        )
        ttk.Label(
            root,
            text="Glisser dans une boîte : déplacer | Glisser un coin : redimensionner | "
            "Validation : tous les joueurs, y compris une frame vide | Sauvegarde automatique",
            padding=6,
        ).pack(fill="x")
        root.bind("<KeyPress>", self.key)
        self.open_media(0)

    def perform(self, action):
        try:
            action()
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            self.gesture = None
            self.status.set(str(exc))
            self.redraw()
            messagebox.showerror("Annotation joueurs", str(exc), parent=self.root)

    def close(self):
        if self.reader:
            self.reader.close()
        self.root.destroy()

    def open_media(self, position):
        if not 0 <= position < len(self.paths):
            return
        reader = MediaReader(self.paths[position])
        try:
            store = Store(reader, self.source_root, **self.identity)
            positions = sorted(
                set(
                    selected_frames(
                        reader.frame_count,
                        start=self.start,
                        stop=self.stop,
                        step=self.step,
                    )
                )
                | {f["frame_index"] for f in store.document["frames"]}
            )
            index = next(
                (i for i in positions if store.frame(i)["review_status"] != "verified"),
                positions[0],
            )
            image = Image.fromarray(cv2.cvtColor(reader.read(index), cv2.COLOR_BGR2RGB))
        except BaseException:
            reader.close()
            raise
        if self.reader:
            self.reader.close()
        self.reader, self.store, self.positions = reader, store, positions
        self.media_index, self.index, self.image = position, index, image
        self.selected, self.gesture = None, None
        self.syncing = True
        self.slider.configure(to=len(positions) - 1)
        self.slider.set(positions.index(index))
        self.syncing = False
        self.redraw()

    def goto(self, index):
        if index not in self.positions:
            raise ValueError("Frame not in selected sample")
        image = Image.fromarray(
            cv2.cvtColor(self.reader.read(index), cv2.COLOR_BGR2RGB)
        )
        self.index, self.image, self.selected, self.gesture = index, image, None, None
        self.syncing = True
        self.slider.set(self.positions.index(index))
        self.syncing = False
        self.redraw()

    def navigate(self, delta):
        target = self.positions.index(self.index) + delta
        if 0 <= target < len(self.positions):
            self.goto(self.positions[target])

    def seek(self, value):
        if not self.syncing and self.positions:
            self.perform(lambda: self.goto(self.positions[int(value)]))

    def next_unverified(self):
        ordered = (
            self.positions[self.positions.index(self.index) + 1 :]
            + self.positions[: self.positions.index(self.index)]
        )
        target = next(
            (i for i in ordered if self.store.frame(i)["review_status"] != "verified"),
            None,
        )
        if target is None:
            self.status.set(
                "Toutes les autres frames sélectionnées de ce média sont vérifiées."
            )
        else:
            self.goto(target)

    def set_zoom(self, zoom):
        self.zoom, self.gesture = max(1.0, min(6.0, zoom)), None
        self.redraw()

    def redraw(self):
        if self.image is None:
            return
        width, height = self.image.size
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(cw / width, ch / height) * self.zoom
        dw, dh = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = max(0, (cw - dw) / 2), max(0, (ch - dh) / 2)
        self.transform = Transform(scale, left, top, width, height)
        self.canvas.delete("all")
        self.photo = ImageTk.PhotoImage(self.image.resize((dw, dh)), master=self.root)
        self.canvas.create_image(left, top, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, max(cw, dw), max(ch, dh)))
        frame = self.store.frame(self.index)
        for box in frame["boxes"]:
            color = "#39dc87" if frame["review_status"] == "verified" else "#f4b942"
            selected = box["object_id"] == self.selected
            coords = self.transform.display(box["bbox"])
            self.canvas.create_rectangle(
                *coords, outline="#ffffff" if selected else color, width=2
            )
            self.canvas.create_text(
                coords[0] + 3, coords[1] + 3, text="joueur", fill=color, anchor="nw"
            )
            if selected:
                x1, y1, x2, y2 = coords
                for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                    self.canvas.create_rectangle(
                        x - 4, y - 4, x + 4, y + 4, fill="white"
                    )
        selected = next(
            (b for b in frame["boxes"] if b["object_id"] == self.selected), None
        )
        visibility = {None: "inconnue", True: "oui", False: "non"}
        states = {
            "unannotated": "Non annotée",
            "proposed": "Propositions à vérifier",
            "in_progress": "Correction en cours",
            "verified": "Vérifiée",
        }
        detail = (
            ""
            if not selected
            else f" | occultation : {visibility[selected['occluded']]} | troncature : {visibility[selected['truncated']]}"
        )
        self.info.set(
            f"{self.media_index + 1}/{len(self.paths)} : {self.reader.path.name} | "
            f"frame {self.index}/{self.reader.frame_count - 1} | {states[frame['review_status']]} | "
            f"{len(frame['boxes'])} joueur(s){detail}"
        )

    def point(self, x, y, *, clamp=False):
        return self.transform.source(
            self.canvas.canvasx(x), self.canvas.canvasy(y), clamp=clamp
        )

    def press(self, x, y):
        start = self.point(x, y)
        if start is None:
            return
        frame = self.store.frame(self.index)
        selected = next(
            (b for b in frame["boxes"] if b["object_id"] == self.selected), None
        )
        self.gesture = None
        if not self.new_box.get() and selected:
            x1, y1, x2, y2 = selected["bbox"]
            for corner, (cx, cy) in enumerate(((x1, y1), (x2, y1), (x2, y2), (x1, y2))):
                if (
                    max(abs(start[0] - cx), abs(start[1] - cy)) * self.transform.scale
                    <= 8
                ):
                    self.gesture = {
                        "start": start,
                        "box": deepcopy(selected),
                        "corner": corner,
                    }
                    return
        hit = None
        if not self.new_box.get():
            hits = [
                b
                for b in frame["boxes"]
                if b["bbox"][0] <= start[0] <= b["bbox"][2]
                and b["bbox"][1] <= start[1] <= b["bbox"][3]
            ]
            hit = min(
                hits,
                key=lambda b: (
                    (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
                ),
                default=None,
            )
        self.selected = hit["object_id"] if hit else None
        self.gesture = {"start": start, "box": deepcopy(hit), "corner": None}
        self.redraw()

    def preview_bbox(self, x, y):
        end, gesture = self.point(x, y, clamp=True), self.gesture
        if gesture["box"]:
            return drag_box(
                gesture["box"]["bbox"],
                gesture["start"],
                end,
                self.reader.width,
                self.reader.height,
                gesture["corner"],
            )
        start = gesture["start"]
        return [
            min(start[0], end[0]),
            min(start[1], end[1]),
            max(start[0], end[0]),
            max(start[1], end[1]),
        ]

    def motion(self, x, y):
        if self.gesture:
            self.canvas.delete("preview")
            self.canvas.create_rectangle(
                *self.transform.display(self.preview_bbox(x, y)),
                outline="#63c7ff",
                width=2,
                tags="preview",
            )

    def release(self, x, y):
        if not self.gesture:
            return
        bbox, gesture = self.preview_bbox(x, y), self.gesture
        self.gesture = None
        if bbox[2] - bbox[0] >= 1 and bbox[3] - bbox[1] >= 1:
            if gesture["box"]:
                if bbox != gesture["box"]["bbox"]:
                    self.store.edit_box(self.index, self.selected, bbox=bbox)
            else:
                self.selected = self.store.add_box(self.index, bbox)
        self.redraw()

    def delete(self):
        if self.selected:
            self.store.delete_box(self.index, self.selected)
            self.selected = None
            self.redraw()

    def toggle(self, field):
        box = next(
            (
                b
                for b in self.store.frame(self.index)["boxes"]
                if b["object_id"] == self.selected
            ),
            None,
        )
        if box:
            value = True if box[field] is None else False if box[field] else None
            self.store.edit_box(self.index, self.selected, **{field: value})
            self.redraw()

    def verify(self):
        self.store.verify(self.index)
        self.status.set(
            "Frame vérifiée et sauvegardée, y compris son absence de joueur si elle est vide."
        )
        self.redraw()

    def reopen(self):
        self.store.reopen(self.index)
        self.redraw()

    def key(self, event):
        actions = {
            "Left": lambda: self.navigate(-1),
            "Right": lambda: self.navigate(1),
            "Delete": self.delete,
            "BackSpace": self.delete,
            "v": self.verify,
            "r": self.reopen,
            "o": lambda: self.toggle("occluded"),
            "t": lambda: self.toggle("truncated"),
            "f": self.next_unverified,
            "n": lambda: self.new_box.set(not self.new_box.get()),
            "Escape": lambda: (setattr(self, "gesture", None), self.redraw()),
        }
        if event.keysym in actions:
            self.perform(actions[event.keysym])


def run(media, source_root, **options):
    media = Path(media).expanduser().resolve(strict=True)
    paths = [media]
    if media.is_dir():
        paths = sorted(
            p
            for p in media.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS and sidecar_path(p).is_file()
        )
    if not paths:
        raise ValueError("No images with player sidecars in this directory")
    root = tk.Tk()
    app = None
    try:
        app = App(root, paths, source_root, **options)
        root.mainloop()
    finally:
        if app and app.reader:
            app.reader.close()
        try:
            root.destroy()
        except tk.TclError:
            pass
