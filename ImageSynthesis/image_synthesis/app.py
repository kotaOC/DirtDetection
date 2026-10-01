"""Dark, inspection-system style Tkinter UI for image synthesis."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .stitcher import ImageStitcher, StitchError, StitchProgress, save_image


class ImageSynthesisApp:
    """Desktop front end. Image registration and blending live in ``stitcher.py``."""

    C = {
        "bg": "#08111F",
        "panel": "#0E1B2D",
        "card": "#13243A",
        "card_hover": "#182C45",
        "field": "#091727",
        "line": "#29435F",
        "line_bright": "#3D6489",
        "text": "#F3F7FC",
        "muted": "#91A6BE",
        "blue": "#2389D7",
        "blue_hover": "#319DEB",
        "green": "#24A66A",
        "green_hover": "#2ABA78",
        "red": "#B84A5B",
        "red_hover": "#CC586A",
        "disabled": "#263A50",
        "disabled_text": "#6F849B",
    }

    THUMBNAIL_SIZE = (104, 72)
    MIN_ZOOM = 0.1
    MAX_ZOOM = 4.0

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.paths: list[Path] = []
        self.result: np.ndarray | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.selected_index: int | None = None
        self.thumbnail_photos: list[ImageTk.PhotoImage] = []
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.preview_zoom = 1.0
        self.fit_mode = True
        self.busy = False

        root.title("Image Synthesis | 金属部品 画像合成")
        root.geometry("1440x860")
        root.minsize(1060, 680)
        root.configure(bg=self.C["bg"])

        self._configure_styles()
        self._build_ui()
        root.after(100, self._process_events)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Dark.Horizontal.TProgressbar", troughcolor=self.C["field"], background=self.C["blue"],
                        bordercolor=self.C["field"], lightcolor=self.C["blue"], darkcolor=self.C["blue"], thickness=8)
        style.configure("Dark.Vertical.TScrollbar", background=self.C["card"], troughcolor=self.C["field"],
                        bordercolor=self.C["field"], arrowcolor=self.C["muted"])
        style.configure("Dark.Horizontal.TScrollbar", background=self.C["card"], troughcolor=self.C["field"],
                        bordercolor=self.C["field"], arrowcolor=self.C["muted"])

    def _button(self, parent: tk.Misc, text: str, command, kind: str = "secondary", width: int | None = None) -> tk.Button:
        palette = {
            "primary": (self.C["blue"], self.C["blue_hover"], self.C["text"]),
            "success": (self.C["green"], self.C["green_hover"], self.C["text"]),
            "danger": (self.C["red"], self.C["red_hover"], self.C["text"]),
            "secondary": (self.C["card"], self.C["card_hover"], self.C["text"]),
            "ghost": (self.C["panel"], self.C["card_hover"], self.C["muted"]),
        }
        bg, active, fg = palette[kind]
        button = tk.Button(parent, text=text, command=command, bg=bg, fg=fg, activebackground=active,
                           activeforeground=self.C["text"], disabledforeground=self.C["disabled_text"],
                           relief=tk.FLAT, borderwidth=0, cursor="hand2", font=("Segoe UI Semibold", 10),
                           padx=14, pady=9, highlightthickness=0)
        if width is not None:
            button.configure(width=width)
        return button

    def _build_ui(self) -> None:
        shell = tk.Frame(self.root, bg=self.C["bg"])
        shell.pack(fill=tk.BOTH, expand=True, padx=24, pady=20)
        self._build_header(shell)

        workspace = tk.Frame(shell, bg=self.C["bg"])
        workspace.pack(fill=tk.BOTH, expand=True, pady=(18, 0))
        workspace.grid_columnconfigure(0, weight=0, minsize=390)
        workspace.grid_columnconfigure(1, weight=1, minsize=600)
        workspace.grid_rowconfigure(0, weight=1)
        self._build_input_card(workspace)
        self._build_preview_card(workspace)
        self._build_footer(shell)
        self._update_action_states()

    def _build_header(self, parent: tk.Misc) -> None:
        header = tk.Frame(parent, bg=self.C["bg"], height=70)
        header.pack(fill=tk.X)
        title_area = tk.Frame(header, bg=self.C["bg"])
        title_area.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(title_area, text="IMAGE PROCESSING  /  STITCHING", bg=self.C["bg"], fg="#54B6F4",
                 font=("Segoe UI Semibold", 9)).pack(anchor=tk.W)
        row = tk.Frame(title_area, bg=self.C["bg"])
        row.pack(anchor=tk.W, pady=(3, 0))
        tk.Label(row, text="Image Synthesis", bg=self.C["bg"], fg=self.C["text"],
                 font=("Segoe UI Semibold", 23)).pack(side=tk.LEFT)
        tk.Label(row, text="金属部品 画像合成", bg=self.C["bg"], fg=self.C["muted"],
                 font=("Yu Gothic UI", 11)).pack(side=tk.LEFT, padx=(18, 0), pady=(7, 0))

        info = tk.Frame(header, bg=self.C["panel"], highlightthickness=1, highlightbackground=self.C["line"])
        info.pack(side=tk.RIGHT, pady=4)
        self.count_value = tk.Label(info, text="0", bg=self.C["panel"], fg="#5EC4FF", font=("Segoe UI Semibold", 17), width=3)
        self.count_value.pack(side=tk.LEFT, padx=(12, 0), pady=8)
        tk.Label(info, text="INPUT\nIMAGES", bg=self.C["panel"], fg=self.C["muted"], justify=tk.LEFT,
                 font=("Segoe UI Semibold", 8)).pack(side=tk.LEFT, padx=(0, 15))

    def _card_header(self, parent: tk.Misc, number: str, title: str, subtitle: str) -> tk.Frame:
        header = tk.Frame(parent, bg=self.C["panel"])
        header.pack(fill=tk.X, padx=18, pady=(16, 12))
        badge = tk.Label(header, text=number, bg=self.C["blue"], fg="white", font=("Segoe UI Semibold", 9), width=3, pady=4)
        badge.pack(side=tk.LEFT, padx=(0, 10))
        text = tk.Frame(header, bg=self.C["panel"])
        text.pack(side=tk.LEFT)
        tk.Label(text, text=title, bg=self.C["panel"], fg=self.C["text"], font=("Yu Gothic UI Semibold", 12)).pack(anchor=tk.W)
        tk.Label(text, text=subtitle, bg=self.C["panel"], fg=self.C["muted"], font=("Yu Gothic UI", 8)).pack(anchor=tk.W, pady=(2, 0))
        return header

    def _build_input_card(self, parent: tk.Misc) -> None:
        card = tk.Frame(parent, bg=self.C["panel"], highlightthickness=1, highlightbackground=self.C["line"])
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        self._card_header(card, "01", "入力画像", "撮影順に並べてください")

        add_wrap = tk.Frame(card, bg=self.C["panel"])
        add_wrap.pack(fill=tk.X, padx=18, pady=(0, 12))
        self.add_button = self._button(add_wrap, "＋  画像を追加", self._add_files, "primary")
        self.add_button.pack(fill=tk.X)

        list_shell = tk.Frame(card, bg=self.C["field"], highlightthickness=1, highlightbackground=self.C["line"])
        list_shell.pack(fill=tk.BOTH, expand=True, padx=18)
        self.list_canvas = tk.Canvas(list_shell, bg=self.C["field"], highlightthickness=0, borderwidth=0)
        list_scroll = ttk.Scrollbar(list_shell, orient=tk.VERTICAL, style="Dark.Vertical.TScrollbar", command=self.list_canvas.yview)
        self.list_canvas.configure(yscrollcommand=list_scroll.set)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.items_frame = tk.Frame(self.list_canvas, bg=self.C["field"])
        self.items_window = self.list_canvas.create_window((0, 0), window=self.items_frame, anchor="nw")
        self.items_frame.bind("<Configure>", self._update_items_scrollregion)
        self.list_canvas.bind("<Configure>", lambda event: self.list_canvas.itemconfigure(self.items_window, width=event.width))
        self.list_canvas.bind_all("<MouseWheel>", self._on_list_wheel)
        self.empty_list = tk.Label(self.items_frame, text="画像が追加されていません\n\n対応形式  PNG / JPEG / BMP / TIFF",
                                   bg=self.C["field"], fg=self.C["muted"], font=("Yu Gothic UI", 10), justify=tk.CENTER)
        self.empty_list.pack(fill=tk.X, pady=72)

        order = tk.Frame(card, bg=self.C["panel"])
        order.pack(fill=tk.X, padx=18, pady=(12, 8))
        tk.Label(order, text="撮影順序", bg=self.C["panel"], fg=self.C["muted"], font=("Yu Gothic UI", 9)).pack(side=tk.LEFT)
        self.up_button = self._button(order, "↑  上へ", lambda: self._move(-1), "secondary", 8)
        self.up_button.pack(side=tk.RIGHT)
        self.down_button = self._button(order, "↓  下へ", lambda: self._move(1), "secondary", 8)
        self.down_button.pack(side=tk.RIGHT, padx=(0, 7))

        destructive = tk.Frame(card, bg=self.C["panel"])
        destructive.pack(fill=tk.X, padx=18, pady=(0, 16))
        self.remove_button = self._button(destructive, "選択画像を削除", self._remove_selected, "danger")
        self.remove_button.pack(side=tk.LEFT)
        self.clear_button = self._button(destructive, "すべてクリア", self._clear, "ghost")
        self.clear_button.pack(side=tk.RIGHT)

    def _build_preview_card(self, parent: tk.Misc) -> None:
        card = tk.Frame(parent, bg=self.C["panel"], highlightthickness=1, highlightbackground=self.C["line"])
        card.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        header = self._card_header(card, "02", "合成結果", "Fit表示・拡大縮小・スクロールに対応")
        self.result_meta = tk.Label(header, text="NO RESULT", bg=self.C["panel"], fg=self.C["muted"],
                                    font=("Consolas", 9), padx=10)
        self.result_meta.pack(side=tk.RIGHT)

        toolbar = tk.Frame(card, bg=self.C["panel"])
        toolbar.pack(fill=tk.X, padx=18, pady=(0, 10))
        self.fit_button = self._button(toolbar, "Fit", self._fit_preview, "primary", 6)
        self.fit_button.pack(side=tk.LEFT)
        self.zoom_out_button = self._button(toolbar, "−", lambda: self._change_zoom(0.8), "secondary", 3)
        self.zoom_out_button.pack(side=tk.LEFT, padx=(8, 4))
        self.zoom_in_button = self._button(toolbar, "＋", lambda: self._change_zoom(1.25), "secondary", 3)
        self.zoom_in_button.pack(side=tk.LEFT)
        self.zoom_label = tk.Label(toolbar, text="FIT", bg=self.C["panel"], fg="#5EC4FF", font=("Consolas", 10), width=7)
        self.zoom_label.pack(side=tk.LEFT, padx=8)
        tk.Label(toolbar, text="Ctrl + ホイールでも拡大縮小", bg=self.C["panel"], fg=self.C["muted"],
                 font=("Yu Gothic UI", 8)).pack(side=tk.RIGHT)

        viewer = tk.Frame(card, bg="#030910", highlightthickness=1, highlightbackground=self.C["line"])
        viewer.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 18))
        self.preview_canvas = tk.Canvas(viewer, bg="#030910", highlightthickness=0, borderwidth=0)
        vscroll = ttk.Scrollbar(viewer, orient=tk.VERTICAL, style="Dark.Vertical.TScrollbar", command=self.preview_canvas.yview)
        hscroll = ttk.Scrollbar(viewer, orient=tk.HORIZONTAL, style="Dark.Horizontal.TScrollbar", command=self.preview_canvas.xview)
        self.preview_canvas.configure(xscrollcommand=hscroll.set, yscrollcommand=vscroll.set)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.preview_canvas.bind("<Configure>", self._on_preview_resize)
        self.preview_canvas.bind("<Control-MouseWheel>", self._on_preview_zoom)
        self._draw_empty_preview()

    def _build_footer(self, parent: tk.Misc) -> None:
        footer = tk.Frame(parent, bg=self.C["panel"], highlightthickness=1, highlightbackground=self.C["line"])
        footer.pack(fill=tk.X, pady=(18, 0))
        state = tk.Frame(footer, bg=self.C["panel"])
        state.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=18, pady=13)
        status_row = tk.Frame(state, bg=self.C["panel"])
        status_row.pack(fill=tk.X, pady=(0, 7))
        self.status_dot = tk.Label(status_row, text="●", bg=self.C["panel"], fg=self.C["muted"], font=("Segoe UI", 9))
        self.status_dot.pack(side=tk.LEFT, padx=(0, 7))
        self.status = tk.StringVar(value="待機中 — 画像を2枚以上追加してください")
        tk.Label(status_row, textvariable=self.status, bg=self.C["panel"], fg=self.C["text"],
                 font=("Yu Gothic UI", 9)).pack(side=tk.LEFT)
        self.progress_text = tk.Label(status_row, text="0%", bg=self.C["panel"], fg=self.C["muted"], font=("Consolas", 9))
        self.progress_text.pack(side=tk.RIGHT)
        self.progress = ttk.Progressbar(state, style="Dark.Horizontal.TProgressbar", mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X)

        actions = tk.Frame(footer, bg=self.C["panel"])
        actions.pack(side=tk.RIGHT, padx=18, pady=12)
        self.save_button = self._button(actions, "結果を保存", self._save, "success", 13)
        self.save_button.configure(state=tk.DISABLED, cursor="arrow")
        self.save_button.pack(side=tk.LEFT, padx=(0, 10))
        self.run_button = self._button(actions, "合成を開始  ▶", self._start, "primary", 16)
        self.run_button.configure(font=("Yu Gothic UI Semibold", 11), pady=12)
        self.run_button.pack(side=tk.LEFT)

    def _add_files(self) -> None:
        selected = filedialog.askopenfilenames(
            title="合成する画像を選択",
            filetypes=[("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("すべてのファイル", "*.*")],
        )
        known = {str(path.resolve()) for path in self.paths}
        for value in selected:
            path = Path(value)
            if str(path.resolve()) not in known:
                self.paths.append(path)
                known.add(str(path.resolve()))
        if selected:
            self.selected_index = len(self.paths) - 1
        self._rebuild_image_cards()

    def _load_thumbnail(self, path: Path) -> ImageTk.PhotoImage:
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail(self.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", self.THUMBNAIL_SIZE, "#050D17")
                canvas.paste(image, ((self.THUMBNAIL_SIZE[0] - image.width) // 2, (self.THUMBNAIL_SIZE[1] - image.height) // 2))
        except (OSError, ValueError):
            canvas = Image.new("RGB", self.THUMBNAIL_SIZE, "#091727")
        return ImageTk.PhotoImage(canvas)

    def _rebuild_image_cards(self) -> None:
        for widget in self.items_frame.winfo_children():
            widget.destroy()
        self.thumbnail_photos.clear()
        if not self.paths:
            self.empty_list = tk.Label(self.items_frame, text="画像が追加されていません\n\n対応形式  PNG / JPEG / BMP / TIFF",
                                       bg=self.C["field"], fg=self.C["muted"], font=("Yu Gothic UI", 10), justify=tk.CENTER)
            self.empty_list.pack(fill=tk.X, pady=72)
        else:
            for index, path in enumerate(self.paths):
                selected = index == self.selected_index
                bg = self.C["card_hover"] if selected else self.C["card"]
                border = self.C["blue"] if selected else self.C["line"]
                item = tk.Frame(self.items_frame, bg=bg, highlightthickness=2 if selected else 1, highlightbackground=border)
                item.pack(fill=tk.X, padx=9, pady=(9 if index == 0 else 0, 7))
                order = tk.Label(item, text=f"{index + 1:02d}", bg=bg, fg="#5EC4FF", font=("Consolas", 12, "bold"), width=3)
                order.pack(side=tk.LEFT, padx=(8, 3))
                photo = self._load_thumbnail(path)
                self.thumbnail_photos.append(photo)
                thumb = tk.Label(item, image=photo, bg="#050D17", borderwidth=0)
                thumb.pack(side=tk.LEFT, padx=(0, 10), pady=8)
                details = tk.Frame(item, bg=bg)
                details.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=9)
                name = path.name if len(path.name) <= 29 else path.name[:26] + "…"
                tk.Label(details, text=name, bg=bg, fg=self.C["text"], anchor=tk.W,
                         font=("Yu Gothic UI Semibold", 9)).pack(fill=tk.X)
                size_text = self._image_size_text(path)
                tk.Label(details, text=f"撮影順序 {index + 1}  ·  {size_text}", bg=bg, fg=self.C["muted"], anchor=tk.W,
                         font=("Yu Gothic UI", 8)).pack(fill=tk.X, pady=(7, 0))
                for widget in (item, order, thumb, details):
                    widget.bind("<Button-1>", lambda _event, i=index: self._select_image(i))
                for child in details.winfo_children():
                    child.bind("<Button-1>", lambda _event, i=index: self._select_image(i))
        self.count_value.configure(text=str(len(self.paths)))
        self._update_action_states()
        self.items_frame.update_idletasks()
        self.list_canvas.configure(scrollregion=self.list_canvas.bbox("all"))

    @staticmethod
    def _image_size_text(path: Path) -> str:
        try:
            with Image.open(path) as image:
                return f"{image.width} × {image.height} px"
        except (OSError, ValueError):
            return "サイズ不明"

    def _select_image(self, index: int) -> None:
        self.selected_index = index
        self._rebuild_image_cards()

    def _remove_selected(self) -> None:
        if self.selected_index is None or not 0 <= self.selected_index < len(self.paths):
            return
        del self.paths[self.selected_index]
        if self.paths:
            self.selected_index = min(self.selected_index, len(self.paths) - 1)
        else:
            self.selected_index = None
        self._rebuild_image_cards()

    def _clear(self) -> None:
        if not self.paths:
            return
        if not messagebox.askyesno("入力画像をクリア", "追加した画像をすべて取り除きますか？"):
            return
        self.paths.clear()
        self.selected_index = None
        self.result = None
        self.preview_photo = None
        self.progress.configure(value=0)
        self.progress_text.configure(text="0%")
        self.result_meta.configure(text="NO RESULT")
        self.save_button.configure(state=tk.DISABLED)
        self._draw_empty_preview()
        self._rebuild_image_cards()

    def _move(self, direction: int) -> None:
        if self.selected_index is None:
            return
        new_index = self.selected_index + direction
        if not 0 <= new_index < len(self.paths):
            return
        self.paths[self.selected_index], self.paths[new_index] = self.paths[new_index], self.paths[self.selected_index]
        self.selected_index = new_index
        self._rebuild_image_cards()

    def _update_action_states(self) -> None:
        count = len(self.paths)
        selected = self.selected_index is not None and count > 0
        self._set_button_state(self.run_button, count >= 2 and not self.busy)
        self._set_button_state(self.add_button, not self.busy)
        self._set_button_state(self.remove_button, selected and not self.busy)
        self._set_button_state(self.clear_button, count > 0 and not self.busy)
        self._set_button_state(self.up_button, selected and self.selected_index > 0 and not self.busy)
        self._set_button_state(self.down_button, selected and self.selected_index < count - 1 and not self.busy)
        if not self.busy:
            if count >= 2:
                self._set_status(f"準備完了 — {count}枚の画像を撮影順に合成します", "ready")
            elif count == 1:
                self._set_status("画像をあと1枚追加してください", "idle")
            else:
                self._set_status("待機中 — 画像を2枚以上追加してください", "idle")

    def _set_button_state(self, button: tk.Button, enabled: bool) -> None:
        button.configure(state=tk.NORMAL if enabled else tk.DISABLED, cursor="hand2" if enabled else "arrow")

    def _start(self) -> None:
        if len(self.paths) < 2 or self.busy:
            return
        self.busy = True
        self.progress.configure(value=0)
        self.progress_text.configure(text="0%")
        self.save_button.configure(state=tk.DISABLED)
        self._set_status("画像を読み込んでいます…", "working")
        self._update_action_states()
        threading.Thread(target=self._worker, args=(list(self.paths),), daemon=True).start()

    def _worker(self, paths: list[Path]) -> None:
        try:
            self.events.put(("done", ImageStitcher().stitch_files(paths, self._report_progress)))
        except Exception as exc:
            self.events.put(("error", exc))

    def _report_progress(self, update: StitchProgress) -> None:
        percent = 100 * update.current / max(update.total, 1)
        self.events.put(("progress", (percent, update.message)))

    def _process_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "progress":
                    percent, message = payload  # type: ignore[misc]
                    self.progress.configure(value=percent)
                    self.progress_text.configure(text=f"{percent:.0f}%")
                    self._set_status(str(message), "working")
                elif event == "done":
                    self.result = payload  # type: ignore[assignment]
                    self.busy = False
                    height, width = self.result.shape[:2]
                    self.progress.configure(value=100)
                    self.progress_text.configure(text="100%")
                    self.result_meta.configure(text=f"{width:,} × {height:,} PX")
                    self._set_button_state(self.save_button, True)
                    self._update_action_states()
                    self._set_status("合成が完了しました — 結果を確認して保存してください", "success")
                    self._fit_preview()
                elif event == "error":
                    self.busy = False
                    self._update_action_states()
                    self._set_status("合成に失敗しました", "error")
                    text = str(payload) if isinstance(payload, StitchError) else f"予期しないエラー: {payload}"
                    messagebox.showerror("合成エラー", text)
        except queue.Empty:
            pass
        self.root.after(100, self._process_events)

    def _set_status(self, text: str, state: str) -> None:
        colors = {"idle": self.C["muted"], "ready": "#5EC4FF", "working": "#F1B84B",
                  "success": "#42D38B", "error": "#F06B78"}
        self.status.set(text)
        self.status_dot.configure(fg=colors[state])

    def _draw_empty_preview(self) -> None:
        self.preview_canvas.delete("all")
        width = max(self.preview_canvas.winfo_width(), 600)
        height = max(self.preview_canvas.winfo_height(), 400)
        x, y = width / 2, height / 2
        self.preview_canvas.create_rectangle(x - 34, y - 62, x + 34, y - 16, outline=self.C["line_bright"], width=2)
        self.preview_canvas.create_line(x - 24, y - 27, x - 7, y - 43, x + 7, y - 31, x + 20, y - 45,
                                        fill=self.C["line_bright"], width=2)
        self.preview_canvas.create_text(x, y + 17, text="画像を2枚以上追加して\n合成を開始してください",
                                        fill=self.C["text"], font=("Yu Gothic UI Semibold", 12), justify=tk.CENTER)
        self.preview_canvas.create_text(x, y + 66, text="合成結果がここに表示されます", fill=self.C["muted"], font=("Yu Gothic UI", 9))
        self.preview_canvas.configure(scrollregion=(0, 0, width, height))

    def _fit_preview(self) -> None:
        if self.result is None:
            return
        self.fit_mode = True
        canvas_w = max(self.preview_canvas.winfo_width() - 24, 100)
        canvas_h = max(self.preview_canvas.winfo_height() - 24, 100)
        image_h, image_w = self.result.shape[:2]
        self.preview_zoom = min(canvas_w / image_w, canvas_h / image_h, 1.0)
        self.zoom_label.configure(text="FIT")
        self._render_preview()

    def _change_zoom(self, factor: float) -> None:
        if self.result is None:
            return
        self.fit_mode = False
        self.preview_zoom = min(self.MAX_ZOOM, max(self.MIN_ZOOM, self.preview_zoom * factor))
        self.zoom_label.configure(text=f"{self.preview_zoom * 100:.0f}%")
        self._render_preview()

    def _render_preview(self) -> None:
        if self.result is None:
            self._draw_empty_preview()
            return
        rgb = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)
        source = Image.fromarray(rgb)
        width = max(1, round(source.width * self.preview_zoom))
        height = max(1, round(source.height * self.preview_zoom))
        interpolation = Image.Resampling.LANCZOS if self.preview_zoom < 1 else Image.Resampling.BICUBIC
        image = source.resize((width, height), interpolation)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_canvas.delete("all")
        viewport_w = self.preview_canvas.winfo_width()
        viewport_h = self.preview_canvas.winfo_height()
        x = max(viewport_w // 2, width // 2 + 12)
        y = max(viewport_h // 2, height // 2 + 12)
        self.preview_canvas.create_image(x, y, image=self.preview_photo, anchor=tk.CENTER)
        content_w = max(viewport_w, width + 24)
        content_h = max(viewport_h, height + 24)
        self.preview_canvas.configure(scrollregion=(0, 0, content_w, content_h))
        self.preview_canvas.xview_moveto(0.5 if width > viewport_w else 0)
        self.preview_canvas.yview_moveto(0.5 if height > viewport_h else 0)

    def _on_preview_resize(self, _event: tk.Event) -> None:
        if self.result is None:
            self._draw_empty_preview()
        elif self.fit_mode:
            self.root.after_idle(self._fit_preview)

    def _on_preview_zoom(self, event: tk.Event) -> str:
        self._change_zoom(1.25 if event.delta > 0 else 0.8)
        return "break"

    def _update_items_scrollregion(self, _event: tk.Event) -> None:
        self.list_canvas.configure(scrollregion=self.list_canvas.bbox("all"))

    def _on_list_wheel(self, event: tk.Event) -> None:
        pointer = self.root.winfo_containing(event.x_root, event.y_root)
        if pointer is not None and self._is_descendant(pointer, self.list_canvas.master):
            self.list_canvas.yview_scroll(int(-event.delta / 120), "units")

    @staticmethod
    def _is_descendant(widget: tk.Misc, ancestor: tk.Misc) -> bool:
        current: tk.Misc | None = widget
        while current is not None:
            if current == ancestor:
                return True
            current = current.master
        return False

    def _save(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            title="合成画像を保存", defaultextension=".png", initialfile="synthesized.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("TIFF", "*.tif *.tiff"), ("BMP", "*.bmp")],
        )
        if not path:
            return
        try:
            save_image(path, self.result)
            self._set_status(f"保存しました — {Path(path).name}", "success")
            messagebox.showinfo("保存完了", "合成画像を保存しました。")
        except StitchError as exc:
            messagebox.showerror("保存エラー", str(exc))


def main() -> None:
    root = tk.Tk()
    ImageSynthesisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
