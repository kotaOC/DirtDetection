"""Tkinter desktop user interface for image synthesis."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .stitcher import ImageStitcher, StitchError, StitchProgress, save_image


class ImageSynthesisApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("画像合成 - DirtDetection")
        self.root.geometry("1050x700")
        self.root.minsize(820, 560)
        self.paths: list[Path] = []
        self.result = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._build_ui()
        self.root.after(100, self._process_events)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("TkDefaultFont", 16, "bold"))
        style.configure("Hint.TLabel", foreground="#555555")
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text="横長部品の画像合成", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            outer,
            text="重なりのある画像を撮影順に並べ、［合成を実行］を押してください。",
            style="Hint.TLabel",
        ).pack(anchor=tk.W, pady=(2, 12))

        paned = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left = ttk.LabelFrame(paned, text="入力画像（上から撮影順）", padding=10)
        right = ttk.LabelFrame(paned, text="合成プレビュー", padding=10)
        paned.add(left, weight=2)
        paned.add(right, weight=3)

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, activestyle="dotbox")
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        buttons = ttk.Frame(left)
        buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(buttons, text="画像を追加…", command=self._add_files).pack(side=tk.LEFT)
        ttk.Button(buttons, text="削除", command=self._remove_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="↑", width=4, command=lambda: self._move(-1)).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Button(buttons, text="↓", width=4, command=lambda: self._move(1)).pack(side=tk.LEFT)
        ttk.Button(buttons, text="すべて消去", command=self._clear).pack(side=tk.RIGHT)

        self.preview = ttk.Label(right, text="合成結果がここに表示されます", anchor=tk.CENTER)
        self.preview.pack(fill=tk.BOTH, expand=True)
        self.preview.bind("<Configure>", lambda _event: self._refresh_preview())

        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X, pady=(12, 0))
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X)
        self.status = tk.StringVar(value="画像を2枚以上追加してください。")
        ttk.Label(footer, textvariable=self.status).pack(anchor=tk.W, pady=(4, 8))
        action = ttk.Frame(footer)
        action.pack(fill=tk.X)
        self.save_button = ttk.Button(action, text="結果を保存…", command=self._save, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT)
        self.run_button = ttk.Button(action, text="合成を実行", command=self._start, state=tk.DISABLED)
        self.run_button.pack(side=tk.RIGHT, padx=(0, 8))

    def _add_files(self) -> None:
        selected = filedialog.askopenfilenames(
            title="合成する画像を選択",
            filetypes=[("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("すべて", "*.*")],
        )
        known = {str(path.resolve()) for path in self.paths}
        for value in selected:
            path = Path(value)
            if str(path.resolve()) not in known:
                self.paths.append(path)
                known.add(str(path.resolve()))
        self._sync_list()

    def _remove_selected(self) -> None:
        for index in reversed(self.listbox.curselection()):
            del self.paths[index]
        self._sync_list()

    def _clear(self) -> None:
        self.paths.clear()
        self.result = None
        self.preview_photo = None
        self.preview.configure(image="", text="合成結果がここに表示されます")
        self.save_button.configure(state=tk.DISABLED)
        self._sync_list()

    def _move(self, direction: int) -> None:
        selected = list(self.listbox.curselection())
        if len(selected) != 1:
            return
        old = selected[0]
        new = old + direction
        if not 0 <= new < len(self.paths):
            return
        self.paths[old], self.paths[new] = self.paths[new], self.paths[old]
        self._sync_list()
        self.listbox.selection_set(new)
        self.listbox.activate(new)

    def _sync_list(self) -> None:
        self.listbox.delete(0, tk.END)
        for number, path in enumerate(self.paths, start=1):
            self.listbox.insert(tk.END, f"{number:02d}  {path.name}")
        self.run_button.configure(state=tk.NORMAL if len(self.paths) >= 2 else tk.DISABLED)
        self.status.set(f"{len(self.paths)}枚を選択中" if self.paths else "画像を2枚以上追加してください。")

    def _start(self) -> None:
        self.run_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        self.progress.configure(value=0)
        self.status.set("合成を開始しています…")
        paths = list(self.paths)
        threading.Thread(target=self._worker, args=(paths,), daemon=True).start()

    def _worker(self, paths: list[Path]) -> None:
        try:
            result = ImageStitcher().stitch_files(paths, self._report_progress)
            self.events.put(("done", result))
        except Exception as exc:  # delivered safely to the GUI thread
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
                    self.status.set(message)
                elif event == "done":
                    self.result = payload
                    self.progress.configure(value=100)
                    self.status.set(f"合成完了: {self.result.shape[1]} × {self.result.shape[0]} px")
                    self.run_button.configure(state=tk.NORMAL)
                    self.save_button.configure(state=tk.NORMAL)
                    self._refresh_preview()
                elif event == "error":
                    self.run_button.configure(state=tk.NORMAL if len(self.paths) >= 2 else tk.DISABLED)
                    self.status.set("合成に失敗しました。")
                    text = str(payload) if isinstance(payload, StitchError) else f"予期しないエラー: {payload}"
                    messagebox.showerror("合成エラー", text)
        except queue.Empty:
            pass
        self.root.after(100, self._process_events)

    def _refresh_preview(self) -> None:
        if self.result is None:
            return
        rgb = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(self.preview.winfo_width() - 10, 100)
        height = max(self.preview.winfo_height() - 10, 100)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_photo, text="")

    def _save(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            title="合成画像を保存",
            defaultextension=".png",
            initialfile="synthesized.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("TIFF", "*.tif *.tiff"), ("BMP", "*.bmp")],
        )
        if not path:
            return
        try:
            save_image(path, self.result)
            self.status.set(f"保存しました: {path}")
            messagebox.showinfo("保存完了", "合成画像を保存しました。")
        except StitchError as exc:
            messagebox.showerror("保存エラー", str(exc))


def main() -> None:
    root = tk.Tk()
    ImageSynthesisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
