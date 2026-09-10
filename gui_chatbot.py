"""Launch the local desktop assistant.

Usage::

    python gui_chatbot.py

All processing happens locally — no Ollama or network required.
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
from tkinter import ttk

from assistant_core import CustomAssistant, Store, find_files, open_app, read_attachment

APP_NAME = "Sam"
MSG_LIMIT = 26000
VIEWS = (("chat", "Chat"), ("notes", "Notes"), ("tasks", "Tasks"), ("files", "Files & Apps"))


def enable_dpi_awareness():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Styling: palette + fonts
# ---------------------------------------------------------------------------

class Palette:
    BG = "#F4F5F7"
    SURFACE = "#FFFFFF"
    SIDEBAR = "#FBFBFC"
    BORDER = "#E4E7EC"
    INK = "#1C2128"
    MUTED = "#64707E"
    FAINT = "#9AA3B2"
    ACCENT = "#5B5BD6"
    ACCENT_HOVER = "#4A4AC2"
    ACCENT_PRESS = "#4040AC"
    ACCENT_TINT = "#EEF0FB"
    USER_BUBBLE = "#5B5BD6"
    USER_TEXT = "#FFFFFF"
    USER_TS = "#C9CDF6"
    BOT_BUBBLE = "#FFFFFF"
    READY = "#2E9E6B"
    READY_TINT = "#E4F5EC"
    WORKING = "#B0851A"
    WORKING_TINT = "#F7F0DC"
    ERROR = "#C04848"
    ERROR_TINT = "#FBEAE8"
    SELECT = "#DFE4F5"


def _pick_family(candidates):
    available = set(tkfont.families())
    for name in candidates:
        if name in available:
            return name
    return "TkDefaultFont"


def build_fonts(root):
    if sys.platform == "win32":
        family = _pick_family(["Segoe UI", "TkDefaultFont"])
    else:
        family = _pick_family(["Noto Sans", "DejaVu Sans", "Helvetica", "TkDefaultFont"])
    return {
        "ui": (family, 10),
        "ui_b": (family, 10, "bold"),
        "small": (family, 9),
        "small_b": (family, 9, "bold"),
        "title": (family, 15, "bold"),
        "msg": (family, 10),
        "ts": (family, 8),
        "brand": (family, 12, "bold"),
    }


# ---------------------------------------------------------------------------
# Widget factories (plain tk for deterministic colors on Win + Linux)
# ---------------------------------------------------------------------------

def styled_button(master, text, command=None, kind="secondary", font=None):
    """kind: 'primary' (accent), 'secondary' (white), 'danger', 'ghost'."""
    colors = {
        "primary": (Palette.ACCENT, Palette.ACCENT_HOVER, Palette.ACCENT_PRESS, Palette.SURFACE),
        "danger": (Palette.SURFACE, Palette.ERROR_TINT, Palette.ERROR_TINT, Palette.ERROR),
        "ghost": (Palette.SURFACE, Palette.BG, Palette.BORDER, Palette.MUTED),
        "secondary": (Palette.SURFACE, Palette.BG, Palette.BORDER, Palette.INK),
    }
    base, hover, press, fg = colors.get(kind, colors["secondary"])
    btn = tk.Button(
        master, text=text, command=command, cursor="hand2", relief="flat",
        bd=0, padx=14, pady=6, bg=base, fg=fg, activebackground=hover,
        activeforeground=fg, font=font, highlightthickness=1,
        highlightbackground=Palette.BORDER,
    )

    def _set(color):
        if str(btn["state"]) == "normal":
            btn.configure(bg=color)

    def on_enter(_e):
        _set(hover)

    def on_leave(_e):
        _set(base)

    def on_press(_e):
        _set(press)

    def on_release(_e):
        inside = btn.winfo_containing(btn.winfo_pointerx(), btn.winfo_pointery()) is btn
        _set(hover if inside else base)

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    btn.bind("<ButtonPress-1>", on_press)
    btn.bind("<ButtonRelease-1>", on_release)
    btn._palette = (base, base, fg)  # stored base for disabled-state restore
    return btn


def set_button_state(btn, enabled):
    base, _, fg = btn._palette
    if enabled:
        btn.configure(state="normal", bg=base, fg=fg)
    else:
        btn.configure(state="disabled", bg=Palette.BORDER, fg=Palette.FAINT)


def title_label(master, text, **kw):
    return tk.Label(master, text=text, bg=kw.pop("bg", Palette.SURFACE),
                    fg=kw.pop("fg", Palette.INK), font=kw.pop("font", None), anchor="w", **kw)


def pill(parent, text, bg, fg, font):
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=font, padx=12, pady=4)


# ---------------------------------------------------------------------------
# Chat transcript on a Canvas (rounded bubbles)
# ---------------------------------------------------------------------------

class ChatFeed(tk.Canvas):
    """Scrollable chat transcript with rounded message bubbles."""

    PAD_X = 14
    PAD_Y = 9
    GAP = 18
    RADIUS = 14
    MAX_WIDTH_FRACTION = 0.78
    Y_TOP = 22
    HINT = (
        "I can open apps, search files, manage notes and tasks, and more.\n"
        "Try:  open notepad  ·  find report.txt  ·  save a note  ·  what time is it"
    )

    def __init__(self, master, fonts, **kw):
        kw.setdefault("bg", Palette.BG)
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("bd", 0)
        super().__init__(master, **kw)
        self.fonts = fonts
        self._messages: list[tuple[str, str, str]] = []
        self._hits: list[tuple[int, int, int, int, str]] = []
        self._reflow_job = None
        self.bind("<Configure>", self._on_resize)
        self.bind("<Button-3>", self._copy_at)

    def add(self, role: str, content: str) -> None:
        stamped = datetime.now().strftime("%H:%M")
        self._messages.append((role, content, stamped))
        self._schedule_reflow(30)
        self.after(50, self._scroll_to_bottom)

    def clear(self) -> None:
        self._messages.clear()
        self._schedule_reflow(0)

    def _on_resize(self, _event):
        self._schedule_reflow(80)

    def _schedule_reflow(self, delay=60):
        if self._reflow_job:
            try:
                self.after_cancel(self._reflow_job)
            except tk.TclError:
                pass
        self._reflow_job = self.after(delay, self._reflow)

    def _scroll_to_bottom(self):
        self.yview_moveto(1.0)

    def _round_rect(self, x0, y0, x1, y1, r, **kw):
        return self.create_polygon(
            x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
            x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
            smooth=True, splinesteps=24, **kw,
        )

    def _reflow(self):
        self._reflow_job = None
        self.delete("all")
        self._hits.clear()

        width = max(self.winfo_width(), 300)
        max_bubble = int(width * self.MAX_WIDTH_FRACTION)
        side = int(min(120, width * 0.07))  # side gutter
        wrap = max(200, max_bubble - 2 * self.PAD_X)
        y = self.Y_TOP

        if not self._messages:
            text_id = self.create_text(
                width // 2, y, text=self.HINT, width=wrap, anchor="n",
                fill=Palette.FAINT, font=self.fonts["small"], justify="center",
            )
            _, ty0, _, ty1 = self.bbox(text_id)
            shape = self._round_rect(
                width * 0.08, ty0 - self.PAD_Y, width - width * 0.08,
                ty1 + self.PAD_Y, self.RADIUS, fill=Palette.SURFACE, outline=Palette.BORDER,
            )
            self.tag_lower(shape)
            y = ty1 + 2 * self.PAD_Y + 16
            self.configure(scrollregion=(0, 0, width, y))
            return

        for role, content, ts in self._messages:
            is_user = role == "user"
            anchor = "ne" if is_user else "nw"
            side_x = width - side if is_user else side

            text_id = self.create_text(
                side_x, y, text=content, width=wrap, anchor=anchor,
                fill=Palette.USER_TEXT if is_user else Palette.INK,
                font=self.fonts["msg"], justify="left",
            )
            tx0, ty0, tx1, ty1 = self.bbox(text_id)
            bx1 = tx1 + self.PAD_X if is_user else tx1 + self.PAD_X
            bx0 = bx1 - wrap - 2 * self.PAD_X if is_user else tx0 - self.PAD_X
            by0 = ty0 - self.PAD_Y
            by1 = ty1 + self.PAD_Y + 14  # reserve a line for the timestamp

            if is_user:
                shape = self._round_rect(bx0, by0, bx1, by1, self.RADIUS,
                                         fill=Palette.USER_BUBBLE, outline="")
                self.tag_lower(shape)
                self.create_text(bx1 - 4, by1 - 4, text=ts, anchor="se",
                                 fill=Palette.USER_TS, font=self.fonts["ts"])
            else:
                shape = self._round_rect(bx0, by0, bx1, by1, self.RADIUS,
                                         fill=Palette.BOT_BUBBLE, outline=Palette.BORDER)
                self.tag_lower(shape)
                self.create_text(bx1 - 4, by1 - 4, text=ts, anchor="se",
                                 fill=Palette.FAINT, font=self.fonts["ts"])

            self._hits.append((bx0, by0, bx1, by1, content))
            y = by1 + self.GAP

        self.configure(scrollregion=(0, 0, width, y))

    def _copy_at(self, event):
        for bx0, by0, bx1, by1, content in self._hits:
            if bx0 <= event.x <= bx1 and by0 <= event.y <= by1:
                self.clipboard_clear()
                self.clipboard_append(content)
                break


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class AssistantApp:
    def __init__(self, root, store=None):
        self.root = root
        self.store = store or Store()
        self.assistant = CustomAssistant(store=self.store)
        self.events: queue.Queue = queue.Queue()
        self.search_events: queue.Queue = queue.Queue()
        self.busy = False
        self._working_job = None
        self._active_view = None
        self.fonts = build_fonts(root)

        self.views: dict[str, tk.Frame] = {}
        self.nav_buttons: dict[str, tk.Button] = {}
        self.lists: dict[str, ttk.Treeview] = {}
        self.count_labels: dict[str, tk.Label] = {}
        self.empty_labels: dict[str, tk.Label] = {}

        self._style()
        self._build_shell()
        self._build_sidebar()
        self._build_views()
        self._show("chat")
        self._set_status("Ready", "ready")

    # -- setup ------------------------------------------------------------

    def _style(self):
        root = self.root
        root.title(APP_NAME)
        root.geometry("1000x720")
        root.minsize(840, 580)
        root.configure(bg=Palette.BG)
        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Item.Treeview",
                        background=Palette.SURFACE, fieldbackground=Palette.SURFACE,
                        foreground=Palette.INK, rowheight=34, borderwidth=0,
                        font=self.fonts["ui"], relief="flat")
        style.map("Item.Treeview",
                  background=[("selected", Palette.SELECT)],
                  foreground=[("selected", Palette.INK)])
        style.configure("Vertical.TScrollbar", background=Palette.SURFACE,
                        troughcolor=Palette.BG, borderwidth=0, arrowcolor=Palette.MUTED)

    def _build_shell(self):
        self.shell = tk.Frame(self.root, bg=Palette.BG)
        self.shell.pack(fill="both", expand=True)
        self.shell.grid_columnconfigure(1, weight=1)
        self.shell.grid_rowconfigure(0, weight=1)

    def _build_sidebar(self):
        sidebar = tk.Frame(self.shell, bg=Palette.SIDEBAR, width=196,
                           highlightthickness=1, highlightbackground=Palette.BORDER)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)
        self.sidebar = sidebar

        brand = tk.Frame(sidebar, bg=Palette.SIDEBAR)
        brand.pack(fill="x", padx=18, pady=(20, 26))
        dot = tk.Canvas(brand, width=16, height=16, bg=Palette.SIDEBAR, highlightthickness=0)
        dot.create_oval(2, 2, 14, 14, fill=Palette.ACCENT, outline="")
        dot.pack(side="left")
        tk.Label(brand, text=APP_NAME, bg=Palette.SIDEBAR, fg=Palette.INK,
                 font=self.fonts["brand"]).pack(side="left", padx=(10, 0))

        for key, label in VIEWS:
            btn = tk.Button(
                sidebar, text=label, anchor="w", relief="flat", bd=0,
                padx=16, pady=10, bg=Palette.SIDEBAR, fg=Palette.MUTED,
                activebackground=Palette.BG, activeforeground=Palette.INK,
                cursor="hand2", font=self.fonts["ui"], command=lambda k=key: self._show(k),
            )
            btn.pack(fill="x", padx=10, pady=1)
            self.nav_buttons[key] = btn

        badge = tk.Frame(sidebar, bg=Palette.SIDEBAR)
        badge.pack(side="bottom", fill="x", padx=18, pady=18)
        pip = tk.Canvas(badge, width=8, height=8, bg=Palette.SIDEBAR, highlightthickness=0)
        pip.create_oval(1, 1, 7, 7, fill=Palette.READY, outline="")
        pip.pack(side="left")
        tk.Label(badge, text="100% local · no cloud", bg=Palette.SIDEBAR,
                 fg=Palette.FAINT, font=self.fonts["small"]).pack(side="left", padx=(6, 0))

    def _build_views(self):
        container = tk.Frame(self.shell, bg=Palette.BG)
        container.grid(row=0, column=1, sticky="nsew")
        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.container = container

        self.status_pill = pill(container, "Ready", Palette.READY_TINT,
                                Palette.READY, self.fonts["small_b"])
        self.status_pill.grid(row=0, column=0, sticky="ne", padx=24, pady=(18, 10))

        for key, _ in VIEWS:
            frame = tk.Frame(container, bg=Palette.BG)
            frame.grid(row=1, column=0, sticky="nsew")
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)
            self.views[key] = frame

        self._build_chat_view()
        self._build_list_view("notes", "Notes", "note", "Add a note…")
        self._build_list_view("tasks", "Tasks", "task", "Add a task…")
        self._build_files_view()

    def _panel(self, parent, title):
        panel = tk.Frame(parent, bg=Palette.SURFACE, highlightthickness=1,
                         highlightbackground=Palette.BORDER)
        panel.grid(row=0, column=0, sticky="nsew", padx=24, pady=(0, 24))
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        header = tk.Frame(panel, bg=Palette.SURFACE)
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 12))
        header.grid_columnconfigure(0, weight=1)
        title_label(header, title, font=self.fonts["title"]).pack(side="left")

        body = tk.Frame(panel, bg=Palette.SURFACE)
        body.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 20))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        panel._header = header
        return panel, body

    def _add_header_right(self, header, widget):
        widget.pack_forget()
        widget.pack(side="right")

    # ------------------------------------------------------------------
    # Chat view
    # ------------------------------------------------------------------

    def _build_chat_view(self):
        panel, body = self._panel(self.views["chat"], "Chat")
        clear_btn = styled_button(panel._header, "Clear chat", command=self.clear_chat,
                                  kind="ghost", font=self.fonts["small"])
        self._add_header_right(panel._header, clear_btn)

        self.chat_feed = ChatFeed(body, self.fonts)
        self.chat_feed.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.chat_feed.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.chat_feed.configure(yscrollcommand=scroll.set)

        for msg in self.store.messages():
            self.chat_feed.add(msg["role"], msg["content"])
        self.chat_feed.after(80, self.chat_feed._scroll_to_bottom)

        composer = tk.Frame(body, bg=Palette.SURFACE)
        composer.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        composer.grid_columnconfigure(0, weight=1)

        self.entry = tk.Text(composer, height=2, wrap="word", relief="flat", bd=0,
                             font=self.fonts["msg"], bg=Palette.BG, fg=Palette.INK,
                             highlightthickness=1, highlightbackground=Palette.BORDER,
                             highlightcolor=Palette.ACCENT, padx=14, pady=10,
                             insertbackground=Palette.INK)
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.entry.tag_configure("ph", foreground=Palette.FAINT)
        self.entry.bind("<Return>", self._enter)
        self.entry.bind("<FocusIn>", lambda e: self._sync_placeholder())
        self.entry.bind("<FocusOut>", lambda e: self._sync_placeholder())
        self.entry.bind("<KeyRelease>", lambda e: self._sync_placeholder())

        self.send_button = styled_button(composer, "Send", command=self.send,
                                         kind="primary", font=self.fonts["ui_b"])
        self.send_button.grid(row=0, column=1, sticky="ns")
        self._sync_placeholder()

    def _sync_placeholder(self):
        empty = self.entry.get("1.0", "end").strip() == ""
        has_focus = self.entry.focus_get() is self.entry
        self.entry.tag_remove("ph", "1.0", "end")
        if empty and not has_focus:
            self.entry.insert("1.0", "Message your assistant…", "ph")

    # ------------------------------------------------------------------
    # Notes / Tasks
    # ------------------------------------------------------------------

    def _build_list_view(self, key, title, kind, placeholder):
        panel, body = self._panel(self.views[key], title)
        self.count_labels[key] = tk.Label(panel._header, text="", bg=Palette.SURFACE,
                                          fg=Palette.FAINT, font=self.fonts["small"])
        self._add_header_right(panel._header, self.count_labels[key])

        tree = ttk.Treeview(body, columns=("content",), show="tree",
                            selectmode="browse", style="Item.Treeview")
        tree.heading("#0", text="")
        tree.column("#0", width=600)
        tree.pack(fill="both", expand=True)
        tree.tag_configure("done", foreground=Palette.FAINT)
        self.lists[kind] = tree

        self.empty_labels[kind] = tk.Label(
            body, text="Nothing here yet", bg=Palette.SURFACE, fg=Palette.FAINT,
            font=self.fonts["ui"])

        row = tk.Frame(body, bg=Palette.SURFACE)
        row.pack(fill="x", pady=(12, 0))
        row.grid_columnconfigure(0, weight=1)
        entry = tk.Entry(row, relief="flat", bg=Palette.BG, fg=Palette.FAINT,
                         font=self.fonts["ui"], bd=0, insertbackground=Palette.INK,
                         highlightthickness=1, highlightbackground=Palette.BORDER,
                         highlightcolor=Palette.ACCENT)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 10), ipady=8, ipadx=10)
        entry._placeholder = placeholder
        entry.insert(0, placeholder)
        add_btn = styled_button(row, "Add", command=lambda: self._add_item(kind, entry),
                                kind="primary", font=self.fonts["ui_b"])
        add_btn.grid(row=0, column=1, sticky="ns")
        entry.bind("<FocusIn>", lambda e: self._entry_focus_in(entry))
        entry.bind("<FocusOut>", lambda e: self._entry_focus_out(entry))
        entry.bind("<Return>", lambda e: self._add_item(kind, entry))

        actions = tk.Frame(body, bg=Palette.SURFACE)
        actions.pack(fill="x", pady=(14, 0), side="bottom")
        ttk.Separator(actions, orient="horizontal").pack(fill="x", pady=(0, 12))
        styled_button(actions, "View", command=lambda: self._view_item(kind)).pack(side="left")
        if kind == "task":
            styled_button(actions, "Toggle done",
                          command=lambda: self._change_item(kind, False)).pack(side="left", padx=6)
        styled_button(actions, "Delete", kind="danger",
                      command=lambda: self._change_item(kind, True)).pack(side="right")

        self._reload_items(kind)

    def _entry_focus_in(self, entry):
        if entry.get() == entry._placeholder:
            entry.delete(0, "end")
            entry.configure(fg=Palette.INK)

    def _entry_focus_out(self, entry):
        if entry.get() == "":
            entry.insert(0, entry._placeholder)
            entry.configure(fg=Palette.FAINT)

    # ------------------------------------------------------------------
    # Files & Apps
    # ------------------------------------------------------------------

    def _build_files_view(self):
        panel, body = self._panel(self.views["files"], "Files & Apps")

        folder_ln = tk.Label(body, text="Folder", bg=Palette.SURFACE, fg=Palette.MUTED,
                             font=self.fonts["small_b"])
        folder_ln.grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.folder = tk.StringVar(value=self.store.setting("folder", str(Path.home())))
        folder_row = tk.Frame(body, bg=Palette.SURFACE)
        folder_row.grid(row=1, column=0, sticky="ew")
        folder_row.grid_columnconfigure(0, weight=1)
        self.folder_entry = tk.Entry(folder_row, textvariable=self.folder, relief="flat",
                                     bg=Palette.BG, fg=Palette.INK, font=self.fonts["ui"],
                                     bd=0, insertbackground=Palette.INK,
                                     highlightthickness=1, highlightbackground=Palette.BORDER,
                                     highlightcolor=Palette.ACCENT)
        self.folder_entry.pack(side="left", fill="x", expand=True, ipady=8, ipadx=10)
        styled_button(folder_row, "Choose…", command=self._choose_folder).pack(side="right", padx=(10, 0))

        search_ln = tk.Label(body, text="Search", bg=Palette.SURFACE, fg=Palette.MUTED,
                             font=self.fonts["small_b"])
        search_ln.grid(row=2, column=0, sticky="w", pady=(16, 4))
        search_row = tk.Frame(body, bg=Palette.SURFACE)
        search_row.grid(row=3, column=0, sticky="ew")
        search_row.grid_columnconfigure(0, weight=1)
        self.query = tk.Entry(search_row, relief="flat", bg=Palette.BG, fg=Palette.INK,
                              font=self.fonts["ui"], bd=0, insertbackground=Palette.INK,
                              highlightthickness=1, highlightbackground=Palette.BORDER,
                              highlightcolor=Palette.ACCENT)
        self.query.pack(side="left", fill="x", expand=True, ipady=8, ipadx=10)
        self.search_button = styled_button(search_row, "Search", command=self._search,
                                           kind="primary", font=self.fonts["ui_b"])
        self.search_button.pack(side="right", padx=(10, 0))
        self.query.bind("<Return>", lambda e: self._search())

        result_head = tk.Frame(body, bg=Palette.SURFACE)
        result_head.grid(row=4, column=0, sticky="ew", pady=(16, 6))
        title_label(result_head, "Results", fg=Palette.MUTED, font=self.fonts["small_b"]).pack(side="left")
        self.result_count = tk.Label(result_head, text="", bg=Palette.SURFACE, fg=Palette.FAINT,
                                     font=self.fonts["small"])
        self.result_count.pack(side="right")

        self.results = tk.Listbox(body, font=self.fonts["ui"], bg=Palette.BG, fg=Palette.INK,
                                  relief="flat", activestyle="none", borderwidth=0,
                                  highlightthickness=1, highlightbackground=Palette.BORDER,
                                  selectbackground=Palette.SELECT, selectforeground=Palette.INK)
        self.results.grid(row=5, column=0, sticky="nsew")
        body.grid_rowconfigure(5, weight=1)

        bar = tk.Frame(body, bg=Palette.SURFACE)
        bar.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        styled_button(bar, "Show in file manager", command=self._reveal).pack(side="left")
        styled_button(bar, "Attach text to chat", command=self._attach).pack(side="left", padx=6)

        apps = tk.Frame(body, bg=Palette.SURFACE)
        apps.grid(row=7, column=0, sticky="ew", pady=(14, 0))
        tk.Label(apps, text="Quick launch", bg=Palette.SURFACE, fg=Palette.MUTED,
                 font=self.fonts["small_b"]).pack(side="left")
        for name in ("Notepad", "Calculator"):
            styled_button(apps, name, command=lambda n=name: self._launch_app(n)).pack(side="left", padx=(6, 0))

    # ------------------------------------------------------------------
    # Navigation + status
    # ------------------------------------------------------------------

    def _show(self, key):
        if self._active_view == key:
            return
        for name, btn in self.nav_buttons.items():
            active = name == key
            btn.configure(bg=Palette.ACCENT_TINT if active else Palette.SIDEBAR,
                          fg=Palette.ACCENT if active else Palette.MUTED,
                          activebackground=Palette.ACCENT_TINT if active else Palette.BG)
        self._active_view = key
        self.views[key].lift()
        self.views[key].focus_set()

    def _set_status(self, text, tone="ready"):
        tones = {
            "ready": (Palette.READY_TINT, Palette.READY),
            "working": (Palette.WORKING_TINT, Palette.WORKING),
            "error": (Palette.ERROR_TINT, Palette.ERROR),
        }
        bg, fg = tones.get(tone, tones["ready"])
        self.status_pill.configure(text=text, bg=bg, fg=fg)

    def _start_working(self, label="Working"):
        if self._working_job:
            try:
                self.root.after_cancel(self._working_job)
            except tk.TclError:
                pass
        self._set_status(label, "working")
        count = [0]

        def tick():
            if not self.busy:
                return
            dots = "." * (count[0] % 4)
            self.status_pill.configure(text=label + dots)
            count[0] += 1
            self._working_job = self.root.after(360, tick)

        tick()

    def _stop_working(self):
        if self._working_job:
            try:
                self.root.after_cancel(self._working_job)
            except tk.TclError:
                pass
            self._working_job = None
        self._set_status("Ready", "ready")

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def _record(self, role, content):
        self.store.append(role, content)
        self.chat_feed.add(role, content)
        if self._active_view == "chat":
            self.views["chat"].lift()

    def _enter(self, event):
        if event.state & 1:  # Shift+Enter adds a newline
            return None
        self.send()
        return "break"

    def send(self):
        text = self.entry.get("1.0", "end").strip()
        if self.busy or not text:
            return
        if len(text) > MSG_LIMIT:
            messagebox.showerror("Message too long", "Keep messages under 26,000 characters.")
            return
        if text.startswith("/"):
            try:
                reply = self._command(text)
            except Exception as exc:
                messagebox.showerror("Command failed", str(exc))
                return
            self._record("user", text)
            self._record("assistant", reply)
            self.entry.delete("1.0", "end")
            return

        self._record("user", text)
        self.entry.delete("1.0", "end")
        self.busy = True
        set_button_state(self.send_button, False)
        self._start_working("Thinking")

        def run():
            try:
                reply = self.assistant.reply(text)
                self.events.put(("reply", reply, None))
            except Exception as exc:
                self.events.put(("reply", None, str(exc)))

        threading.Thread(target=run, daemon=True).start()
        self.root.after(80, self._poll)

    def _poll(self):
        try:
            kind, value, error = self.events.get_nowait()
            self.busy = False
            set_button_state(self.send_button, True)
            self._stop_working()
            if kind == "reply" and not error:
                self._record("assistant", value)
            elif error:
                self._set_status("Action failed", "error")
                messagebox.showerror("Error", error)
        except queue.Empty:
            self.root.after(80, self._poll)
            return
        if not self.events.empty():  # drain any remaining events
            self.root.after(20, self._poll)

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def _command(self, text):
        command, _, argument = text.partition(" ")
        if command in ("/note", "/todo"):
            kind = "note" if command == "/note" else "task"
            self.store.add(kind, argument)
            self._reload_items(kind)
            return "Saved."
        if command in ("/notes", "/todos"):
            rows = self.store.items("note" if command == "/notes" else "task")
            return "\n".join(f"{i}. {'[Done] ' if done else ''}{c}"
                             for i, c, done in rows) or "Nothing saved yet."
        if command == "/time":
            return datetime.now().astimezone().strftime("%A, %d %B %Y, %H:%M %Z")
        if command == "/open":
            result = open_app(argument)
            return result.get("reply", f"Opened {argument}.")
        if command == "/help":
            return (
                "Commands: /note TEXT, /todo TEXT, /notes, /todos, /time, "
                "/open notepad, /open calculator, /help.\n"
                "Or just chat naturally — I can search files, open apps, "
                "and manage notes and tasks."
            )
        return "Unknown command. Try /help."

    # ------------------------------------------------------------------
    # Files & apps
    # ------------------------------------------------------------------

    def _choose_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder.set(folder)
            self.store.set_setting("folder", folder)

    def _search(self):
        if str(self.search_button["state"]) == "disabled":
            return
        folder, query = self.folder.get(), self.query.get()
        set_button_state(self.search_button, False)
        self._start_working("Searching")

        def run():
            try:
                paths, limited = find_files(folder, query)
                self.search_events.put(("search", (paths, limited), None))
            except Exception as exc:
                self.search_events.put(("search", None, str(exc)))

        threading.Thread(target=run, daemon=True).start()
        self.root.after(80, self._poll_search)

    def _poll_search(self):
        try:
            kind, value, error = self.search_events.get_nowait()
            set_button_state(self.search_button, True)
            self._stop_working()
            if kind == "search" and not error:
                paths, limited = value
                self.results.delete(0, "end")
                for path in paths:
                    self.results.insert("end", path)
                self.result_count.configure(text=f"{len(paths)} found" + (" (limit)" if limited else ""))
                self._set_status("Search complete", "ready")
            elif error:
                self._set_status("Search failed", "error")
                messagebox.showerror("Error", error)
        except queue.Empty:
            self.root.after(80, self._poll_search)
            return
        if not self.search_events.empty():
            self.root.after(20, self._poll_search)

    def _reveal(self):
        selection = self.results.curselection()
        if selection:
            path = self.results.get(selection[0])
            try:
                if sys.platform == "win32":
                    subprocess.Popen(["explorer.exe", "/select,", path], shell=False)
                else:
                    subprocess.Popen(["xdg-open", str(Path(path).parent)], shell=False)
            except OSError as exc:
                messagebox.showerror("Could not open file manager", str(exc))

    def _attach(self):
        path = filedialog.askopenfilename(title="Attach a text or source-code file")
        if path:
            try:
                text = read_attachment(path)
                self.entry.insert("end", "\n" + text + "\n")
                self._show("chat")
                self.entry.focus_set()
            except (OSError, UnicodeError, ValueError) as exc:
                messagebox.showerror("Cannot attach file", str(exc))

    def _launch_app(self, name):
        try:
            open_app(name)
            self._set_status(f"Opened {name}", "ready")
        except OSError as exc:
            messagebox.showerror("Could not open app", str(exc))

    # ------------------------------------------------------------------
    # Notes / Tasks helpers
    # ------------------------------------------------------------------

    def _reload_items(self, kind):
        tree = self.lists[kind]
        for item in tree.get_children():
            tree.delete(item)
        rows = self.store.items(kind)
        for item_id, content, done in rows:
            label = ("✓ " if done else "○ ") if kind == "task" else content
            tree.insert("", "end", iid=str(item_id), text=label,
                        tags=("done",) if done else ())
        if rows:
            self.empty_labels[kind].place_forget()
        else:
            self.empty_labels[kind].place(relx=0.5, rely=0.42, anchor="center")
        self.count_labels["notes" if kind == "note" else "tasks"].configure(
            text=f"{len(rows)} item{'s' if len(rows) != 1 else ''}")

    def _add_item(self, kind, entry):
        value = entry.get().strip()
        if value and value != entry._placeholder:
            self.store.add(kind, value)
            entry.delete(0, "end")
            self._entry_focus_out(entry)
            self._reload_items(kind)
            self._set_status("Saved", "ready")

    def _view_item(self, kind):
        selection = self.lists[kind].selection()
        if selection:
            messagebox.showinfo("Saved item", self.lists[kind].item(selection[0], "text"))

    def _change_item(self, kind, delete):
        selection = self.lists[kind].selection()
        if selection:
            if delete:
                if not messagebox.askyesno("Delete item", "Delete the selected item?"):
                    return
                self.store.delete(int(selection[0]))
            else:
                self.store.toggle(int(selection[0]))
            self._reload_items(kind)

    def clear_chat(self):
        if not self.busy and messagebox.askyesno("Clear chat", "Delete saved chat history?"):
            self.store.clear_chat()
            self.assistant.reset()
            self.chat_feed.clear()


if __name__ == "__main__":
    enable_dpi_awareness()
    root = tk.Tk()
    AssistantApp(root)
    root.mainloop()
