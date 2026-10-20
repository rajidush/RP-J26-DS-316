"""
overlay.py — Response overlay window for Component 1 (Function 2: Post-Detection Response).

Responsibilities
----------------
* Own a full-screen, always-on-top tkinter window that starts hidden.
* Render one of two response modes:
    - blur  : a heavily blurred copy of the flagged frame covers the screen,
              with a short caption.
    - block : a solid opaque panel with a short explanatory message.
* Expose show_blur() / show_block() / hide() / stop() that are safe to call
  from ANY thread — the capture + detection loop runs in a worker thread and
  drives the overlay from there.
* Keep all real Tk calls on the main thread (required by Tk, and strictly
  enforced on macOS) by queueing commands and draining them with root.after().
* A visible "Dismiss" button (and Esc) hides the overlay and notifies the
  controller via the on_dismiss callback.
* Exclude the overlay window from screen capture via
  capture_exclusion.exclude_from_capture() (macOS sharingType none / Windows
  WDA_EXCLUDEFROMCAPTURE), so Function 1 keeps seeing the content underneath.
  If that is unavailable, capture_excluded stays False and the monitor uses the
  "pause" fallback, which needs hide_and_wait() / restore() below.

Dependencies
------------
    tkinter (Python standard library — no extra install)
    Pillow>=10.3.0  (ImageFilter for the blur, ImageTk to display it)
    capture_exclusion.py — per-platform capture exclusion (ctypes, no extra install)
    macos_window.py      — macOS keyboard focus for Esc (imported only on macOS)
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import tkinter as tk
from typing import Any, Callable

from PIL import Image, ImageFilter, ImageTk

try:
    from .capture_exclusion import exclude_from_capture
except ImportError:
    from capture_exclusion import exclude_from_capture


def _macwin():
    """macos_window module — imported lazily and only on macOS."""
    try:
        from . import macos_window
    except ImportError:
        import macos_window
    return macos_window


# ---------------------------------------------------------------------------
# Blur helper
# ---------------------------------------------------------------------------

#: Downscale factor applied before blurring. Blurring a tiny image and
#: upscaling it is far cheaper than a large-radius blur at full resolution.
BLUR_DOWNSCALE: int = 16

#: Gaussian radius (in downscaled pixels).
BLUR_RADIUS: float = 2.0


def blur_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """
    Return a heavily blurred copy of *image*, resized to *size*.

    Safe to call from any thread (pure Pillow, no Tk).
    """
    small = image.convert("RGB").resize(
        (max(1, size[0] // BLUR_DOWNSCALE), max(1, size[1] // BLUR_DOWNSCALE)),
        Image.Resampling.BILINEAR,
    )
    small = small.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    return small.resize(size, Image.Resampling.BILINEAR)


# ---------------------------------------------------------------------------
# Overlay window
# ---------------------------------------------------------------------------

#: How often (ms) the Tk main loop checks for queued commands.
POLL_INTERVAL_MS: int = 50

BLOCK_BG = "#111111"
TEXT_FG = "#ffffff"
WINDOW_TITLE = "Component 1 — Response Overlay"

log = logging.getLogger(__name__)


class ResponseOverlay:
    """
    Full-screen, always-on-top overlay window with blur and block modes.

    Usage
    -----
    The Tk main loop must run on the main thread; drive the overlay from a
    worker thread::

        overlay = ResponseOverlay()
        threading.Thread(target=worker, args=(overlay,), daemon=True).start()
        overlay.run()               # blocks until overlay.stop()

    where ``worker`` calls ``overlay.show_blur(img)`` / ``overlay.show_block()``
    / ``overlay.hide()`` as needed. The Dismiss button (or Esc) hides the
    overlay and calls *on_dismiss* on the Tk thread.
    """

    def __init__(self, on_dismiss: Callable[[], None] | None = None,
                 exclude_from_capture: bool = True) -> None:
        self._commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._visible = False
        self._stopped = False
        self._mode: str | None = None
        self._peeking = False                       # hidden by hide_and_wait(), mode kept
        self._photo: ImageTk.PhotoImage | None = None   # keep a reference or Tk drops it
        self.on_dismiss = on_dismiss
        self._want_exclusion = exclude_from_capture
        self.capture_excluded = False               # True once capture exclusion is active
        self.exclusion_checked = False              # True after the first attempt (first show)

        self.root = tk.Tk()
        self.root.withdraw()                         # start hidden
        self.root.title(WINDOW_TITLE)
        self.root.configure(bg=BLOCK_BG)
        self.root.overrideredirect(True)             # no title bar / borders

        # Cover the primary screen (logical points, not Retina pixels).
        self.screen_size = (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        self.root.geometry(f"{self.screen_size[0]}x{self.screen_size[1]}+0+0")
        self.root.attributes("-topmost", True)

        # One label renders both modes: image + caption (blur) or text only (block).
        self._label = tk.Label(
            self.root, bg=BLOCK_BG, fg=TEXT_FG, compound="center",
            font=("Helvetica", 28, "bold"), justify="center", wraplength=900,
        )
        self._label.pack(fill="both", expand=True)

        self.dismiss_button = tk.Button(self.root, text="Dismiss  (Esc)", command=self._dismiss,
                                        font=("Helvetica", 18), padx=18, pady=8)
        self.dismiss_button.place(relx=0.5, rely=0.88, anchor="center")

        self.root.bind("<Escape>", lambda _e: self._dismiss())
        self.root.after(POLL_INTERVAL_MS, self._poll)

    # -- thread-safe public API ---------------------------------------------

    def show_blur(self, image: Image.Image, caption: str = "",
                  on_visible: Callable[[float], None] | None = None) -> None:
        """
        Cover the screen with a blurred copy of *image*. Safe from any thread.
        *on_visible(epoch_s)* is called on the Tk thread once the window is shown.
        """
        self._commands.put(("blur", (blur_image(image, self.screen_size), caption, on_visible)))

    def show_block(self, message: str = "Content blocked",
                   on_visible: Callable[[float], None] | None = None) -> None:
        """Cover the screen with a solid panel and *message*. Safe from any thread."""
        self._commands.put(("block", (message, on_visible)))

    def hide(self) -> None:
        """Request the overlay be hidden. Safe to call from any thread."""
        self._commands.put(("hide", None))

    def stop(self) -> None:
        """Request the overlay close and run() return. Safe from any thread."""
        self._commands.put(("stop", None))

    def hide_and_wait(self, timeout: float = 2.0, settle_s: float = 0.12) -> bool:
        """
        Fallback "peek": withdraw the window WITHOUT forgetting its mode, and block
        (worker thread) until it is off screen, so one clean frame can be captured.
        Pair with restore(). Returns False if the Tk thread did not respond.
        """
        done = threading.Event()
        self._commands.put(("peek_hide", done))
        ok = done.wait(timeout)
        time.sleep(settle_s)                         # let the window server redraw
        return ok

    def restore(self) -> None:
        """Re-show the overlay hidden by hide_and_wait(), same mode and content."""
        self._commands.put(("restore", None))

    @property
    def is_visible(self) -> bool:
        """True once a show request has been applied on the Tk thread."""
        return self._visible

    @property
    def mode(self) -> str | None:
        """``"blur"``, ``"block"`` or ``None`` (hidden)."""
        return self._mode

    def run(self) -> None:
        """Enter the Tk main loop. Must be called from the main thread."""
        self.root.mainloop()

    # -- Tk-thread internals ------------------------------------------------

    def process_pending(self) -> None:
        """Apply every queued command. Must run on the Tk (main) thread."""
        while True:
            try:
                cmd, arg = self._commands.get_nowait()
            except queue.Empty:
                return

            if cmd == "blur":
                blurred, caption, on_visible = arg
                self._photo = ImageTk.PhotoImage(blurred, master=self.root)
                self._label.configure(image=self._photo, text=caption)
                self._present("blur", on_visible)
            elif cmd == "block":
                message, on_visible = arg
                self._photo = None
                self._label.configure(image="", text=message)
                self._present("block", on_visible)
            elif cmd == "hide" and (self._visible or self._peeking):
                self.root.withdraw()
                self._visible = self._peeking = False
                self._mode = None
            elif cmd == "peek_hide":
                if self._visible:
                    self.root.withdraw()
                    self.root.update()
                    self._visible, self._peeking = False, True
                arg.set()
            elif cmd == "restore" and self._peeking:
                self._peeking = False
                self._present(self._mode)
            elif cmd == "stop":
                self._visible = False
                self._mode = None
                self._stopped = True
                self.root.quit()
                self.root.destroy()
                return

    def _present(self, mode: str, on_visible: Callable[[float], None] | None = None) -> None:
        self._mode = mode
        if not self._visible:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)   # re-assert after deiconify
            if self._want_exclusion and not self.exclusion_checked:
                self.capture_excluded = exclude_from_capture(self.root)
                self.exclusion_checked = True
            self.take_keyboard_focus()               # so Esc reaches us
            self._visible = True
        self.root.update_idletasks()
        if on_visible is not None:
            on_visible(time.time())

    def _dismiss(self) -> None:
        """Dismiss button / Esc (Tk thread): hide now, then tell the controller."""
        if not self._visible:
            return
        self.root.withdraw()
        self._visible, self._mode = False, None
        if self.on_dismiss is not None:
            self.on_dismiss()

    # -- keyboard focus (so Esc reaches the borderless window) ---------------

    def ns_window(self):
        """This overlay's NSWindow pointer on macOS, else None."""
        if sys.platform != "darwin":
            return None
        self.root.update_idletasks()
        return _macwin().find_window(WINDOW_TITLE)

    def take_keyboard_focus(self) -> bool:
        """
        Bring the window to the front with keyboard focus. Tk's focus_force is
        enough on Windows/Linux; on macOS a borderless window must also be
        allowed to become the key window (verified: Esc then works).
        """
        self.root.focus_force()
        if sys.platform != "darwin":
            return True
        macwin, win = _macwin(), self.ns_window()
        macwin.allow_key_focus(win)
        return macwin.make_key(win)

    def _poll(self) -> None:
        self.process_pending()
        if not self._stopped:
            self.root.after(POLL_INTERVAL_MS, self._poll)


# ---------------------------------------------------------------------------
# CLI entry-point: python overlay.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from component1_screen_monitoring.src.capture import capture_frame

    # Drive the overlay from a worker thread, exactly as the pipeline will.
    def _demo(overlay: ResponseOverlay, seconds: float = 2.0) -> None:
        frame = capture_frame(monitor_index=1)
        print("[overlay demo] blur mode")
        overlay.show_blur(frame.image, caption="Blurred — demo")
        time.sleep(seconds)
        overlay.hide()
        time.sleep(1.0)
        print("[overlay demo] block mode")
        overlay.show_block("Content blocked — demo")
        time.sleep(seconds)
        overlay.hide()
        time.sleep(0.5)
        print("[overlay demo] done — stopping")
        overlay.stop()

    _overlay = ResponseOverlay()
    threading.Thread(target=_demo, args=(_overlay,), daemon=True).start()
    _overlay.run()
