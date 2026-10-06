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

Dependencies
------------
    tkinter (Python standard library — no extra install)
    Pillow>=10.3.0  (ImageFilter for the blur, ImageTk to display it)
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from typing import Any

from PIL import Image, ImageFilter, ImageTk


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
    / ``overlay.hide()`` as needed. Pressing Esc while the overlay is visible
    hides it (developer safety hatch).
    """

    def __init__(self) -> None:
        self._commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._visible = False
        self._stopped = False
        self._mode: str | None = None
        self._photo: ImageTk.PhotoImage | None = None   # keep a reference or Tk drops it

        self.root = tk.Tk()
        self.root.withdraw()                         # start hidden
        self.root.title("Component 1 — Response Overlay")
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

        self.root.bind("<Escape>", lambda _e: self.hide())
        self.root.after(POLL_INTERVAL_MS, self._poll)

    # -- thread-safe public API ---------------------------------------------

    def show_blur(self, image: Image.Image, caption: str = "") -> None:
        """Cover the screen with a blurred copy of *image*. Safe from any thread."""
        self._commands.put(("blur", (blur_image(image, self.screen_size), caption)))

    def show_block(self, message: str = "Content blocked") -> None:
        """Cover the screen with a solid panel and *message*. Safe from any thread."""
        self._commands.put(("block", message))

    def hide(self) -> None:
        """Request the overlay be hidden. Safe to call from any thread."""
        self._commands.put(("hide", None))

    def stop(self) -> None:
        """Request the overlay close and run() return. Safe from any thread."""
        self._commands.put(("stop", None))

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
                blurred, caption = arg
                self._photo = ImageTk.PhotoImage(blurred, master=self.root)
                self._label.configure(image=self._photo, text=caption)
                self._present("blur")
            elif cmd == "block":
                self._photo = None
                self._label.configure(image="", text=arg)
                self._present("block")
            elif cmd == "hide" and self._visible:
                self.root.withdraw()
                self._visible = False
                self._mode = None
            elif cmd == "stop":
                self._visible = False
                self._mode = None
                self._stopped = True
                self.root.quit()
                self.root.destroy()
                return

    def _present(self, mode: str) -> None:
        self._mode = mode
        if not self._visible:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)   # re-assert after deiconify
            self.root.focus_force()                  # so Esc reaches us
            self._visible = True

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
