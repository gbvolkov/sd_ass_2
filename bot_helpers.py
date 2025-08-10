import time
import contextlib
import threading

# Keeps Telegram "typing..." status alive by re-sending it every few seconds
class TypingKeeper:
    def __init__(self, bot, chat_id: int, interval: float = 4.0):
        self.bot = bot
        self.chat_id = chat_id
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    def stop(self):
        self._stop.set()
        if self._thread:
            with contextlib.suppress(Exception):
                self._thread.join(timeout=0.2)
    def _run(self):
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self.bot.send_chat_action(self.chat_id, "typing")
            time.sleep(self.interval)
