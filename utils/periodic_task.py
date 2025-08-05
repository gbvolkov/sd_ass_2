import threading  # Ensure threading is imported
import time

import config

# -----------------------------
# Periodic Knowledge Base Update
# -----------------------------
class PeriodicTask:
    def __init__(self, job_func):
        self.lock = threading.Lock()
        self.running = True
        self.job = job_func
        
    def periodic_function(self):
        while self.running:
            # Try to acquire the lock without blocking
            if self.lock.acquire(blocking=False):
                try:
                    self.job()
                finally:
                    # Always release the lock, even if an exception occurs
                    self.lock.release()
            time.sleep(int(config.UPD_TIMEOUT))  # Wait for a timeout before the next update
    
    def start(self):
        self.thread = threading.Thread(
            target=self.periodic_function,
            daemon=True
        )
        self.running = True
        self.thread.start()

    def resume(self):
        self.lock.acquire(blocking=True)
        self.running = True
        self.lock.release()

    def pause(self):
        self.lock.acquire(blocking=True)
        self.running = False
        self.lock.release()

    def stop(self):
        self.running = False
        self.thread.join()
