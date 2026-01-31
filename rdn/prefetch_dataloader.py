"""
Prefetching wrapper for dataloaders

Wraps any dataloader and prefetches batches in a background thread,
eliminating GPU starvation from CPU-bound tokenization.
"""
import threading
import queue


class PrefetchDataLoader:
    """
    Wraps a dataloader and prefetches batches in a background thread.

    This eliminates GPU starvation by having batches ready before they're needed.
    """

    def __init__(self, dataloader, prefetch_count=3):
        """
        Args:
            dataloader: Iterator to wrap (e.g., tokenizing dataloader)
            prefetch_count: Number of batches to prefetch (default: 3)
        """
        self.dataloader = dataloader
        self.prefetch_count = prefetch_count
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.thread = None
        self.stop_event = threading.Event()
        self._start_worker()

    def _start_worker(self):
        """Start background thread that fetches batches."""
        def worker():
            try:
                for batch in self.dataloader:
                    if self.stop_event.is_set():
                        break
                    # This blocks if queue is full (desired behavior - don't prefetch too much)
                    self.queue.put(batch)
            except Exception as e:
                # Put exception in queue so main thread can raise it
                self.queue.put(e)

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch from prefetch queue."""
        try:
            batch = self.queue.get(timeout=30)  # 30s timeout to detect hangs

            # If worker thread raised exception, re-raise it here
            if isinstance(batch, Exception):
                raise batch

            return batch
        except queue.Empty:
            raise RuntimeError("Prefetch worker thread stalled or died")

    def __del__(self):
        """Clean up background thread."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
