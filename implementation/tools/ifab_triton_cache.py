"""Immutable Triton cache for private AFS mounts that reject rename(2).

Selected through Triton's TRITON_CACHE_MANAGER extension point. Completion
markers keep interrupted writes invisible; flock serializes competing writers.
Use a fresh cache directory so all published entries have completion markers.
"""
import fcntl
import os

from triton.runtime.cache import FileCacheManager


class AfsCacheManager(FileCacheManager):
    def has_file(self, filename):
        path = self._make_path(filename)
        return os.path.isfile(path + ".ready") and os.path.isfile(path)

    def put(self, data, filename, binary=True):
        if not self.cache_dir or self.lock_path is None:
            raise RuntimeError("Cache directory is not writable")
        path = self._make_path(filename)
        with open(self.lock_path + ".afs", "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # Triton keys include compiler inputs. Published entries are immutable,
            # so a path returned to another process is never overwritten.
            if self.has_file(filename):
                return path
            if not isinstance(data, bytes):
                data = str(data).encode("utf-8")
            with open(path, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            with open(path + ".ready", "wb") as marker:
                marker.flush()
                os.fsync(marker.fileno())
        return path
