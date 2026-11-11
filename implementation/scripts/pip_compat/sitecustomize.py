"""File-only EXDEV fallback for pip build subprocesses on virtual filesystems."""
import errno
import os
import shutil

if os.environ.get("H3_PIP_FILE_RENAME_FALLBACK") == "1":
    def _file_move(operation):
        def move(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
            try:
                return operation(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
            except OSError as error:
                if (error.errno != errno.EXDEV or src_dir_fd is not None
                        or dst_dir_fd is not None or not os.path.isfile(src)
                        or os.path.islink(src)):
                    raise
                shutil.copy2(src, dst)
                os.unlink(src)
        return move

    os.rename = _file_move(os.rename)
    os.replace = _file_move(os.replace)
