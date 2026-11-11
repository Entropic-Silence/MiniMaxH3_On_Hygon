import os,sys,mimetypes
from pathlib import Path
compat = str(Path(__file__).resolve().parent / "pip_compat")
os.environ["H3_PIP_FILE_RENAME_FALLBACK"] = "1"
os.environ["PYTHONPATH"] = compat + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, compat)
import runpy
runpy.run_path(str(Path(compat) / "sitecustomize.py"))
os.environ.setdefault("PIP_CONFIG_FILE",os.devnull)
mimetypes.knownfiles=[]
mimetypes.init()
import pip._internal.network.session as session
session.user_agent=lambda: "minimax-h3-build"
from pip._internal.cli.main import main
raise SystemExit(main(sys.argv[1:]))
