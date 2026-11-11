import os,sys,mimetypes
os.environ.setdefault("PIP_CONFIG_FILE",os.devnull)
mimetypes.knownfiles=[]
mimetypes.init()
import pip._internal.network.session as session
session.user_agent=lambda: "minimax-h3-build"
from pip._internal.cli.main import main
raise SystemExit(main(sys.argv[1:]))
