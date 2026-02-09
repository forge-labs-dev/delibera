"""Streamlit Cloud entry point — delegates to the main app."""

import sys
from pathlib import Path

# Ensure src/ is importable (needed on Streamlit Cloud where the
# package is not pip-installed).
_src = str(Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from delibera.web.app import main  # noqa: E402

main()
