from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cytof_archetypes.cli import train_cli


if __name__ == "__main__":
    train_cli()
