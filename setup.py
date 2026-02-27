import os
import sys
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
VENV_PY = VENV_DIR / "bin" / "python"
REQ_FILE = ROOT / "requirements.txt"


def run(cmd, *, check=True):
    print(f"+ {' '.join(map(str, cmd))}")
    return subprocess.run(list(map(str, cmd)), check=check)


def ensure_venv():
    if VENV_PY.exists():
        print("- .venv already exists")
        return

    print("- Creating .venv virtual environment")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])

    # Upgrade pip tooling inside the venv
    run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])


def reexec_into_venv():
    """If we're not running the venv's python, restart this script using it."""
    current = Path(sys.executable).resolve()
    expected = VENV_PY.resolve()

    print(f"- Current python:  {current}")
    print(f"- Expected python: {expected}")

    if current != expected:
        print("- Re-launching inside .venv")
        os.execv(str(expected), [str(expected), *sys.argv])
    else:
        print("- Running inside the correct virtual environment")


def install_requirements():
    if REQ_FILE.exists():
        print("- Installing requirements.txt")
        run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(REQ_FILE)])
    else:
        print("- No requirements.txt found, skipping pip install")


def install_with_poetry(optional=True):
    """
    Optional: if you use Poetry (pyproject.toml/poetry.lock),
    install via Poetry instead of (or in addition to) requirements.txt.
    """
    if not optional:
        return

    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        print("- No pyproject.toml found, skipping Poetry")
        return

    # Make Poetry use the in-project .venv if possible
    # This avoids Poetry creating a separate venv somewhere else.
    try:
        run(["poetry", "config", "virtualenvs.in-project", "true"])
        run(["poetry", "install"])
    except FileNotFoundError as e:
        print("- Poetry not installed; skipping Poetry install")


def load_env():
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("- python-dotenv not installed; skipping .env loading")
        return

    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print("- Loaded environment variables from .env")
    else:
        print("- No .env file found")


def main():
    ensure_venv()
    reexec_into_venv()  # after this line, we're definitely in .venv

    install_requirements()
    # If you use Poetry, consider choosing ONE source of truth:
    # - either requirements.txt OR poetry.lock (recommended)
    install_with_poetry(optional=True)  # set True if you want poetry install too

    load_env()
    print("✅ Setup complete.")


if __name__ == "__main__":
    main()