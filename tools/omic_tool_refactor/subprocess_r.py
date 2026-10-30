"""Invoke R without interpreting process failures as successful analysis."""
from pathlib import Path
import subprocess


def run_r_script(filename: str, args: list | None = None, *, timeout: float = 3600) -> str:
    """Return stdout; propagate nonzero exits, timeouts and missing executables."""
    script = Path(filename)
    if not script.is_file():
        raise FileNotFoundError(f"R script not found: {script}")
    command = ["Rscript", str(script), *[str(value) for value in (args or [])]]
    print(f"Running R script: {script}")
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=True,
        )
    except subprocess.CalledProcessError as error:
        for output in (error.stdout, error.stderr):
            if output:
                print(output.rstrip())
        raise
    if result.stderr:
        print(result.stderr.rstrip())
    return result.stdout
