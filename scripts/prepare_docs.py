"""Prepare generated assets used by the MkDocs site."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "docs" / "_generated"

FILES = {
    ROOT / "notebooks" / "step_by_step_analysis.ipynb": GENERATED / "quickstart.ipynb",
    ROOT / "src" / "ui" / "assets" / "poremind_logo.png": GENERATED / "poremind_logo.png",
    ROOT / "dlmagjam.png": GENERATED / "dlmagjam.png",
    ROOT / "poremind_ui_demo.png": GENERATED / "poremind_ui_demo.png",
}


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    for source, destination in FILES.items():
        if not source.is_file():
            raise FileNotFoundError(f"Documentation input is missing: {source}")
        shutil.copy2(source, destination)
        print(f"Prepared {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
