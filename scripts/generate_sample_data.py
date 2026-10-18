"""Regenera los CSV de data/sample/ (datos sintéticos y reproducibles)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecaster.samples import generate_cafe_example, sample_csv_bytes  # noqa: E402

folder = Path(__file__).resolve().parents[1] / "data" / "sample"
for name, content in (
    ("ventas_ejemplo.csv", sample_csv_bytes()),
    ("ejemplo_cafeteria.csv", generate_cafe_example()),
):
    (folder / name).write_bytes(content)
    print(f"Escrito {folder / name}")
