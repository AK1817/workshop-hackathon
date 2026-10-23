"""Load the first 50 rows of every data file in your Kaggle datasets.

Usage (from load_data/):
    uv add pandas openpyxl
    uv run .\\extract_samples.py

Saves each sample to load_50/<dataset>__<file>.csv (in your current directory) and prints its columns.
In other code:  from extract_samples import load_samples;  dfs = load_samples()
"""
from pathlib import Path
import kagglehub
import pandas as pd

N_ROWS = 50
OUT_DIR = Path.cwd() / "load_50"  # folder in the directory you run the script from

# kagglehub uses its cache, so nothing is downloaded again
DATASETS = {
    "openfoodfacts": "openfoodfacts/world-food-facts",
    "usda_fdc": "stoicstatic/fooddata-central-nutrition-and-component-data",
    "fruit_veg_prices": "vstacknocopyright/fruit-and-vegetable-prices",
}

DATA_EXT = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"}


def read_head(path: Path, n: int = N_ROWS) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=n)
    if ext == ".json":
        try:
            return pd.read_json(path, lines=True, nrows=n)
        except ValueError:
            return pd.read_json(path).head(n)
    # .tsv is tab-separated; otherwise sep=None lets pandas detect , ; or tab
    sep = "\t" if ext == ".tsv" else None
    return pd.read_csv(
        path,
        sep=sep,
        nrows=n,  # reads only 50 rows, even from the very large Open Food Facts file
        engine="python",
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="skip",
    )


def load_samples(save: bool = True) -> dict[str, pd.DataFrame]:
    samples: dict[str, pd.DataFrame] = {}
    if save:
        OUT_DIR.mkdir(exist_ok=True)

    for name, handle in DATASETS.items():
        root = Path(kagglehub.dataset_download(handle))
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in DATA_EXT]
        if not files:
            print(f"[{name}] no data files found in {root}")
            continue

        for f in sorted(files):
            key = f"{name}__{f.stem}"
            try:
                df = read_head(f)
            except Exception as e:  # keep going if one file is odd
                print(f"[{key}] skipped: {e}")
                continue
            samples[key] = df
            print(f"\n[{key}] {f.name}: {len(df)} rows x {df.shape[1]} cols")
            print("  columns:", ", ".join(map(str, df.columns[:15])), "..." if df.shape[1] > 15 else "")
            if save:
                df.to_csv(OUT_DIR / f"{key}.csv", index=False)

    if save:
        print(f"\nSaved {len(samples)} samples to {OUT_DIR}")
    return samples


if __name__ == "__main__":
    load_samples()