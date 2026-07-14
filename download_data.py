"""
download_data.py
----------------
Download the PBMC 3k reference dataset used in this benchmark.

The dataset (2,638 cells x 1,838 highly variable genes, 8 annotated cell types)
is fetched as a preprocessed .h5ad file and saved to data/pbmc3k.h5ad.

Usage:
    python download_data.py
"""
import os
import urllib.request

URL = "https://github.com/chanzuckerberg/cellxgene/raw/main/example-dataset/pbmc3k.h5ad"
OUT = "data/pbmc3k.h5ad"


def main():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(OUT):
        print(f"{OUT} already exists — skipping download.")
        return
    print(f"Downloading PBMC 3k dataset to {OUT} ...")
    urllib.request.urlretrieve(URL, OUT)
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"Done. ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
