"""Build data/mnist.js from the four original MNIST IDX files.

Usage:  python tools/make-mnist-bundle.py <folder-with-idx-files>

Each IDX file is gzip-compressed and stored as base64 inside one script, so the
site works both from a web server and when index.html is opened directly
(browsers block fetch() on file:// pages, but they still run <script> tags).
"""
import base64
import gzip
import pathlib
import sys

FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]

source = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
target = pathlib.Path(__file__).resolve().parent.parent / "data" / "mnist.js"

lines = ["globalThis.MNIST_GZ = {"]
for name in FILES:
    raw = (source / name).read_bytes()
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    lines.append(f'"{name}": "{base64.b64encode(packed).decode("ascii")}",')
    print(f"{name}: {len(raw):,} -> {len(packed):,} bytes")
lines.append("};")
target.write_text("\n".join(lines) + "\n", encoding="ascii")
print(f"wrote {target} ({target.stat().st_size:,} bytes)")
