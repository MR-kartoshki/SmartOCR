# smartocr

Fast local OCR from the command line using RapidOCR and ONNX Runtime.

```console
$ smartocr screenshot.png
extracted text...
```

## Usage

```text
smartocr IMAGE
smartocr IMAGE -l de
smartocr IMAGE -l ru
smartocr IMAGE -o output.txt
smartocr IMAGE --cpu
smartocr IMAGE --debug
```

English and German use PP-OCRv6. Russian currently uses PP-OCRv5.
Models that are not bundled with RapidOCR are downloaded on first use to
`$XDG_CACHE_HOME/smartocr/models` (or `~/.cache/smartocr/models`).

smartocr selects an available ONNX Runtime backend in this order: CUDA, ROCm,
then CPU. If GPU inference fails, it tries the next available backend and
ultimately falls back to CPU. `--cpu` always forces CPU inference.

## Arch Linux

Arch packaging files live in `packaging/arch/`. The package depends on Arch's
virtual `python-onnxruntime` package. During installation, pacman will let you
choose one of its providers:

- `python-onnxruntime-cuda` for NVIDIA GPUs
- `python-onnxruntime-rocm` for AMD GPUs (experimental and untested)
- `python-onnxruntime-cpu` for CPU-only systems

The GPU runtime packages supply their accelerator dependencies, so CUDA,
cuDNN, and ROCm do not need to be listed as direct smartocr dependencies.

`python-rapidocr` is currently an AUR dependency. PDF input additionally uses
the official-repository `python-pymupdf` package.

Install through the AUR, or build the provided `PKGBUILD` with `makepkg`.

## License

MIT
