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
- `python-onnxruntime-opt-cuda` for NVIDIA GPUs with AVX2 CPU optimizations
- `python-onnxruntime-rocm` for AMD GPUs (experimental and untested)
- `python-onnxruntime-opt-rocm` for AMD GPUs with AVX2 CPU optimizations (experimental and untested)
- `python-onnxruntime-cpu` for CPU-only systems

The `-opt` variants use the same CUDA or ROCm backend as their regular
counterparts, but compile ONNX Runtime's CPU code for AVX2-capable processors, potentially increasing performance.

The smartocr AUR package is not published yet because new AUR account
registration is temporarily closed.

Until it is published, build the provided `PKGBUILD` with `makepkg`.

## License

MIT
