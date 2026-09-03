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
smartocr IMAGE -l ja
smartocr IMAGE -l ar
smartocr --list-languages
smartocr IMAGE -o output.txt
smartocr IMAGE --cpu
smartocr IMAGE --debug
```

Supported languages:

- European and Latin-script: Azerbaijani (`az`), Czech (`cs`), Danish (`da`),
  Dutch (`nl`), English (`en`), Finnish (`fi`), French (`fr`), German (`de`),
  Hungarian (`hu`), Indonesian (`id`), Italian (`it`), Malay (`ms`), Norwegian
  (`no`), Polish (`pl`), Portuguese (`pt`), Romanian (`ro`), Serbian Latin
  (`sr-latn`), Spanish (`es`), Swedish (`sv`), Swahili (`sw`), Tagalog (`tl`),
  Turkish (`tr`), Uzbek (`uz`), Vietnamese (`vi`), and Kurdish Latin (`ku-latn`)
- Cyrillic: Belarusian (`be`), Bulgarian (`bg`), Kazakh (`kk`), Kyrgyz (`ky`),
  Macedonian (`mk`), Mongolian (`mn`), Russian (`ru`), Serbian (`sr`), Tajik
  (`tg`), and Ukrainian (`uk`)
- Arabic script: Arabic (`ar`), Persian (`fa`), Kurdish (`ku`), Pashto (`ps`),
  Sindhi (`sd`), Uyghur (`ug`), and Urdu (`ur`)
- South Asian: Hindi (`hi`), Marathi (`mr`), Nepali (`ne`), Bhojpuri (`bho`),
  Kannada (`kn`), Tamil (`ta`), and Telugu (`te`)
- East Asian: Simplified Chinese (`zh`), Traditional Chinese (`zh-tw`), Japanese
  (`ja`), and Korean (`ko`)
- Greek (`el`) and Thai (`th`)

Common English names and ISO 639-2 codes such as `english`, `eng`, `german`,
`deu`, and `jpn` are also accepted.

Latin-script, Chinese, and Japanese languages use PP-OCRv6. Most other scripts
use PP-OCRv5; Kannada uses PP-OCRv4 because it is the latest compatible model.
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
