#!/usr/bin/env python3

import argparse
import contextlib
import os
import re
import statistics
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

__version__ = "1.0.0"

LANG_ALIASES = {
    "eng": "en",
    "english": "en",
    "en": "en",
    "deu": "de",
    "german": "de",
    "de": "de",
    "rus": "ru",
    "russian": "ru",
    "ru": "ru",
}


def die(message, code=1):
    print(f"smartocr: {message}", file=sys.stderr)
    raise SystemExit(code)


def cleanup_text(text):
    lines = text.splitlines()
    cleaned = []

    decoration_prefix = re.compile(
        r"^[^A-Za-z0-9\s]{3,}\s+"
        r"(?=[A-Za-z][A-Za-z0-9 _./()+-]{0,30}:)"
    )

    for line in lines:
        line = line.replace(chr(0x2018), "'")
        line = line.replace(chr(0x2019), "'")
        line = line.replace(chr(0x201C), '"')
        line = line.replace(chr(0x201D), '"')

        if line.startswith("» "):
            line = "❯ " + line[2:]

        # Paddle sometimes merges terminal art into adjacent "Label: value" text.
        line = decoration_prefix.sub("", line)

        if line.strip() == "|":
            continue

        cleaned.append(line.rstrip())

    return "\n".join(cleaned).strip()


def normalize_lang(value):
    key = value.strip().lower()

    if key not in LANG_ALIASES:
        die(
            f"unsupported language '{value}'. "
            "Use en/eng, de/deu, or ru/rus."
        )

    return LANG_ALIASES[key]


def model_version_for_lang(lang):
    # PP-OCRv6 has no Russian recognition model yet.
    return "PP-OCRv5" if lang == "ru" else "PP-OCRv6"


@contextlib.contextmanager
def silence_native_output(enabled=True):
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()

    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)

            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()

    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def create_engine(lang, device, debug=False):
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        die(
            "PaddleOCR could not be imported:\n"
            f"{exc}"
        )

    version = model_version_for_lang(lang)

    if debug:
        print(
            f"[engine] model={version} lang={lang} "
            f"engine=onnxruntime device={device}",
            file=sys.stderr,
        )

    kwargs = dict(
        lang=lang,
        ocr_version=version,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="onnxruntime",
        device=device,
    )

    with silence_native_output(enabled=not debug):
        return PaddleOCR(**kwargs)


def get_payload(result):
    data = result.json

    if callable(data):
        data = data()

    if not isinstance(data, dict):
        return {}

    if isinstance(data.get("res"), dict):
        return data["res"]

    return data


def as_list(value):
    if value is None:
        return []

    if hasattr(value, "tolist"):
        return value.tolist()

    return list(value)


def box_bounds(box):
    box = as_list(box)

    if len(box) == 4 and all(
        isinstance(v, (int, float)) for v in box
    ):
        x1, y1, x2, y2 = box
        return float(x1), float(y1), float(x2), float(y2)

    points = [as_list(point) for point in box]
    xs = [float(point[0]) for point in points if len(point) >= 2]
    ys = [float(point[1]) for point in points if len(point) >= 2]

    if not xs or not ys:
        return 0.0, 0.0, 0.0, 0.0

    return min(xs), min(ys), max(xs), max(ys)


def reconstruct_text(texts, boxes):
    if not texts:
        return ""

    if not boxes or len(boxes) != len(texts):
        return "\n".join(
            str(text)
            for text in texts
            if str(text).strip()
        )

    items = []

    for text, box in zip(texts, boxes):
        text = str(text).strip()

        if not text:
            continue

        x1, y1, x2, y2 = box_bounds(box)
        height = max(1.0, y2 - y1)

        items.append({
            "text": text,
            "x1": x1,
            "x2": x2,
            "yc": (y1 + y2) / 2.0,
            "height": height,
        })

    if not items:
        return ""

    items.sort(key=lambda item: (item["yc"], item["x1"]))

    median_height = statistics.median(
        item["height"] for item in items
    )
    line_tolerance = max(4.0, median_height * 0.45)

    lines = []

    for item in items:
        best_line = None
        best_delta = None

        for line in reversed(lines[-3:]):
            delta = abs(item["yc"] - line["yc"])

            if delta <= line_tolerance:
                if best_delta is None or delta < best_delta:
                    best_line = line
                    best_delta = delta

        if best_line is None:
            lines.append({
                "yc": item["yc"],
                "items": [item],
            })
        else:
            best_line["items"].append(item)
            best_line["yc"] = statistics.mean(
                part["yc"] for part in best_line["items"]
            )

    rendered = []

    def is_symbol_heavy(fragment):
        compact = "".join(
            char
            for char in fragment
            if not char.isspace()
        )

        if len(compact) < 3:
            return False

        alnum = sum(char.isalnum() for char in compact)
        symbols = len(compact) - alnum

        return (
            symbols >= 4
            and symbols / len(compact) >= 0.70
            and alnum <= 2
        )

    for line in lines:
        parts = sorted(
            line["items"],
            key=lambda item: item["x1"],
        )

        if len(parts) == 1:
            rendered.append(parts[0]["text"])
            continue

        line_height = statistics.median(
            part["height"] for part in parts
        )

        # Large gaps are usually separate columns that happen to share a row.
        split_gap = max(36.0, line_height * 2.75)

        segments = []
        current = [parts[0]]

        for previous, part in zip(parts, parts[1:]):
            gap = part["x1"] - previous["x2"]

            if gap >= split_gap:
                segments.append(current)
                current = [part]
            else:
                current.append(part)

        segments.append(current)

        segment_texts = [
            " ".join(
                part["text"]
                for part in segment
            ).strip()
            for segment in segments
        ]

        if len(segment_texts) > 1:
            has_real_text = any(
                not is_symbol_heavy(segment)
                and any(char.isalnum() for char in segment)
                for segment in segment_texts
            )

            if has_real_text:
                segment_texts = [
                    segment
                    for segment in segment_texts
                    if not is_symbol_heavy(segment)
                ]

        rendered.extend(
            segment
            for segment in segment_texts
            if segment
        )

    return "\n".join(
        line
        for line in rendered
        if line
    )


def extract_result(result):
    payload = get_payload(result)

    texts = as_list(payload.get("rec_texts"))
    scores = [
        float(score)
        for score in as_list(payload.get("rec_scores"))
    ]

    raw_boxes = payload.get("rec_boxes")

    if raw_boxes is None:
        raw_boxes = payload.get("rec_polys")

    boxes = as_list(raw_boxes)

    return {
        "text": reconstruct_text(texts, boxes),
        "scores": scores,
        "count": len(
            [text for text in texts if str(text).strip()]
        ),
    }


def execution_backends(force_cpu=False):
    if force_cpu:
        return [("CPUExecutionProvider", "cpu")]

    try:
        import onnxruntime

        available = set(onnxruntime.get_available_providers())
    except Exception as exc:
        die(f"ONNX Runtime provider detection failed: {exc}")

    backends = [
        ("CUDAExecutionProvider", "gpu:0"),
        ("ROCMExecutionProvider", "gpu:0"),
        ("CPUExecutionProvider", "cpu"),
    ]

    return [
        backend
        for backend in backends
        if backend[0] in available
    ]


def run_ocr(image, lang, force_cpu=False, debug=False):
    backends = execution_backends(force_cpu)

    if not backends:
        die("no supported ONNX Runtime execution provider found")

    last_error = None

    for index, (provider, device) in enumerate(backends):
        try:
            if debug:
                print(
                    f"[backend] provider={provider} device={device}",
                    file=sys.stderr,
                )

            engine = create_engine(
                lang,
                device,
                debug=debug,
            )

            if debug:
                print(
                    f"[ocr] running on {device}",
                    file=sys.stderr,
                )

            results = list(
                engine.predict(str(image))
            )

            extracted = [
                extract_result(result)
                for result in results
            ]

            text = "\n\n".join(
                item["text"]
                for item in extracted
                if item["text"]
            ).strip()

            scores = [
                score
                for item in extracted
                for score in item["scores"]
            ]

            count = sum(
                item["count"]
                for item in extracted
            )

            if not text:
                die("no text detected", 2)

            return {
                "text": text,
                "scores": scores,
                "count": count,
                "device": device,
                "provider": provider,
                "version": model_version_for_lang(lang),
            }

        except SystemExit:
            raise

        except Exception as exc:
            last_error = exc

            if device == "gpu:0" and index + 1 < len(backends):
                next_provider, next_device = backends[index + 1]

                if next_device == "cpu":
                    retry = "CPU"
                else:
                    retry = next_provider.removesuffix("ExecutionProvider")

                print(
                    "smartocr: GPU inference failed, "
                    f"retrying on {retry}.",
                    file=sys.stderr,
                )

                if debug:
                    print(
                        f"[gpu error] "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )

                continue

            break

    die(
        "OCR failed"
        + (
            f": {type(last_error).__name__}: {last_error}"
            if last_error
            else ""
        )
    )


def main():
    parser = argparse.ArgumentParser(
        prog="smartocr",
        description=(
            "GPU-first local OCR using PaddleOCR "
            "and ONNX Runtime."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "image",
        help="image or PDF to OCR",
    )

    parser.add_argument(
        "-l",
        "--lang",
        default="en",
        help=(
            "OCR language: en/eng, de/deu, ru/rus "
            "(default: en)"
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        help="write extracted text to a file",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="force CPU inference instead of GPU acceleration",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="show engine, device, and confidence information",
    )

    parser.add_argument(
        "--show-info",
        action="store_true",
        help="show final OCR engine information",
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="disable conservative punctuation cleanup",
    )

    args = parser.parse_args()

    image = Path(
        args.image
    ).expanduser().resolve()

    if not image.is_file():
        die(f"file not found: {image}")

    lang = normalize_lang(args.lang)

    result = run_ocr(
        image,
        lang,
        force_cpu=args.cpu,
        debug=args.debug,
    )

    text = result["text"]

    if not args.no_cleanup:
        text = cleanup_text(text)

    scores = result["scores"]

    mean_conf = (
        statistics.mean(scores) * 100
        if scores
        else 0.0
    )

    median_conf = (
        statistics.median(scores) * 100
        if scores
        else 0.0
    )

    if args.debug or args.show_info:
        print(
            f"[result] model={result['version']} "
            f"lang={lang} "
            f"provider={result['provider']} "
            f"device={result['device']} "
            f"lines={result['count']} "
            f"mean_conf={mean_conf:.1f}% "
            f"median_conf={median_conf:.1f}%",
            file=sys.stderr,
        )

    if args.output:
        output = Path(
            args.output
        ).expanduser()

        output.write_text(
            text + "\n",
            encoding="utf-8",
        )
    else:
        print(text)


if __name__ == "__main__":
    main()
