import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import smartocr


class Output:
    def __init__(self, texts=None, scores=None, boxes=None):
        self.txts = texts
        self.scores = scores
        self.boxes = boxes


class SmartOCRTests(unittest.TestCase):
    def test_execution_backend_priority(self):
        onnxruntime = types.ModuleType("onnxruntime")
        onnxruntime.get_available_providers = lambda: [
            "CPUExecutionProvider",
            "ROCMExecutionProvider",
            "CUDAExecutionProvider",
        ]

        with mock.patch.dict(sys.modules, {"onnxruntime": onnxruntime}):
            self.assertEqual(
                smartocr.execution_backends(),
                [
                    ("CUDAExecutionProvider", "gpu:0"),
                    ("ROCMExecutionProvider", "gpu:0"),
                    ("CPUExecutionProvider", "cpu"),
                ],
            )

    def test_force_cpu_ignores_available_gpu_providers(self):
        self.assertEqual(
            smartocr.execution_backends(force_cpu=True),
            [("CPUExecutionProvider", "cpu")],
        )

    def test_extracts_rapidocr_output(self):
        result = smartocr.extract_result(Output(
            texts=("hello", "world"),
            scores=(0.9, 0.8),
            boxes=[
                [[0, 0], [40, 0], [40, 10], [0, 10]],
                [[45, 0], [85, 0], [85, 10], [45, 10]],
            ],
        ))

        self.assertEqual(result["text"], "hello world")
        self.assertEqual(result["scores"], [0.9, 0.8])
        self.assertEqual(result["count"], 2)

    def test_empty_gpu_result_falls_back_to_rocm(self):
        attempted = []
        empty = Output()
        success = Output(
            texts=("hello",),
            scores=(0.9,),
            boxes=[[[0, 0], [40, 0], [40, 10], [0, 10]]],
        )

        def create_engine(lang, provider, debug=False):
            attempted.append(provider)
            return lambda image: (
                empty if provider == "CUDAExecutionProvider" else success
            )

        backends = [
            ("CUDAExecutionProvider", "gpu:0"),
            ("ROCMExecutionProvider", "gpu:0"),
            ("CPUExecutionProvider", "cpu"),
        ]

        with (
            mock.patch.object(smartocr, "execution_backends", return_value=backends),
            mock.patch.object(smartocr, "load_inputs", return_value=["image.png"]),
            mock.patch.object(smartocr, "create_engine", side_effect=create_engine),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = smartocr.run_ocr(Path("image.png"), "en")

        self.assertEqual(
            attempted,
            ["CUDAExecutionProvider", "ROCMExecutionProvider"],
        )
        self.assertEqual(result["provider"], "ROCMExecutionProvider")
        self.assertEqual(result["device"], "gpu:0")

    def test_cpu_no_text_exit_code_is_preserved(self):
        with (
            mock.patch.object(
                smartocr,
                "execution_backends",
                return_value=[("CPUExecutionProvider", "cpu")],
            ),
            mock.patch.object(smartocr, "load_inputs", return_value=["image.png"]),
            mock.patch.object(
                smartocr,
                "create_engine",
                return_value=lambda image: Output(),
            ),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                smartocr.run_ocr(Path("image.png"), "en")

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
