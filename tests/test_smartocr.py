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
    def test_normalizes_language_names_and_aliases(self):
        self.assertEqual(smartocr.normalize_lang("French"), "fr")
        self.assertEqual(smartocr.normalize_lang("JPN"), "ja")
        self.assertEqual(smartocr.normalize_lang("zh-Hant"), "zh-tw")

    def test_selects_model_for_language(self):
        self.assertEqual(smartocr.model_version_for_lang("fr"), "PP-OCRv6")
        self.assertEqual(smartocr.rec_lang_for_lang("zh"), "ch")
        self.assertEqual(smartocr.model_version_for_lang("ur"), "PP-OCRv5")
        self.assertEqual(smartocr.rec_lang_for_lang("ur"), "arabic")
        self.assertEqual(smartocr.model_version_for_lang("kn"), "PP-OCRv4")
        self.assertEqual(smartocr.rec_lang_for_lang("kn"), "ka")

    def test_all_languages_have_valid_model_routes(self):
        v5_rec_langs = {
            "arabic",
            "cyrillic",
            "devanagari",
            "el",
            "korean",
            "ta",
            "te",
            "th",
        }

        for lang in smartocr.LANGUAGE_NAMES:
            version = smartocr.model_version_for_lang(lang)
            rec_lang = smartocr.rec_lang_for_lang(lang)

            if version == "PP-OCRv4":
                self.assertEqual(rec_lang, "ka")
            elif version == "PP-OCRv5":
                self.assertIn(rec_lang, v5_rec_langs)
            else:
                self.assertEqual(version, "PP-OCRv6")

    def test_engine_uses_the_language_model_route(self):
        created = []

        class Values:
            CH = "ch"
            MEDIUM = "medium"
            MOBILE = "mobile"
            SERVER = "server"
            PPOCRV4 = "PP-OCRv4"
            PPOCRV5 = "PP-OCRv5"
            PPOCRV6 = "PP-OCRv6"

        class LangRec:
            def __new__(cls, value):
                return value

        class RapidOCR:
            def __init__(self, params):
                created.append(params)

        rapidocr = types.ModuleType("rapidocr")
        rapidocr.LangDet = Values
        rapidocr.LangRec = LangRec
        rapidocr.ModelType = Values
        rapidocr.OCRVersion = Values
        rapidocr.RapidOCR = RapidOCR

        onnxruntime = types.ModuleType("onnxruntime")

        with (
            mock.patch.dict(
                sys.modules,
                {"onnxruntime": onnxruntime, "rapidocr": rapidocr},
            ),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            smartocr.create_engine("fr", "CPUExecutionProvider", debug=True)
            smartocr.create_engine("ur", "CPUExecutionProvider", debug=True)
            smartocr.create_engine("kn", "CPUExecutionProvider", debug=True)

        self.assertEqual(created[0]["Rec.ocr_version"], "PP-OCRv6")
        self.assertEqual(created[0]["Rec.lang_type"], "fr")
        self.assertEqual(created[1]["Rec.ocr_version"], "PP-OCRv5")
        self.assertEqual(created[1]["Rec.lang_type"], "arabic")
        self.assertEqual(created[2]["Rec.ocr_version"], "PP-OCRv4")
        self.assertEqual(created[2]["Rec.lang_type"], "ka")

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

    def test_reconstructs_right_to_left_text(self):
        result = smartocr.extract_result(
            Output(
                texts=("واحد", "اثنان", "ثلاثة"),
                scores=(0.9, 0.9, 0.9),
                boxes=[
                    [[0, 0], [30, 0], [30, 10], [0, 10]],
                    [[35, 0], [65, 0], [65, 10], [35, 10]],
                    [[70, 0], [100, 0], [100, 10], [70, 10]],
                ],
            ),
            rtl=True,
        )

        self.assertEqual(result["text"], "ثلاثة اثنان واحد")

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
