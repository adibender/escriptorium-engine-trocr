"""Conformance and behaviour of the TrOCR engine.

Like the Tesseract package, the contract is imported from eScriptorium rather than restated here.
Tests that need the Hugging Face Hub skip when it cannot be reached, so the package still tests
offline -- the spec, reference parsing and line cropping never need the network.
"""
import os
import unittest
from types import SimpleNamespace

from PIL import Image, ImageDraw

from engines.base import SOURCE_HF_REPO, LineInput, ModelValidationError
from engines.conformance import EngineConformanceMixin

from escriptorium_engine_trocr import TrOCREngine
from escriptorium_engine_trocr.engine import parse_reference

#: small enough to run on CPU inside the test run; printed, because the conformance image is printed
TEST_REFERENCE = os.environ.get("TROCR_TEST_REFERENCE", "microsoft/trocr-small-printed")


def _hub_model():
    """A registered-looking model for TEST_REFERENCE, or None when the Hub is out of reach."""
    try:
        info = TrOCREngine().validate_reference(TEST_REFERENCE)
    except ModelValidationError:
        return None
    return SimpleNamespace(name=TEST_REFERENCE.split("/")[-1], reference=info.reference)


class TrOCRConformanceTests(EngineConformanceMixin, unittest.TestCase):
    engine_class = TrOCREngine

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.recognition_model = _hub_model()
        cls.can_run_inference = cls.recognition_model is not None


class ReferenceParsingTests(unittest.TestCase):
    def test_plain_and_pinned_ids(self):
        self.assertEqual(parse_reference("microsoft/trocr-base-handwritten"),
                         ("microsoft/trocr-base-handwritten", None))
        self.assertEqual(parse_reference(" dh-unibe/trocr-kurrent@abc123 "),
                         ("dh-unibe/trocr-kurrent", "abc123"))

    def test_rejects_anything_that_is_not_an_id(self):
        for bad in ("", "trocr-base", "a/b/c", "../etc/passwd", "/models/x",
                    "https://huggingface.co/microsoft/trocr-base-printed", "owner/name@",
                    "owner/na me"):
            with self.subTest(reference=bad):
                with self.assertRaises(ModelValidationError):
                    parse_reference(bad)

    def test_malformed_reference_fails_before_any_network_call(self):
        with self.assertRaises(ModelValidationError):
            TrOCREngine().validate_reference("not a model id")


class SpecTests(unittest.TestCase):
    def test_models_are_referenced_not_uploaded(self):
        spec = TrOCREngine.spec
        self.assertEqual(spec.model_sources, (SOURCE_HF_REPO,))
        self.assertEqual(spec.file_extensions, ())
        with self.assertRaises(ModelValidationError):
            TrOCREngine().validate_model("/models/ab/german_print.mlmodel")

    def test_declares_only_what_it_does(self):
        spec = TrOCREngine.spec
        self.assertTrue(spec.can_recognize)
        self.assertFalse(spec.can_segment)
        self.assertFalse(spec.can_train)
        self.assertEqual(spec.preferred_queue, "intensive-inference")

    def test_recognizing_without_a_reference_is_an_error_not_silence(self):
        page = Image.new("RGB", (200, 60), "white")
        line = LineInput(id="1", baseline=[[5, 40], [190, 40]])
        with self.assertRaises(ValueError):
            list(TrOCREngine().recognize(page, [line], model=SimpleNamespace(reference=""),
                                         options=None))


class LineImageTests(unittest.TestCase):
    def setUp(self):
        # black everywhere, so anything the mask keeps shows up dark and anything it drops is white
        self.page = Image.new("RGB", (400, 200), "black")

    def test_outside_the_mask_is_blanked(self):
        triangle = LineInput(id="1", baseline=[[10, 90], [210, 90]],
                             boundary=[[10, 10], [210, 10], [10, 110]])
        crop = TrOCREngine.line_image(self.page, triangle)
        self.assertEqual(crop.size, (200, 100))
        self.assertEqual(crop.getpixel((5, 5)), (0, 0, 0), "inside the polygon is kept")
        self.assertEqual(crop.getpixel((195, 95)), (255, 255, 255), "outside the polygon is blank")

    def test_no_mask_falls_back_to_a_band_around_the_baseline(self):
        crop = TrOCREngine.line_image(self.page, LineInput(id="1", baseline=[[20, 100], [300, 100]]))
        self.assertIsNotNone(crop)
        self.assertEqual(crop.size[0], 280)
        self.assertGreater(crop.size[1], 8)

    def test_degenerate_geometry_yields_nothing(self):
        sliver = LineInput(id="1", baseline=[[20, 100], [21, 100]],
                           boundary=[[20, 99], [21, 99], [21, 100]])
        self.assertIsNone(TrOCREngine.line_image(self.page, sliver))

    def test_crop_is_clamped_to_the_page(self):
        overhang = LineInput(id="1", baseline=[[-50, 190], [450, 190]],
                             boundary=[[-50, 150], [450, 150], [450, 260], [-50, 260]])
        crop = TrOCREngine.line_image(self.page, overhang)
        self.assertEqual(crop.size, (400, 50))


@unittest.skipIf(_hub_model() is None, "Hugging Face Hub not reachable")
class HubValidationTests(unittest.TestCase):
    def test_a_trocr_checkpoint_is_claimed_and_pinned(self):
        info = TrOCREngine().validate_reference(TEST_REFERENCE)
        self.assertEqual(info.job, "recognition")
        repo, revision = parse_reference(info.reference)
        self.assertEqual(repo, TEST_REFERENCE)
        self.assertRegex(revision, r"^[0-9a-f]{40}$", "registration pins the commit")
        self.assertTrue(info.architecture.startswith("TrOCR/"))

    def test_an_image_to_text_model_that_is_not_trocr_is_refused(self):
        """The dangerous case: a VisionEncoderDecoder with every file a processor needs, which would
        load and run -- and caption the line image instead of transcribing it."""
        with self.assertRaisesRegex(ModelValidationError, "not a TrOCR model"):
            TrOCREngine().validate_reference("nlpconnect/vit-gpt2-image-captioning")

    def test_a_text_only_model_is_refused(self):
        with self.assertRaises(ModelValidationError):
            TrOCREngine().validate_reference("google-bert/bert-base-uncased")

    def test_weights_come_from_the_pinned_commit_not_a_conversion_pull_request(self):
        """microsoft/trocr-small-handwritten ships only pytorch_model.bin and has an open safetensors
        conversion pull request. Given the repository id, transformers loaded refs/pr/7 instead of the
        pinned commit; the snapshot must be the pinned revision and use the .bin that is really there."""
        import os

        info = TrOCREngine().validate_reference("microsoft/trocr-small-handwritten")
        repo, revision = parse_reference(info.reference)
        directory, weights = TrOCREngine._snapshot(repo, revision)
        self.assertEqual(os.path.basename(directory), revision)
        self.assertEqual(weights, "pytorch_model.bin")
        self.assertFalse(os.path.exists(os.path.join(directory, "model.safetensors")))

    def test_a_repository_that_does_not_exist_is_refused(self):
        with self.assertRaises(ModelValidationError):
            TrOCREngine().validate_reference("microsoft/definitely-not-a-trocr-model-xyz")


@unittest.skipIf(_hub_model() is None, "Hugging Face Hub not reachable")
class RecognitionTests(unittest.TestCase):
    def test_reads_printed_text(self):
        """Not an accuracy claim: proof that weights, processor and tokenizer all load and agree."""
        from PIL import ImageFont

        page = Image.new("RGB", (900, 120), "white")
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
        ImageDraw.Draw(page).text((30, 30), "Hello world 1870", fill="black", font=font)
        line = LineInput(id="7", baseline=[[20, 90], [880, 90]],
                         boundary=[[20, 20], [880, 20], [880, 100], [20, 100]])
        from engines.base import RecognizeOptions

        (result,) = list(TrOCREngine().recognize(page, [line], model=_hub_model(),
                                                 options=RecognizeOptions()))
        self.assertEqual(result.line_id, "7")
        self.assertIsNone(result.graphemes)
        self.assertIn("world", result.text.lower())
