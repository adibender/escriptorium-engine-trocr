"""TrOCR engine for eScriptorium.

TrOCR is the second engine outside kraken, and it was chosen because it disagrees with both kraken
and Tesseract about what a model even is:

- a model is not a file anybody uploads, but a **Hugging Face repository**, pinned to a commit when
  it is registered so that it cannot change underneath transcriptions already made with it;
- it has **no layout analysis**: it reads lines that some other engine, or a person, segmented;
- it generates text token by token, so there are **no character positions and no per-character
  confidences** -- `graphemes` is None, which is a different statement from "this line is empty";
- it is a transformer and wants a **GPU**, so it asks for the intensive-inference queue.

Tesseract proved that a second engine can plug in without changing eScriptorium. This package is
the test of whether that holds for an engine whose assumptions are different again.

Any VisionEncoderDecoder checkpoint with a TrOCR decoder is accepted -- Microsoft's printed and
handwritten models, community fine-tunes such as dh-unibe/trocr-kurrent -- whatever its encoder or
tokenizer. Repository code is never executed: `trust_remote_code` stays off.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Iterable, Iterator

from engines.base import (
    SOURCE_HF_REPO,
    BaseEngine,
    EngineError,
    EngineSpec,
    LineInput,
    ModelInfo,
    ModelValidationError,
    RecognitionResult,
    RecognizeOptions,
)
from engines.devices import usable_device
from engines.imaging import crop_line

logger = logging.getLogger(__name__)

#: `owner/name`, optionally pinned as `owner/name@revision`. Deliberately strict: the value comes
#: from a form, and nothing that looks like a path or a URL should reach the Hub client.
REFERENCE_PATTERN = re.compile(
    r"^(?P<repo>[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)"
    r"(?:@(?P<revision>[A-Za-z0-9][A-Za-z0-9_.-]*))?$"
)

#: refuse checkpoints bigger than this at registration. trocr-large is about 2.2 GB; anything far
#: beyond that is not a TrOCR line recogniser, and the Hub cache is shared disk no quota counts.
MAX_WEIGHTS_BYTES = 4 * 1024 ** 3

#: files a checkpoint needs before transformers can even build a processor for it
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "tokenizer_config.json")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
#: everything but weights that a checkpoint may need: configs, BPE vocabularies, sentencepiece models
SUPPORT_FILE_PATTERNS = ("*.json", "*.txt", "*.model")

BATCH_SIZE = {"cuda": 16, "cpu": 4}
MAX_NEW_TOKENS = 128

#: loaded checkpoints per worker process. Engine instances live for one page, and a
#: base checkpoint takes seconds to load, so without this a 50-page document loads it 50 times.
#: Two, because a user comparing models alternates between them.
MODEL_CACHE_SIZE = 2

_loaded: OrderedDict = OrderedDict()
_loaded_lock = threading.Lock()


def parse_reference(reference: str) -> tuple[str, str | None]:
    """Split `owner/name[@revision]`, or raise ModelValidationError."""
    match = REFERENCE_PATTERN.match((reference or "").strip())
    if match is None:
        raise ModelValidationError(
            "Expected a Hugging Face model id such as 'microsoft/trocr-base-handwritten', "
            "optionally pinned as 'owner/name@revision'."
        )
    return match["repo"], match["revision"]


class TrOCREngine(BaseEngine):
    spec = EngineSpec(
        name="trocr",
        label="TrOCR",
        can_segment=False,        # no layout analysis; declaring it would lie
        can_recognize=True,
        can_train=False,
        model_sources=(SOURCE_HF_REPO,),
        file_extensions=(),       # nothing to upload: models are referenced, not stored
        preferred_queue="intensive-inference",
    )

    # --- model handling

    def validate_model(self, path: str) -> ModelInfo:
        raise ModelValidationError(
            "TrOCR models are added by their Hugging Face id, not uploaded as files."
        )

    def validate_reference(self, reference: str) -> ModelInfo:
        """Check that `reference` names a usable TrOCR checkpoint, and pin it to a commit.

        Reads repository metadata and config.json only -- the weights are fetched by the worker the
        first time the model is used, not by a web request.
        """
        repo, revision = parse_reference(reference)

        try:
            from huggingface_hub import HfApi, hf_hub_download
            from huggingface_hub.errors import (
                GatedRepoError,
                RepositoryNotFoundError,
                RevisionNotFoundError,
            )
        except ImportError as exc:  # pragma: no cover - a broken install, not a user error
            raise ModelValidationError(f"huggingface_hub is not installed: {exc}")

        try:
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        except GatedRepoError:
            raise ModelValidationError(
                f"{repo} is gated; this instance holds no Hugging Face access token."
            )
        except RepositoryNotFoundError:
            raise ModelValidationError(f"No public Hugging Face model named {repo}.")
        except RevisionNotFoundError:
            raise ModelValidationError(f"{repo} has no revision {revision!r}.")
        except Exception as exc:
            raise ModelValidationError(f"Could not reach Hugging Face to check {repo}: {exc}")

        if getattr(info, "gated", False) or getattr(info, "private", False):
            raise ModelValidationError(
                f"{repo} is gated or private; this instance holds no Hugging Face access token."
            )

        sizes = {sibling.rfilename: (sibling.size or 0) for sibling in (info.siblings or ())}
        missing = [name for name in REQUIRED_FILES if name not in sizes]
        if missing:
            raise ModelValidationError(
                f"{repo} is missing {', '.join(missing)}, so no processor can be built for it."
            )
        weights = next((name for name in WEIGHT_FILES if name in sizes), None)
        if weights is None:
            raise ModelValidationError(f"{repo} has no model.safetensors or pytorch_model.bin.")
        if sizes[weights] > MAX_WEIGHTS_BYTES:
            raise ModelValidationError(
                f"{repo} weighs {sizes[weights] / 1024 ** 3:.1f} GB, above the "
                f"{MAX_WEIGHTS_BYTES / 1024 ** 3:.0f} GB accepted for a line recogniser."
            )

        try:
            with open(hf_hub_download(repo, "config.json", revision=info.sha)) as handle:
                config = json.load(handle)
        except Exception as exc:
            raise ModelValidationError(f"Could not read config.json of {repo}: {exc}")

        decoder = (config.get("decoder") or {}).get("model_type")
        encoder = (config.get("encoder") or {}).get("model_type")
        if config.get("model_type") != "vision-encoder-decoder" or decoder != "trocr":
            raise ModelValidationError(
                f"{repo} is not a TrOCR model (model_type={config.get('model_type')!r}, "
                f"decoder={decoder!r})."
            )

        return ModelInfo(
            job="recognition",
            architecture=f"TrOCR/{encoder}"[:64],
            reference=f"{repo}@{info.sha}",
            metadata={
                "repository": repo,
                "revision": info.sha,
                "encoder": encoder,
                "weights": weights,
                "weights_bytes": sizes[weights],
            },
        )

    # --- recognition

    def recognize(
        self, image, lines: Iterable[LineInput], *, model, options: RecognizeOptions
    ) -> Iterator[RecognitionResult]:
        page = None
        crops = []
        for line in lines:
            if page is None:
                page = image if image.mode == "RGB" else image.convert("RGB")
            crop = crop_line(page, line, mask=True)
            if crop is not None:
                crops.append((line.id, crop))

        # an unsegmented page must not cost a model load
        if not crops:
            return

        import torch

        image_processor, tokenizer, network, device = self._load(
            self._reference(model), options.device
        )
        batch_size = BATCH_SIZE["cuda" if device.startswith("cuda") else "cpu"]

        for start in range(0, len(crops), batch_size):
            chunk = crops[start:start + batch_size]
            pixel_values = image_processor(
                images=[crop for _, crop in chunk], return_tensors="pt"
            ).pixel_values.to(device)
            with torch.inference_mode():
                generated = network.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS)
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for (line_id, _), text in zip(chunk, texts):
                yield RecognitionResult(
                    line_id=line_id,
                    text=text.strip(),
                    # a generative decoder has no character positions to report
                    graphemes=None,
                )

    @staticmethod
    def _reference(model) -> str:
        reference = getattr(model, "reference", "") if model is not None else ""
        if not reference:
            raise EngineError("TrOCR needs a model registered by its Hugging Face id.")
        return reference

    @staticmethod
    def _load(reference: str, device: str):
        """(image processor, tokenizer, model, device) for `reference`, cached per process.

        The image processor and the tokenizer are loaded separately rather than through
        AutoProcessor. For Microsoft's checkpoints, transformers 5 resolves AutoProcessor to the
        bare tokenizer -- there is no processor_config.json to say otherwise -- so the page image
        was handed to a tokenizer, which fails with "You need to specify either `text` or
        `text_target`". Loading the two halves explicitly works for every checkpoint layout.
        """
        from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

        device = usable_device(device)

        repo, revision = parse_reference(reference)
        key = (repo, revision, device)
        with _loaded_lock:
            if key in _loaded:
                _loaded.move_to_end(key)
                return _loaded[key]

        directory, weights = TrOCREngine._snapshot(repo, revision)
        image_processor = AutoImageProcessor.from_pretrained(directory)
        tokenizer = AutoTokenizer.from_pretrained(directory)
        # transformers logs a LOAD REPORT listing encoder.pooler.dense.* as MISSING and freshly
        # initialised. Harmless: ViT and DeiT build a pooler by default, but VisionEncoderDecoderModel
        # hands the decoder only encoder_outputs[0], the last hidden state, and never the pooled output.
        network = VisionEncoderDecoderModel.from_pretrained(
            directory, use_safetensors=weights.endswith(".safetensors")
        )
        # older checkpoints carry the start token only in the tokenizer
        if network.generation_config.decoder_start_token_id is None:
            network.generation_config.decoder_start_token_id = tokenizer.cls_token_id
        if network.generation_config.pad_token_id is None:
            network.generation_config.pad_token_id = tokenizer.pad_token_id
        network.to(device).eval()

        with _loaded_lock:
            _loaded[key] = (image_processor, tokenizer, network, device)
            _loaded.move_to_end(key)
            while len(_loaded) > MODEL_CACHE_SIZE:
                _loaded.popitem(last=False)
        return image_processor, tokenizer, network, device

    @staticmethod
    def _snapshot(repo: str, revision: str | None) -> tuple[str, str]:
        """Fetch exactly the pinned revision into the Hub cache; return (directory, weights file).

        Loading goes through this rather than handing the repository id to `from_pretrained`,
        because given an id transformers may load weights from a *different commit*: for a
        repository that ships only pytorch_model.bin, it looks for an open "Adding safetensors
        variant" pull request by Hugging Face's conversion bot and loads that branch instead.
        microsoft/trocr-small-handwritten is such a repository -- pinned to its main commit, it
        loaded model.safetensors from refs/pr/7, never merged. From a local directory no such
        substitution is possible, so the weights are the ones the registration pinned.

        Safetensors is preferred when the pinned revision has it; otherwise the .bin, which torch
        loads with weights_only. Honours HF_HUB_OFFLINE, so a pre-provisioned cache needs no network.
        """
        import os

        from huggingface_hub import snapshot_download

        for weights in WEIGHT_FILES:
            directory = snapshot_download(
                repo, revision=revision, allow_patterns=[*SUPPORT_FILE_PATTERNS, weights]
            )
            if os.path.exists(os.path.join(directory, weights)):
                return directory, weights
        raise EngineError(f"{repo}@{revision} has neither {' nor '.join(WEIGHT_FILES)}.")
