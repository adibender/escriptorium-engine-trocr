# escriptorium-engine-trocr

[TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr) as an eScriptorium engine.
Installing this package is the entire integration — the `escriptorium.engines` entry point in
`pyproject.toml` makes the engine appear. Nothing in eScriptorium is edited.

## Install

This is a plugin: it installs *into* an eScriptorium environment. Build an image that layers it onto
eScriptorium's own rather than pip-installing by hand — an engine must not move the dependency
versions eScriptorium and kraken were installed with, and the Dockerfile guards that by freezing the
shared packages as constraints and running `pip check`.

```sh
# The Dockerfile's BASE_IMAGE defaults to escriptorium-tesseract:latest, so that one image can carry
# several engines. Any eScriptorium image works:
docker build --build-arg BASE_IMAGE=registry.gitlab.com/scripta/escriptorium:latest \
             -t escriptorium-engines:latest .
```

Point the app containers at that image, and give them a **shared** Hugging Face cache: weights are
downloaded the first time a model is used, and without a shared cache every worker downloads its own
copy.

```yaml
# docker-compose.override.yml
services:
  web:
    image: escriptorium-engines:latest
    environment:
      - HF_HOME=/hf-cache
    volumes:
      - hf-cache:/hf-cache
  # the same three lines on every app container, including celery-main,
  # celery-gpu and celery-intensive-inference

volumes:
  hf-cache:
```

**Restart the app containers.** Entry points are read when a process starts, so a container that is
already running never notices a newly installed engine.

Verify: `GET /api/engines/` lists `trocr`, and *TrOCR* appears in the model picker of the Transcribe
dialog.

## Use

1. **Add a model.** A TrOCR model is not uploaded. On the *Models* page choose **Add a Hugging Face
   model** and give its id, e.g. `microsoft/trocr-base-handwritten`.
2. **Segment the pages first**, with kraken or tesseract — TrOCR recognises lines and has no
   segmenter of its own.
3. **Transcribe**, choosing the TrOCR model. The model carries its engine, so nothing else has to be
   selected.

The engine checks that the repository is a public VisionEncoderDecoder checkpoint with a TrOCR
decoder, then **pins it to the current commit** — the stored reference is `owner/name@<sha>`, so the
model cannot change underneath transcriptions already made with it. You may pin a revision yourself
with `owner/name@revision`.

Works with any such checkpoint regardless of encoder or tokenizer: Microsoft's `trocr-*-printed` and
`trocr-*-handwritten` (RoBERTa and XLM-R tokenizers), and community fine-tunes like
`dh-unibe/trocr-kurrent`. Each checkpoint's own generation settings (e.g. beam search) are used.

## What it does and does not do

- **recognises lines only.** Segment with kraken or tesseract first.
- crops each line tightly and blanks everything outside its mask, so neighbouring lines of
  handwriting do not bleed in
- reports no character positions or confidences: `graphemes` is `None`
- asks for the `intensive-inference` queue, i.e. a GPU; falls back to CPU where CUDA is absent
- cannot train (`can_train=False`)
- never runs repository code (`trust_remote_code` is off), refuses gated or private repositories,
  and refuses checkpoints above 4 GB

## A log message that looks worse than it is

Loading a checkpoint makes transformers print a *LOAD REPORT* saying `encoder.pooler.dense.weight` and
`.bias` are MISSING and were newly initialised. That is expected and does not affect recognition: the
ViT/DeiT encoder builds a pooling layer by default, but `VisionEncoderDecoderModel` passes only the
encoder's last hidden state to the decoder and never uses the pooled output.

## Tests

```sh
python manage.py test escriptorium_engine_trocr    # inside an eScriptorium container
```

Hub-dependent tests skip when huggingface.co is unreachable.

## Relationship to eScriptorium

This is a plugin, not a standalone library: it imports the engine contract (`engines.base`) from
eScriptorium and installs into an eScriptorium environment, which is also the only place it can run.
That is why `pyproject.toml` declares no dependency on eScriptorium — eScriptorium is deployed as an
application, not published as a distribution.

The practical consequence is the one noted under *Tests*: the suite needs an eScriptorium checkout on
the path. If third-party engines should one day be developed and tested on their own, the way there
is to extract the interface into a small distribution that both sides depend on — a change we would
be glad to contribute rather than expect.
