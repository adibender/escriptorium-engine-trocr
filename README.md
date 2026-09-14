# escriptorium-engine-trocr

[TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr) as an eScriptorium engine.
Installing this package is the entire integration — the `escriptorium.engines` entry point in
`pyproject.toml` makes the engine appear.

## Install

```sh
pip install -e .            # into an environment that already has eScriptorium and torch
docker build -t escriptorium-engines:latest .   # or: layer it onto an eScriptorium image
```

Set `HF_HOME` to a persistent, shared directory on every app container. Weights are downloaded the
first time a model is used and cached there; without a shared cache every worker downloads again.

## Use

A TrOCR model is not uploaded. On the *Models* page choose **Add a Hugging Face model** and give its
id, e.g. `microsoft/trocr-base-handwritten`. The engine checks that the repository is a public
VisionEncoderDecoder checkpoint with a TrOCR decoder, then **pins it to the current commit** — the
stored reference is `owner/name@<sha>`, so the model cannot change underneath transcriptions already
made with it. You may pin a revision yourself with `owner/name@revision`.

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
