# How a deployment adds this engine: extend an eScriptorium image and pip install the package.
# Layered on the tesseract image so that one image carries all three engines; any eScriptorium
# image works as the base. Nothing in eScriptorium changes.
ARG BASE_IMAGE=escriptorium-tesseract:latest
FROM ${BASE_IMAGE}

COPY . /opt/escriptorium-engine-trocr

# An engine must not move the dependencies eScriptorium and kraken were installed with. Newer
# transformers releases require safetensors>=0.8 while kraken 7.0 pins ~=0.7.0, and pip would
# otherwise upgrade it and leave kraken broken. Freezing the shared packages as constraints makes pip
# pick the newest transformers that fits around them instead; pip check then refuses the build if
# anything still conflicts.
RUN pip freeze | grep -i -E '^(safetensors|huggingface[-_]hub|numpy|torch|torchvision|pillow|protobuf)==' \
        > /tmp/base-constraints.txt \
    && pip install --no-cache-dir -c /tmp/base-constraints.txt -e /opt/escriptorium-engine-trocr \
    && pip check \
    && rm /tmp/base-constraints.txt
