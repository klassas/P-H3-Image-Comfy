FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Model Cache mounts the endpoint's Model repo under
# /runpod-volume/huggingface-cache/hub/ but does NOT set HF_HOME for us —
# confirmed on a live worker (both HF_HOME and HF_HUB_CACHE unset). The
# Flux/Krea/Wan2.2 workers set it here themselves; same here.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/runpod-volume/huggingface-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ComfyUI itself, via comfy-cli (its torch is replaced with a cu130 build in
# the next step).
#
# Version floor: PH3ImageConditioning (P-H3-Image support) was merged in
# Comfy-Org/ComfyUI #17506 and ships in v0.44.0+; the pin below (0.46.2)
# includes it (verified against that release's
# comfy_extras/nodes_minimax_h3.py). Pinned to a specific release rather
# than "latest" so a future ComfyUI release cannot change node behavior
# under this worker without a deliberate version bump here. Bump only
# after re-verifying src/workflow.py's node graph against the new
# release's node schemas (comfy_extras/nodes_minimax_h3.py).
ARG COMFYUI_VERSION=0.46.2
ARG COMFY_DIR=/opt/ComfyUI
ENV COMFY_DIR=${COMFY_DIR}
RUN pip install --no-cache-dir comfy-cli && \
    comfy --workspace ${COMFY_DIR} --skip-prompt install \
        --version ${COMFYUI_VERSION} --cuda-version 12.8 --nvidia && \
    test -f ${COMFY_DIR}/main.py

# Swap torch for a cu130 build. On cu128 torch, ComfyUI's comfy_kitchen
# disables its CUDA and Triton kernel backends ("You need pytorch with cu130
# or higher to use optimized CUDA operations") and runs int8_convrot
# weights — the DiT here is int8_convrot — through the eager fallback.
# torchaudio's newest cu130 build is 2.11.0, so the trio is pinned to the
# matching 2.11 set. cu130 wheels bundle their own CUDA runtime but need a
# host driver that supports CUDA 13.0 (>= 580) — restrict the endpoint's
# allowed CUDA versions accordingly.
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
# Explicit +cu130 local versions (PyPI also publishes torch==2.11.0, as a
# different CUDA build); no --force-reinstall, so ComfyUI's other pinned deps
# are only touched if torch actually requires it.
RUN python -m pip install --no-cache-dir --timeout 300 --retries 5 \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple \
        torch==${TORCH_VERSION}+cu130 torchaudio==${TORCHAUDIO_VERSION}+cu130 \
        torchvision==${TORCHVISION_VERSION}+cu130 && \
    python -c "import torch, torchaudio, torchvision; \
print(f'torch={torch.__version__} cuda={torch.version.cuda} torchaudio={torchaudio.__version__} torchvision={torchvision.__version__}'); \
assert torch.version.cuda and torch.version.cuda.startswith('13'), torch.version.cuda"

# This worker's own deps (runpod, requests, websocket-client for
# src/comfy_client.py, boto3 for src/storage.py) — separate from whatever
# comfy-cli installed into the same environment for ComfyUI itself.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --timeout 180 --retries 5 \
    -r requirements.txt

# Application code
COPY src/ src/

# Verify imports at build time. handler.py is excluded — importing it starts
# the ComfyUI subprocess and blocks on model linking. Everything else is
# pure Python with no GPU/process dependency, so any dependency mismatch or
# typo fails the build here instead of at the next worker cold start.
RUN python -c "\
from src.config import MODEL_TAG, HF_REPO, COMFY_DIR, model_tag; \
from src.validator import validate_input; \
from src.workflow import build_workflow, OUTPUT_NODE_ID, UNET_NAME, CLIP_NAME, VAE_NAME; \
from src.comfy_client import ComfyClient, ComfyError; \
from src.model_link import link_model_cache, ModelLinkError; \
from src.image import fetch_images; \
from src.storage import upload_image; \
from src.errors import format_error; \
from src.security import validate_url; \
import runpod, boto3, websocket; \
wf = build_workflow(prompt='a cat', width=2912, height=2912, seed=1); \
assert OUTPUT_NODE_ID in wf, wf.keys(); \
assert wf['conditioning']['class_type'] == 'PH3ImageConditioning', wf['conditioning']; \
assert wf['sampler']['class_type'] == 'SamplerCustomAdvanced', wf['sampler']; \
assert wf['sampler']['inputs']['latent_image'] == ['conditioning', 1], wf['sampler']; \
assert wf['guider']['inputs']['conditioning'] == ['conditioning', 0], wf['guider']; \
assert wf['guider']['inputs']['model'] == ['sigma_shift', 0], wf['guider']; \
assert wf['sigma_shift']['inputs']['model'] == ['unet_loader', 0], wf['sigma_shift']; \
assert wf['sigma_shift']['inputs']['shift_video'] == 1.0 and wf['sigma_shift']['inputs']['shift_audio'] == 1.0, wf['sigma_shift']; \
assert wf['sigmas']['inputs']['sigmas'].startswith('1.0, 0.984753'), wf['sigmas']; \
assert 'vae' not in wf['conditioning']['inputs'], wf['conditioning']; \
assert wf['conditioning']['inputs']['clip'] == ['clip_loader', 0], wf['conditioning']; \
assert wf['conditioning']['inputs']['prompt'] == 'a cat', wf['conditioning']; \
wf = build_workflow(prompt='edit this', width=0, height=0, seed=1, image_names=['in1.png', 'in2.png']); \
assert wf['conditioning']['inputs']['images.image_1'] == ['scale_first', 0], wf['conditioning']; \
assert wf['conditioning']['inputs']['images.image_2'] == ['load_image_2', 0], wf['conditioning']; \
assert wf['scale_first']['inputs']['image'] == ['load_image_1', 0], wf['scale_first']; \
assert 'vae' in wf['conditioning']['inputs'], wf['conditioning']; \
wf = build_workflow(prompt='lora', extra_loras=[('a.safetensors', 0.8), None, ('b.safetensors', 1.0)]); \
assert wf['lora_extra']['inputs']['model'] == ['unet_loader', 0], wf['lora_extra']; \
assert wf['lora_extra_3']['inputs']['model'] == ['lora_extra', 0], wf['lora_extra_3']; \
assert wf['sigma_shift']['inputs']['model'] == ['lora_extra_3', 0], wf['sigma_shift']; \
assert 'lora_extra_2' not in wf, wf.keys(); \
p = validate_input({'prompt': 'x'}); \
assert p['mode'] == 't2i' and p['width'] == 2912 and p['height'] == 2912, p; \
assert p['lora_strength'] == 0.5 and p['lora_strength_2'] == 0.5, p; \
p = validate_input({'prompt': 'x', 'workflow': {'1': {}}, 'image_base64': 'AAAA'}); \
assert p['mode'] == 'edit' and p['width'] == 0 and p['height'] == 0, p; \
assert p['ignored_params'] == ['workflow'], p; \
print(f'All imports OK (tag={MODEL_TAG} model={model_tag()} repo={HF_REPO} comfy_dir={COMFY_DIR})')"

# Only relevant when SERVE_MODE=api (the on-demand Pod deploy path, see
# src/watchdog.py). Serverless (the default) never
# binds this — EXPOSE is documentation for `docker run -p`/Pod port
# mapping, not a switch.
EXPOSE 8000

CMD ["python", "-u", "src/handler.py"]
