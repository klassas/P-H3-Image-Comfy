# P-H3-Image worker

A RunPod Serverless worker for **P-H3-Image** (Pruna AI's image model
built on the MiniMax H3 *video* transformer — not the same model as
MiniMax's own H3 Image). One job in, one PNG out (uploaded to S3/B2,
returned as a pre-signed `image_url`).

Why this model exists alongside the H3 Image worker: because its base
is the H3 video transformer, **MiniMax H3 video LoRAs work on stills
here** (MysticXXX, PENISLORA, Male_Anatomy, 10eros-style checkpoints'
LoRAs, character-swap LoRAs...). Sampling is Pruna's fixed 8-step
sigma schedule, euler, no CFG. Text-to-image renders at ~8 MP by
default (2912x2912); edits take up to 8 images (<Picture 1> is the
image being edited, the rest are references).

The prompt upsampler from Pruna's official workflows is deliberately
NOT wired: the app sends full prompts, and field reports (5ch MiniMax
H3 thread) say anatomy results are better with prompt_upsampling off.

## Model files (slim repo)

`klassas/p-h3-image-slim` (private) holds only the int8 set from
PrunaAI/P-H3-Image-Comfy (~29.5 GB total):

- `diffusion_models/p-h3-image_int8_convrot.safetensors` (21.2 GB)
- `text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors` (5.1 GB)
- `vae/p-h3-image_vae_int8_convrot.safetensors` (3.1 GB)

The endpoint's Model field points at this repo; RunPod Model Cache
pre-downloads it and the worker symlinks it into ComfyUI at boot
(src/model_link.py). Because the repo is private, the endpoint needs
an HF_TOKEN with read access (the auto-added HF_TOKEN env line from
the endpoint's Hugging Face token setting).

## Endpoint settings (RunPod Serverless)

- GPU tier: **48 GB** (check every GPU type in the tier), Max workers 1
- Model (Model Cache): `klassas/p-h3-image-slim`
- Container disk: 20 GB
- Execution timeout: 1800 s
- FlashBoot: **OFF** (the free FlashBoot allowance is already used by
  the H3 / R2V / H3 Image endpoints)
- Data centers: **AP-JP-1, AP-IN-1, CA-MTL-1, CA-MTL-3, CA-MTL-4 only**
  (the MiniMax H3 Community License excludes hosting in the US / EU /
  UK / South Korea, same as the H3 and H3 Image endpoints)
- Allowed CUDA versions: 13.0+ (the image ships a cu130 torch build)
- Environment variables:
  - `HF_REPO=klassas/p-h3-image-slim`
  - the 5 S3 lines (same values as the other endpoints):
    `S3_ENDPOINT`, `S3_REGION`, `S3_ACCESS_KEY_ID`,
    `S3_SECRET_ACCESS_KEY`, `S3_BUCKET_DEFAULT`
  - the HF_TOKEN line RunPod adds automatically when the account's
    Hugging Face token is set
  - optional LoRA slots (see below)

Do not test before the creation build shows **Completed** and the
endpoint shows **Ready**.

## LoRA slots (optional, up to 3)

Same mechanism as the other workers: a slot is configured by env vars,
its file is fetched from the B2 bucket at boot, and the app steers it
per request with `lora_strength` / `lora_strength_2` / `lora_strength_3`
(0 = slot skipped for that job; endpoint default strength from
`EXTRA_LORA_STRENGTH{,_2,_3}`, 0.5 unless set).

- `EXTRA_LORA_KEY` / `EXTRA_LORA_KEY_2` / `EXTRA_LORA_KEY_3`: object
  keys in the S3 bucket (e.g. the filename of an H3 video LoRA uploaded
  to the bucket), or
- `EXTRA_LORA_URL{,_2,_3}`: direct download URLs instead of keys.

A configured-but-unfetchable LoRA fails the boot on purpose (never
delete a B2 file that an endpoint env still points at).

First candidates, per the 5ch thread's advice for anatomy:
MysticXXX_MMH3-V5 in slot 1. PENISLORA / Male_Anatomy (both directly
downloadable via civarchive) are the free male-anatomy options that
the official H3 Image model lacks a free LoRA for.

## First-test watch items

1. **Edit-mode image keys** — the graph sends references as
   `images.image_1` .. `images.image_8` on PH3ImageConditioning
   (autogrow flattening, verified against the ComfyUI v0.46.2 node
   schema; the same mechanism the H3 Image worker uses live with
   `images.image_0`). If the first edit job is rejected by ComfyUI,
   this is the first suspect.
2. **LoRA node variant** — user LoRAs chain with LoraLoaderModelOnly.
   Pruna's own conversion-LoRA workflow uses
   LoraLoaderBypassModelOnly; if a job's LoRA visibly does nothing,
   that variant is the second suspect.
3. **VRAM at 8 MP** — Pruna rates the int8 set at ~31 GB peak at 8 MP;
   the 48 GB tier should hold it with room to spare. Watch the first
   T2I job's log for CUDA OOM just in case.

## Job contract

Input (JSON): `prompt` (required), optional `width`/`height` (both or
neither; defaults 2912x2912 for t2i, 0 = derive from Picture 1 for
edits), `seed`, `image_base64` or `image_base64s` (<= 8) /
`image_url(s)`, `lora_strength{,_2,_3}`, `output {public, ttl_seconds}`.
`negative_prompt` and `workflow` are accepted and ignored.

Output: `{status: "completed", image_url, meta, timing}`.
