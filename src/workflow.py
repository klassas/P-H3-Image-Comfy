"""ComfyUI node graph for P-H3-Image (Pruna AI), built in code per job.

A faithful transcription of Pruna AI's official workflows
(PrunaAI/P-H3-Image-Comfy: workflows/p-image-oss_text_to_image.json and
p-image-oss_edit.json), minus the prompt upsampler branch (a second
text model behind a "Generate Text" node). The upsampler is omitted on
purpose: the app sends full prompts, and the 5ch MiniMax H3 thread's
field advice for anatomy work is to keep prompt_upsampling OFF. The
prompt reaches the model exactly as typed.

    UNETLoader ──> [LoraLoaderModelOnly x N] ──> MiniMaxH3SigmaShift (1.0/1.0)
    MiniMaxH3SigmaShift ──> BasicGuider.model
    CLIPLoader (type "minimax") ──> PH3ImageConditioning.clip
    VAELoader ──> PH3ImageConditioning.vae        (edit mode only)
    LoadImage(s) ──> PH3ImageConditioning.images.image_N   (edit mode)
    PH3ImageConditioning ──> BasicGuider.conditioning / SamplerCustomAdvanced.latent_image
    BasicGuider + KSamplerSelect(euler) + ManualSigmas + RandomNoise
        ──> SamplerCustomAdvanced ──> VAEDecode ──> SaveImage

Sampling is the model's own 8-step schedule: the ManualSigmas string
below is copied verbatim from Pruna's workflows (9 sigma values = 8
steps), euler, no CFG (there is no negative pass — anything to avoid
must be written into the positive prompt; the app's negative_prompt
field is accepted and ignored, see src/validator.py).

Text-to-image: no reference images; width/height default to the
conditioning node's own default of 2912x2912 (~8 MP, the size this
model is built for). Edit mode: up to 8 references (<Picture 1> ..
<Picture 8>); Picture 1 is the image being edited and is first scaled
to 4 MP (ImageScaleToTotalPixels, as in Pruna's edit workflow), the
rest are references. With references and no explicit size, width and
height are 0 and the node derives the canvas from Picture 1. The VAE
is connected in edit mode so references are anchored as latents as
well as read by the text encoder.

VERIFIED (2026-11-26) against ComfyUI v0.46.2
comfy_extras/nodes_minimax_h3.py (PH3ImageConditioning.define_schema,
MiniMaxH3SigmaShift.define_schema) and Pruna's three workflow JSONs:
  1. Image input keys are "images.image_1" .. "images.image_9"
     (autogrow group "images", names image_1-based, 1-based — unlike
     MiniMaxH3ImageConditioning's 0-based image_0). <Picture 1> is
     images.image_1. FIRST-TEST WATCH ITEM: this key form follows the
     same autogrow flattening the H3 Image worker verified live for
     "images.image_0"; if the first edit job is rejected, this is the
     first suspect.
  2. Output order is output 0 = positive conditioning, output 1 =
     latent — as written below.
  3. MiniMaxH3SigmaShift takes shift_video / shift_audio; Pruna's
     workflows set both to 1 (the node docstring: "The model needs
     ModelSamplingMiniMaxH3 at 1.0/1.0").
  4. The dedicated image weights load with UNETLoader; user LoRAs
     chain between UNETLoader and MiniMaxH3SigmaShift. Pruna's own
     LoRA workflow uses LoraLoaderBypassModelOnly for its 2 GB
     conversion LoRA on the video FL2VA weights; this worker uses the
     plain LoraLoaderModelOnly every other worker here uses — if a
     LoRA fails to apply on the first test, that node variant is the
     second suspect.
"""

# Model files, as laid out in the slim repo (the Model Cache snapshot is
# symlinked into these exact category dirs by model_link.py). int8 set:
# 21.2 GB DiT + 5.1 GB text encoder + 3.1 GB VAE (~29.5 GB total).
UNET_NAME = "p-h3-image_int8_convrot.safetensors"
CLIP_NAME = "p-h3-image_text_encoder_int8_convrot.safetensors"
VAE_NAME = "p-h3-image_vae_int8_convrot.safetensors"

# Pruna's 8-step sigma schedule, verbatim from the official workflows.
SIGMAS = "1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0"

OUTPUT_NODE_ID = "save_image"

MAX_REFERENCE_IMAGES = 8


def build_workflow(
    *,
    prompt: str,
    width: int = 2912,
    height: int = 2912,
    seed: int = 0,
    image_names: list | None = None,
    filename_prefix: str = "ph3_image/job",
    extra_loras: list | None = None,
) -> dict:
    """Returns the API-format graph: {node_id: {"class_type", "inputs"}}.

    image_names: files already uploaded to ComfyUI's input/ directory,
    in <Picture N> order. Empty/None means text-to-image. width/height
    of 0 mean "inherit the first reference's size" (edit mode only).

    extra_loras: up to 3 slots of (lora_name, strength) or None, chained
    as LoraLoaderModelOnly nodes between the UNETLoader and the
    MiniMaxH3SigmaShift node. These are MiniMax H3 *video* LoRAs — the
    whole point of P-H3-Image is that they apply to stills here.
    """
    image_names = list(image_names or [])

    nodes: dict[str, dict] = {
        "unet_loader": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"},
        },
        "clip_loader": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP_NAME, "type": "minimax", "device": "default"},
        },
        "vae_loader": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_NAME},
        },
    }

    # Optional LoRA chain on the model branch: UNETLoader -> [LoRA x N]
    # -> MiniMaxH3SigmaShift -> BasicGuider.model. Slots that are None
    # (not configured, or set to strength 0 for this job) are skipped.
    model_ref = ["unet_loader", 0]
    for i, item in enumerate(extra_loras or []):
        if not item:
            continue
        lora_name, lora_strength = item
        if not lora_name:
            continue
        node_id = "lora_extra" if i == 0 else f"lora_extra_{i + 1}"
        nodes[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": lora_name,
                "strength_model": float(lora_strength),
            },
        }
        model_ref = [node_id, 0]

    nodes["sigma_shift"] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {"model": model_ref, "shift_video": 1.0, "shift_audio": 1.0},
    }

    conditioning_inputs: dict = {
        "clip": ["clip_loader", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
    }
    if image_names:
        # Anchor references as latents as well as text-encoder input.
        conditioning_inputs["vae"] = ["vae_loader", 0]
        for i, name in enumerate(image_names):
            node_id = f"load_image_{i + 1}"
            nodes[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": name},
            }
            source_ref = [node_id, 0]
            if i == 0:
                # Picture 1 sets the canvas: Pruna's edit workflow scales
                # it to 4 MP (32-pixel steps) before conditioning.
                nodes["scale_first"] = {
                    "class_type": "ImageScaleToTotalPixels",
                    "inputs": {
                        "image": source_ref,
                        "upscale_method": "lanczos",
                        "megapixels": 4.0,
                        "resolution_steps": 32,
                    },
                }
                source_ref = ["scale_first", 0]
            # 1-based API key: <Picture 1> = images.image_1.
            conditioning_inputs[f"images.image_{i + 1}"] = source_ref
    nodes["conditioning"] = {
        "class_type": "PH3ImageConditioning",
        "inputs": conditioning_inputs,
    }

    nodes["guider"] = {
        "class_type": "BasicGuider",
        "inputs": {
            "model": ["sigma_shift", 0],
            "conditioning": ["conditioning", 0],
        },
    }
    nodes["noise"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": seed},
    }
    nodes["sampler_select"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler"},
    }
    nodes["sigmas"] = {
        "class_type": "ManualSigmas",
        "inputs": {"sigmas": SIGMAS},
    }
    nodes["sampler"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["noise", 0],
            "guider": ["guider", 0],
            "sampler": ["sampler_select", 0],
            "sigmas": ["sigmas", 0],
            # Verified: output 0 = positive, output 1 = latent.
            "latent_image": ["conditioning", 1],
        },
    }
    nodes["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler", 0], "vae": ["vae_loader", 0]},
    }
    nodes[OUTPUT_NODE_ID] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["decode", 0], "filename_prefix": filename_prefix},
    }
    return nodes
