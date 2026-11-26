"""Input validation for the P-H3-Image (Pruna AI) ComfyUI worker.

The field set mirrors what the RunPod Media Studio app sends for its
image models (see app RunPodClient.kt): flat fields prompt /
negative_prompt / image_url / image_base64 / width / height, plus a
`workflow` object this worker accepts and ignores — it builds the
graph itself in src/workflow.py.

There are no steps / cfg fields: sampling is Pruna's fixed 8-step
sigma schedule with no CFG pass (see src/workflow.py).

Reference images: one via image_url / image_base64, or up to 8 via
image_urls / image_base64s lists, in <Picture N> order. Picture 1 is
the image being edited; the rest are references. The two forms must
not be mixed.
"""

from src import config
from src.errors import ValidationError


def _validate_image_sources(inp: dict) -> list:
    """Collect the reference images as a list of ('url'|'b64', value)."""
    singles = [k for k in ("image_url", "image_base64") if inp.get(k)]
    lists = [k for k in ("image_urls", "image_base64s") if inp.get(k)]
    if singles and lists:
        raise ValidationError(
            "Provide either a single image (image_url/image_base64) or "
            "image lists (image_urls/image_base64s), not both.",
            hint="Remove one of the image source fields.",
        )
    if len(singles) > 1:
        raise ValidationError(
            "Provide either image_url or image_base64, not both.",
            hint="Remove one of the image source fields.",
        )
    sources: list = []
    if singles:
        key = singles[0]
        sources.append(("url" if key == "image_url" else "b64", inp[key]))
    for key in lists:
        value = inp[key]
        if not isinstance(value, list) or not value:
            raise ValidationError(
                f"{key} must be a non-empty list.",
                hint=f"You sent {key}={value!r}.",
            )
        kind = "url" if key == "image_urls" else "b64"
        for item in value:
            if not isinstance(item, str) or not item:
                raise ValidationError(
                    f"Every entry in {key} must be a non-empty string.",
                    hint=f"You sent {key}={value!r}.",
                )
            sources.append((kind, item))
    if len(sources) > config.MAX_REFERENCE_IMAGES:
        raise ValidationError(
            f"At most {config.MAX_REFERENCE_IMAGES} reference images.",
            hint=f"You sent {len(sources)}.",
        )
    return sources


def validate_input(inp: dict) -> dict:
    """Validate and normalize job input. Returns a cleaned dict with defaults applied."""
    if not isinstance(inp, dict):
        raise ValidationError(
            "Input must be a JSON object.",
            hint="Send a JSON object with at least a 'prompt' field.",
        )

    image_sources = _validate_image_sources(inp)
    mode = "edit" if image_sources else "t2i"

    prompt = inp.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValidationError("prompt must be a string.")
    prompt = prompt.strip()
    if not prompt:
        raise ValidationError(
            "prompt is required.",
            hint="Provide a text description of the desired image.",
        )
    if len(prompt) > config.MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"prompt exceeds max length ({config.MAX_PROMPT_LENGTH} chars).",
            hint="Shorten your prompt.",
        )

    negative_prompt = inp.get("negative_prompt", "")
    if not isinstance(negative_prompt, str):
        raise ValidationError("negative_prompt must be a string.")
    if len(negative_prompt) > config.MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"negative_prompt exceeds max length ({config.MAX_PROMPT_LENGTH} chars).",
        )

    seed = inp.get("seed", config.DEFAULT_SEED)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError("seed must be an integer.")

    # Per-request LoRA strengths (0-2). Defaults come from the endpoint's
    # EXTRA_LORA_STRENGTH{,_2,_3} env vars; 0 skips that slot for the job.
    def _strength(field: str, default: float) -> float:
        v = inp.get(field, default)
        if not isinstance(v, (int, float)) or isinstance(v, bool) \
                or v < 0 or v > 2:
            raise ValidationError(
                f"{field} must be a number between 0 and 2.",
                hint=f"You sent {field}={v!r}.",
            )
        return float(v)

    lora_strength = _strength("lora_strength", config.EXTRA_LORA_STRENGTH)
    lora_strength_2 = _strength("lora_strength_2", config.EXTRA_LORA_STRENGTH_2)
    lora_strength_3 = _strength("lora_strength_3", config.EXTRA_LORA_STRENGTH_3)

    # Size: both-or-neither. A plain t2i request sends neither and gets
    # the defaults (2912x2912, ~8 MP). An edit request may also send
    # neither: then width/height are 0 and the conditioning node reuses
    # the first reference's size. The app sends the pair when it wants
    # an explicit canvas (e.g. a character-sheet layout).
    has_w, has_h = "width" in inp, "height" in inp
    if has_w != has_h:
        raise ValidationError(
            "width and height must be sent together.",
            hint="Send both, or neither to use the defaults.",
        )
    if has_w:
        width = inp["width"]
        height = inp["height"]
    elif mode == "edit":
        width, height = 0, 0
    else:
        width, height = config.DEFAULT_WIDTH, config.DEFAULT_HEIGHT
    if not isinstance(width, int) or not isinstance(height, int) \
            or isinstance(width, bool) or isinstance(height, bool):
        raise ValidationError("width and height must be integers.")
    if width < 0 or height < 0:
        raise ValidationError("width and height must not be negative.")
    if (width == 0) != (height == 0):
        raise ValidationError(
            "width and height must both be 0 (auto) or both be set.",
            hint=f"You sent {width}x{height}.",
        )
    if width and height:
        if width < config.MIN_DIMENSION or height < config.MIN_DIMENSION:
            raise ValidationError(
                f"width and height must be at least {config.MIN_DIMENSION}.",
                hint=f"You sent {width}x{height}.",
            )
        if width % config.MOD_VALUE != 0 or height % config.MOD_VALUE != 0:
            snapped_w = width // config.MOD_VALUE * config.MOD_VALUE
            snapped_h = height // config.MOD_VALUE * config.MOD_VALUE
            raise ValidationError(
                f"width and height must be divisible by {config.MOD_VALUE}.",
                hint=f"You sent {width}x{height}. Try {snapped_w}x{snapped_h}.",
            )
        if width * height > config.MAX_RESOLUTION_AREA:
            raise ValidationError(
                f"Total pixels ({width}x{height}={width * height}) exceeds max ({config.MAX_RESOLUTION_AREA}).",
                hint="Use a smaller size or increase MAX_RESOLUTION_AREA.",
            )

    output_cfg = inp.get("output", {})
    if not isinstance(output_cfg, dict):
        raise ValidationError("output must be a JSON object.")

    ignored = [p for p in config.IGNORED_PARAMS if p in inp]

    return {
        "mode": mode,
        "image_sources": image_sources,
        # Kept for the single-image path the app uses today; the handler
        # prefers image_sources and ignores these when it is empty.
        "image_url": inp.get("image_url"),
        "image_base64": inp.get("image_base64"),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "lora_strength": lora_strength,
        "lora_strength_2": lora_strength_2,
        "lora_strength_3": lora_strength_3,
        "width": width,
        "height": height,
        "ignored_params": ignored,
        "output": {
            "bucket": output_cfg.get("bucket", config.S3_BUCKET_DEFAULT),
            "prefix": output_cfg.get("prefix", "ph3-image-comfy/"),
            "public": output_cfg.get("public", True),
            "ttl_seconds": output_cfg.get("ttl_seconds", 0),
        },
    }
