"""Sample-addressed PIL augmentation, independent of model/global RNG."""
from __future__ import annotations

import hashlib
import io
import math
import random
from importlib.metadata import version

from ..contracts import sha256_json


class ClipTransform:
    def __init__(self, processor, *, seed=17, online=False):
        self.processor, self.seed, self.online = processor, seed, online

    @staticmethod
    def identity(processor_digest):
        return sha256_json({"version": "sample-addressed-clip-v1", "processor": processor_digest,
            "Pillow": version("Pillow"), "transformers": version("transformers"),
            "orientation": "exif-transpose-rgb", "online": "rrc224-scale.8-1-ratio.75-1.333-bicubic-flip.5"})

    def __call__(self, raw):
        return self.apply(raw, sample_id="fixed", epoch=0)

    def apply(self, raw, *, sample_id, epoch):
        from PIL import Image, ImageOps
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if not self.online:
            return self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        token = f"{self.seed}:{epoch}:{sample_id}:online-v1"
        rng = random.Random(int.from_bytes(hashlib.sha256(token.encode()).digest(), "big"))
        width, height = image.size
        for _ in range(10):
            area = width * height * rng.uniform(.8, 1.)
            ratio = math.exp(rng.uniform(math.log(.75), math.log(4 / 3)))
            crop_w, crop_h = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
            if 0 < crop_w <= width and 0 < crop_h <= height:
                left, top = rng.randint(0, width - crop_w), rng.randint(0, height - crop_h)
                break
        else:
            ratio = width / height
            crop_w, crop_h = (width, round(width / .75)) if ratio < .75 else (
                (round(height * 4 / 3), height) if ratio > 4 / 3 else (width, height))
            left, top = (width - crop_w) // 2, (height - crop_h) // 2
        image = image.crop((left, top, left + crop_w, top + crop_h)).resize((224, 224), Image.Resampling.BICUBIC)
        if rng.random() < .5:
            image = ImageOps.mirror(image)
        return self.processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"][0]
