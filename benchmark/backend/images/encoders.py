"""Frozen encoders with model-specific preprocessing; no training or text encoder."""
import gc
import importlib.metadata
import os
import threading
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator
from .. import data

GROUPS = ['hsv', 'lbp', 'glcm', 'hog', 'canny']
MODELS = {
    'handcrafted': dict(family='PA1', checkpoint='handcrafted-v1', dimension=None),
    'mobilenet_v3_large': dict(family='PA2', checkpoint='MobileNet_V3_Large_Weights.IMAGENET1K_V2', dimension=960,pretrained_parameters=5483032),
    'resnet50': dict(family='PA2', checkpoint='ResNet50_Weights.IMAGENET1K_V2', dimension=2048,pretrained_parameters=25557032),
    'efficientnet_b0': dict(family='PA2', checkpoint='EfficientNet_B0_Weights.IMAGENET1K_V1', dimension=1280,pretrained_parameters=5288548),
    'dinov2_small': dict(family='PA3', checkpoint='facebook/dinov2-small', dimension=384),
    'dinov2_base': dict(family='PA3', checkpoint='facebook/dinov2-base', dimension=768),
    'dinov2_large': dict(family='PA3', checkpoint='facebook/dinov2-large', dimension=1024),
    'clip_b32': dict(family='PA4', checkpoint='openai/clip-vit-base-patch32', dimension=512),
    'clip_b16': dict(family='PA4', checkpoint='openai/clip-vit-base-patch16', dimension=512),
    'siglip_b16': dict(family='PA4', checkpoint='google/siglip-base-patch16-224', dimension=768),
}


class EncoderConfig(BaseModel):
    model: str = 'handcrafted'
    revision: str = 'main'
    device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    batch_size: int = Field(default=8, ge=1, le=128)
    groups: list[str] = Field(default_factory=lambda: GROUPS.copy(), min_length=1)
    weights: dict[str, float] = Field(default_factory=lambda: dict.fromkeys(GROUPS, 1.0))

    @model_validator(mode='after')
    def validate_config(self):
        if self.model not in MODELS:
            raise ValueError('Unknown image model')
        if len(set(self.groups)) != len(self.groups) or set(self.groups) - set(GROUPS):
            raise ValueError('Unknown/duplicate handcrafted group')
        if set(self.weights) - set(GROUPS) or any(not np.isfinite(v) or v < 0 for v in self.weights.values()):
            raise ValueError('Weights must be finite and nonnegative')
        if not any(self.weights.get(g, 1) > 0 for g in self.groups):
            raise ValueError('At least one selected group needs positive weight')
        return self


def l2(value):
    value = np.asarray(value, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def handcrafted(image, config):
    import cv2
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
    rgb = np.asarray(image.resize((128, 128)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    features = {}
    if 'hsv' in config.groups:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        features['hsv'] = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    if 'lbp' in config.groups:
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        features['lbp'] = np.histogram(lbp, bins=np.arange(11), range=(0, 10))[0]
    if 'glcm' in config.groups:
        matrix = graycomatrix(gray // 16, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16, symmetric=True, normed=True)
        # Fixed theoretical scaling avoids fitting on test queries. Correlation in [-1,1].
        scales = {'contrast': 225., 'dissimilarity': 15., 'homogeneity': 1., 'energy': 1., 'correlation': 1., 'ASM': 1.}
        features['glcm'] = np.concatenate([graycoprops(matrix, prop).ravel() / scale for prop, scale in scales.items()])
    if 'hog' in config.groups:
        features['hog'] = hog(gray, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), block_norm='L2-Hys')
    if 'canny' in config.groups:
        edge = cv2.Canny(gray, 100, 200).astype(np.float32) / 255
        features['canny'] = edge.reshape(8,16,8,16).mean(axis=(1,3)).ravel()
    vector = np.concatenate([l2(features[g]) * config.weights.get(g,1.) for g in config.groups])
    return l2(vector)


class EncoderManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.key = self.model = self.processor = self.info = None

    def unload(self):
        with self.lock:
            self.key = self.model = self.processor = self.info = None
            gc.collect()
            import sys
            torch = sys.modules.get('torch')
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, config):
        with self.lock:
            key = config.model_dump_json()
            if self.key == key:
                return 0.
            self.unload()
            start = time.perf_counter()
            spec = MODELS[config.model]
            if config.model == 'handcrafted':
                self.info = dict(spec, device='cpu', revision='handcrafted-v1', preprocessing='EXIF transpose, RGB, resize 128x128 bicubic; HSV 8x4x4; uniform LBP P8/R1; GLCM 16 levels distances 1,3 / angles 0,45,90,135; HOG 9/16/2; Canny 100/200 8x8 density; each group L2 then weighted concat then L2')
            else:
                import torch
                device = ('cuda' if torch.cuda.is_available() else 'cpu') if config.device == 'auto' else config.device
                if device == 'cuda' and not torch.cuda.is_available():
                    raise ValueError('CUDA unavailable')
                if device=='cpu':
                    torch.set_num_threads(max(1,min(int(os.getenv('IMAGE_CPU_THREADS','4')),os.cpu_count() or 1)))
                if spec['family'] == 'PA2':
                    from torchvision import models
                    enum_class, member = spec['checkpoint'].split('.')
                    weights = getattr(getattr(models, enum_class), member)
                    model = getattr(models, config.model)(weights=weights)
                    if config.model == 'resnet50':
                        model.fc = torch.nn.Identity()
                    else:
                        model.classifier = torch.nn.Identity()
                    self.processor = weights.transforms()
                    resolved = weights.url
                    processor_info = repr(self.processor)
                else:
                    from huggingface_hub import HfApi, snapshot_download
                    from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection, SiglipVisionModel
                    from pathlib import Path
                    remote=HfApi().model_info(spec['checkpoint'],revision=config.revision)
                    use_safe=any(f.rfilename.endswith('.safetensors') for f in remote.siblings)
                    path = snapshot_download(spec['checkpoint'], revision=remote.sha,
                                             local_dir=Path(os.getenv('IMAGE_MODEL_CACHE_DIR',data.ROOT/'image_model_cache'))/config.model/remote.sha,
                                             allow_patterns=['*.json', '*.safetensors' if use_safe else 'pytorch_model*.bin'],
                                             ignore_patterns=['*optimizer*'],max_workers=1)
                    resolved = Path(path).name
                    self.processor = AutoImageProcessor.from_pretrained(path, trust_remote_code=False)
                    klass = CLIPVisionModelWithProjection if config.model.startswith('clip') else SiglipVisionModel if config.model.startswith('siglip') else AutoModel
                    model, loading_info = klass.from_pretrained(path, trust_remote_code=False, output_loading_info=True)
                    if loading_info.get('missing_keys') or loading_info.get('mismatched_keys') or loading_info.get('error_msgs'):
                        raise ValueError('Incomplete pretrained vision checkpoint: '+str(loading_info))
                    processor_info = self.processor.to_dict()
                self.model = model.eval().to(device)
                self.model.requires_grad_(False)
                self.info = dict(spec, revision=resolved, device=device, preprocessing=processor_info,
                                 gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
                                 pooling='CLS' if spec['family']=='PA3' else 'projected image embedding' if spec['family']=='PA4' else 'global average pool before classifier', normalize='L2')
                self.info.update(feature_parameters=sum(p.numel() for p in self.model.parameters()),
                                 frozen=not any(p.requires_grad for p in self.model.parameters()),
                                 cpu_threads=torch.get_num_threads())
            self.info['libraries']={name:importlib.metadata.version(name) for name in ('torch','torchvision','transformers','Pillow','scikit-image')}
            self.key = key
            return (time.perf_counter() - start) * 1000

    def sync(self):
        if self.info and self.info['device']=='cuda':
            import torch
            torch.cuda.synchronize()

    def encode(self, images, config):
        prep_start = time.perf_counter()
        if config.model == 'handcrafted':
            vectors = np.stack([handcrafted(im,config) for im in images])
            return vectors, dict(preprocessing_ms=0., embedding_ms=(time.perf_counter()-prep_start)*1000)
        import torch
        if MODELS[config.model]['family']=='PA2':
            inputs = torch.stack([self.processor(im) for im in images]).to(self.info['device'])
        else:
            inputs = self.processor(images=images, return_tensors='pt')['pixel_values'].to(self.info['device'])
        self.sync()
        preprocessing_ms = (time.perf_counter()-prep_start)*1000
        start = time.perf_counter()
        with torch.inference_mode():
            if MODELS[config.model]['family']=='PA2':
                output = self.model(inputs)
            else:
                result = self.model(pixel_values=inputs)
                output = result.last_hidden_state[:,0] if MODELS[config.model]['family']=='PA3' else result.image_embeds if config.model.startswith('clip') else result.pooler_output
            vectors = torch.nn.functional.normalize(output.float(),dim=-1).cpu().numpy()
        self.sync()
        if vectors.shape[1] != MODELS[config.model]['dimension']:
            raise ValueError('Checkpoint output dimension differs from model registry')
        return vectors, dict(preprocessing_ms=preprocessing_ms, embedding_ms=(time.perf_counter()-start)*1000)


manager = EncoderManager()
