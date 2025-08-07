import math
import os
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from pathlib import Path

from .models.isosurface import MarchingCubeHelper
from .utils import (
    BaseModule,
    ImagePreprocessor,
    find_class,
    get_spherical_cameras,
    scale_tensor,
)


class TSR(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int

        image_tokenizer_cls: str
        image_tokenizer: dict

        tokenizer_cls: str
        tokenizer: dict

        backbone_cls: str
        backbone: dict

        post_processor_cls: str
        post_processor: dict

        decoder_cls: str
        decoder: dict

        renderer_cls: str
        renderer: dict

    cfg: Config

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt)
        return model

    def configure(self):
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None

    def forward(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        device: str,
    ) -> torch.FloatTensor:
        rgb_cond = self.image_processor(image, self.cfg.cond_image_size)[:, None].to(
            device
        )
        batch_size = rgb_cond.shape[0]

        input_image_tokens: torch.Tensor = self.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),
        )

        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
        )

        tokens: torch.Tensor = self.tokenizer(batch_size)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        return scene_codes

    def render(
        self,
        scene_codes,
        n_views: int,
        elevation_deg: float = 0.0,
        camera_distance: float = 1.9,
        fovy_deg: float = 40.0,
        height: int = 256,
        width: int = 256,
        return_type: str = "pil",
    ):
        rays_o, rays_d = get_spherical_cameras(
            n_views, elevation_deg, camera_distance, fovy_deg, height, width
        )
        rays_o, rays_d = rays_o.to(scene_codes.device), rays_d.to(scene_codes.device)

        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                return Image.fromarray(
                    (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
                )
            else:
                raise NotImplementedError

        images = []
        for scene_code in scene_codes:
            images_ = []
            for i in range(n_views):
                with torch.no_grad():
                    image = self.renderer(
                        self.decoder, scene_code, rays_o[i], rays_d[i]
                    )
                images_.append(process_output(image))
            images.append(images_)

        return images

    def set_marching_cubes_resolution(self, resolution: int):
        if (
            self.isosurface_helper is not None
            and self.isosurface_helper.resolution == resolution
        ):
            return
        self.isosurface_helper = MarchingCubeHelper(resolution)

    def extract_mesh(self, scene_codes, has_vertex_color, resolution: int = 256, threshold: float = 25.0):
        self.set_marching_cubes_resolution(resolution)
        meshes = []
        for scene_code in scene_codes:
            with torch.no_grad():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scale_tensor(
                        self.isosurface_helper.grid_vertices.to(scene_codes.device),
                        self.isosurface_helper.points_range,
                        (-self.renderer.cfg.radius, self.renderer.cfg.radius),
                    ),
                    scene_code,
                )["density_act"]
            v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
            v_pos = scale_tensor(
                v_pos,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )
            color = None
            if has_vertex_color:
                with torch.no_grad():
                    color = self.renderer.query_triplane(
                        self.decoder,
                        v_pos,
                        scene_code,
                    )["color"]
            mesh = trimesh.Trimesh(
                vertices=v_pos.cpu().numpy(),
                faces=t_pos_idx.cpu().numpy(),
                vertex_colors=color.cpu().numpy() if has_vertex_color else None,
            )
            meshes.append(mesh)
        return meshes
        @classmethod
    @classmethod


        
    # Ganti method from_pretrained yang ada dengan ini:
    @classmethod
    def from_pretrained(cls, model_name_or_path, config_name, weight_name):
        if Path(model_name_or_path).exists():
            config_path = Path(model_name_or_path) / config_name
            weight_path = Path(model_name_or_path) / weight_name
        else:
            config_path = hf_hub_download(model_name_or_path, config_name)
            weight_path = hf_hub_download(model_name_or_path, weight_name)
        
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        
        print("Loading SF3D checkpoint for TripoSR...")
        
        # Load checkpoint
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        
        # Debug checkpoint structure
        print("Checkpoint keys:", list(ckpt.keys()))
        
        # Extract model state dict
        if 'model_state_dict' in ckpt:
            sf3d_state_dict = ckpt['model_state_dict']
            print("Using 'model_state_dict' from SF3D checkpoint")
        elif 'state_dict' in ckpt:
            sf3d_state_dict = ckpt['state_dict']
            print("Using 'state_dict' from SF3D checkpoint")
        else:
            sf3d_state_dict = ckpt
            print("Using entire checkpoint as state_dict")
        
        print(f"SF3D state dict has {len(sf3d_state_dict)} parameters")
        
        # Analisis parameter groups dari SF3D
        sf3d_param_groups = {}
        for key in sf3d_state_dict.keys():
            prefix = key.split('.')[0] if '.' in key else key
            if prefix not in sf3d_param_groups:
                sf3d_param_groups[prefix] = 0
            sf3d_param_groups[prefix] += 1
        
        print("SF3D parameter groups:")
        for group, count in sf3d_param_groups.items():
            print(f"  {group}: {count} parameters")
        
        # Get TripoSR expected keys
        triposr_keys = set(model.state_dict().keys())
        sf3d_keys = set(sf3d_state_dict.keys())
        
        print(f"\nTripoSR expects {len(triposr_keys)} parameters")
        print(f"SF3D provides {len(sf3d_keys)} parameters")
        
        # Convert SF3D keys to TripoSR format
        converted_state_dict = {}
        used_sf3d_keys = set()
        
        # 1. Direct matches
        direct_matches = triposr_keys.intersection(sf3d_keys)
        print(f"Direct matches: {len(direct_matches)}")
        for key in direct_matches:
            converted_state_dict[key] = sf3d_state_dict[key]
            used_sf3d_keys.add(key)
        
        # 2. Try different prefix mappings
        mapping_strategies = [
            # Remove model prefix
            lambda k: k.replace('model.', '') if k.startswith('model.') else None,
            # Add model prefix
            lambda k: f'model.{k}' if f'model.{k}' in sf3d_keys else None,
            # Decoder specific mappings
            lambda k: k.replace('decoder.layers.', 'model.decoder.layers.') if k.startswith('decoder.layers.') else None,
            lambda k: k.replace('model.decoder.layers.', 'decoder.layers.') if k.startswith('model.decoder.layers.') else None,
            # Backbone mappings
            lambda k: k.replace('backbone.', 'model.backbone.') if k.startswith('backbone.') else None,
            lambda k: k.replace('model.backbone.', 'backbone.') if k.startswith('model.backbone.') else None,
            # Post processor mappings
            lambda k: k.replace('post_processor.', 'model.post_processor.') if k.startswith('post_processor.') else None,
            lambda k: k.replace('model.post_processor.', 'post_processor.') if k.startswith('model.post_processor.') else None,
        ]
        
        for triposr_key in triposr_keys:
            if triposr_key in converted_state_dict:
                continue
                
            for strategy in mapping_strategies:
                mapped_key = strategy(triposr_key)
                if mapped_key and mapped_key in sf3d_state_dict:
                    converted_state_dict[triposr_key] = sf3d_state_dict[mapped_key]
                    used_sf3d_keys.add(mapped_key)
                    break
        
        # 3. Handle specific component mappings
        # Decoder layers
        for triposr_key in triposr_keys:
            if triposr_key.startswith('decoder.layers.') and triposr_key not in converted_state_dict:
                # Try different decoder patterns
                patterns = [
                    f"model.{triposr_key}",
                    triposr_key.replace('decoder.layers.', 'model.head.layers.'),
                    triposr_key.replace('decoder.layers.', 'head.layers.'),
                    triposr_key.replace('decoder.layers.', 'model.decoder.'),
                ]
                for pattern in patterns:
                    if pattern in sf3d_state_dict:
                        converted_state_dict[triposr_key] = sf3d_state_dict[pattern]
                        used_sf3d_keys.add(pattern)
                        break
        
        print(f"\nConversion results:")
        print(f"  Converted: {len(converted_state_dict)} / {len(triposr_keys)} TripoSR parameters")
        print(f"  Used: {len(used_sf3d_keys)} / {len(sf3d_keys)} SF3D parameters")
        
        # Load converted weights
        missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
        
        print(f"\nLoading results:")
        print(f"  Missing keys: {len(missing_keys)}")
        print(f"  Unexpected keys: {len(unexpected_keys)}")
        
        if missing_keys:
            print("Missing key examples:", missing_keys[:5])
            
            # Group missing keys by component
            missing_by_component = {}
            for key in missing_keys:
                component = key.split('.')[0]
                if component not in missing_by_component:
                    missing_by_component[component] = 0
                missing_by_component[component] += 1
            
            print("Missing keys by component:")
            for component, count in missing_by_component.items():
                print(f"  {component}: {count} keys")
            
            # Initialize missing components
            if 'image_tokenizer' in missing_by_component:
                print("WARNING: image_tokenizer missing - will be randomly initialized")
                print("Consider using a pretrained ViT for better results")
            
            if 'backbone' in missing_by_component:
                print("WARNING: backbone components missing - will be randomly initialized")
        
        if unexpected_keys:
            print("Unexpected key examples:", unexpected_keys[:5])
        
        # Initialize any remaining missing parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and torch.allclose(param, torch.zeros_like(param)):
                    if 'weight' in name and len(param.shape) >= 2:
                        torch.nn.init.xavier_uniform_(param)
                    elif 'weight' in name:
                        torch.nn.init.normal_(param, std=0.02)
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param)
                    elif 'embeddings' in name:
                        torch.nn.init.normal_(param, std=0.02)
        
        print("Model loading completed!")
        return model
            print(f"Error loading state dict: {e}")
            print("Attempting to continue anyway...")
        
        return model
