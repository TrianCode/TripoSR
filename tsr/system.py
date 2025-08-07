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
        """
        Load model from Hugging Face repository
        
        Args:
            model_name_or_path: HF repo ID (e.g., "TrianC0de/TripoSR2")
            config_name: config file name (e.g., "config.yaml")  
            weight_name: checkpoint file name (e.g., "sf3d_checkpoint_epoch_3000.ckpt")
        """
        
        print(f"Loading model from Hugging Face: {model_name_or_path}")
        print(f"Config: {config_name}, Weights: {weight_name}")
        
        try:
            # Method 1: Download individual files
            print("Downloading config file...")
            config_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename=config_name,
                cache_dir=None,  # Use default HF cache
                force_download=False  # Use cached version if available
            )
            
            print("Downloading checkpoint file...")
            weight_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename=weight_name,
                cache_dir=None,
                force_download=False
            )
            
            print(f"Config downloaded to: {config_path}")
            print(f"Weights downloaded to: {weight_path}")
            
        except Exception as e:
            print(f"Error downloading from HuggingFace: {e}")
            print("Trying alternative download method...")
            
            # Method 2: Download entire repository
            try:
                repo_path = snapshot_download(
                    repo_id=model_name_or_path,
                    cache_dir=None,
                    force_download=False
                )
                config_path = os.path.join(repo_path, config_name)
                weight_path = os.path.join(repo_path, weight_name)
                
                print(f"Repository downloaded to: {repo_path}")
                
            except Exception as e2:
                print(f"Repository download also failed: {e2}")
                raise e2
        
        # Verify files exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        
        # Load config
        print("Loading configuration...")
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        
        # Initialize model
        print("Initializing model...")
        model = cls(cfg)
        
        # Load checkpoint
        print("Loading SF3D checkpoint for TripoSR...")
        print(f"Checkpoint size: {os.path.getsize(weight_path) / (1024*1024):.1f} MB")
        
        try:
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            # Try with weights_only=True
            try:
                ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
            except Exception as e2:
                print(f"Failed with both loading methods: {e2}")
                raise e2
        
        # Debug checkpoint structure
        print("\n=== CHECKPOINT ANALYSIS ===")
        print("Top-level keys:", list(ckpt.keys()))
        
        # Extract model state dict
        if 'model_state_dict' in ckpt:
            sf3d_state_dict = ckpt['model_state_dict']
            print("✓ Using 'model_state_dict' from SF3D checkpoint")
        elif 'state_dict' in ckpt:
            sf3d_state_dict = ckpt['state_dict']  
            print("✓ Using 'state_dict' from SF3D checkpoint")
        elif isinstance(ckpt, dict) and any('.' in k for k in ckpt.keys()):
            sf3d_state_dict = ckpt
            print("✓ Using entire checkpoint as state_dict (direct parameters)")
        else:
            print("❌ Could not find model parameters in checkpoint")
            print("Available keys:", list(ckpt.keys())[:10])
            raise ValueError("Invalid checkpoint format")
        
        print(f"SF3D state dict has {len(sf3d_state_dict)} parameters")
        
        # Analyze SF3D parameter structure
        sf3d_param_groups = {}
        sample_keys = []
        
        for i, key in enumerate(sf3d_state_dict.keys()):
            # Group by first component
            prefix = key.split('.')[0] if '.' in key else key
            if prefix not in sf3d_param_groups:
                sf3d_param_groups[prefix] = []
            sf3d_param_groups[prefix].append(key)
            
            # Collect samples
            if i < 10:
                sample_keys.append(key)
        
        print("\nSF3D parameter groups:")
        for group, keys in sf3d_param_groups.items():
            print(f"  {group}: {len(keys)} parameters")
            if len(keys) <= 3:
                print(f"    Keys: {keys}")
            else:
                print(f"    Sample: {keys[:3]} ...")
        
        # Get TripoSR expected structure
        triposr_state_dict = model.state_dict()
        triposr_keys = set(triposr_state_dict.keys())
        sf3d_keys = set(sf3d_state_dict.keys())
        
        print(f"\n=== MODEL COMPATIBILITY ===")
        print(f"TripoSR expects: {len(triposr_keys)} parameters")
        print(f"SF3D provides: {len(sf3d_keys)} parameters")
        
        # Analyze TripoSR structure
        triposr_param_groups = {}
        for key in triposr_keys:
            prefix = key.split('.')[0] if '.' in key else key
            if prefix not in triposr_param_groups:
                triposr_param_groups[prefix] = []
            triposr_param_groups[prefix].append(key)
        
        print("\nTripoSR expected groups:")
        for group, keys in triposr_param_groups.items():
            print(f"  {group}: {len(keys)} parameters")
        
        # Start conversion process
        print(f"\n=== CONVERSION PROCESS ===")
        converted_state_dict = {}
        used_sf3d_keys = set()
        
        # Strategy 1: Direct matches
        direct_matches = triposr_keys.intersection(sf3d_keys)
        print(f"Direct matches: {len(direct_matches)}")
        for key in direct_matches:
            converted_state_dict[key] = sf3d_state_dict[key]
            used_sf3d_keys.add(key)
        
        # Strategy 2: Prefix transformations
        prefix_mappings = [
            # Remove model prefix from SF3D
            (r'^model\.', ''),
            # Add model prefix for TripoSR
            (r'^(?!model\.)', 'model.'),
            # Specific component mappings
            (r'^model\.decoder\.', 'decoder.'),
            (r'^decoder\.', 'model.decoder.'),
            (r'^model\.backbone\.', 'backbone.'),
            (r'^backbone\.', 'model.backbone.'),
            (r'^model\.post_processor\.', 'post_processor.'),
            (r'^post_processor\.', 'model.post_processor.'),
            # Head -> decoder mapping (common in SF3D)
            (r'^model\.head\.', 'decoder.'),
            (r'^head\.', 'decoder.'),
            # Transformer specific
            (r'^model\.transformer\.', 'backbone.transformer_blocks.'),
            (r'^transformer\.', 'backbone.transformer_blocks.'),
        ]
        
        import re
        
        for triposr_key in triposr_keys:
            if triposr_key in converted_state_dict:
                continue
                
            for pattern, replacement in prefix_mappings:
                # Try forward mapping (TripoSR key -> SF3D key)
                sf3d_candidate = re.sub(pattern, replacement, triposr_key)
                if sf3d_candidate in sf3d_state_dict:
                    converted_state_dict[triposr_key] = sf3d_state_dict[sf3d_candidate]
                    used_sf3d_keys.add(sf3d_candidate)
                    break
                
                # Try reverse mapping (remove pattern from SF3D key)
                for sf3d_key in sf3d_state_dict.keys():
                    if sf3d_key in used_sf3d_keys:
                        continue
                    reverse_mapped = re.sub(replacement if replacement else r'^model\.', pattern.replace('^', ''), sf3d_key)
                    if reverse_mapped == triposr_key:
                        converted_state_dict[triposr_key] = sf3d_state_dict[sf3d_key]
                        used_sf3d_keys.add(sf3d_key)
                        break
        
        # Strategy 3: Shape-based matching for remaining keys
        remaining_triposr = triposr_keys - set(converted_state_dict.keys())
        remaining_sf3d = sf3d_keys - used_sf3d_keys
        
        if remaining_triposr and remaining_sf3d:
            print(f"Attempting shape-based matching for {len(remaining_triposr)} remaining keys...")
            
            for triposr_key in list(remaining_triposr):
                triposr_shape = triposr_state_dict[triposr_key].shape
                
                for sf3d_key in list(remaining_sf3d):
                    if sf3d_state_dict[sf3d_key].shape == triposr_shape:
                        # Additional check: similar name patterns
                        triposr_parts = triposr_key.split('.')
                        sf3d_parts = sf3d_key.split('.')
                        
                        # Check if they share similar ending (layer type)
                        if triposr_parts[-1] == sf3d_parts[-1] or triposr_parts[-2:] == sf3d_parts[-2:]:
                            converted_state_dict[triposr_key] = sf3d_state_dict[sf3d_key]
                            used_sf3d_keys.add(sf3d_key)
                            remaining_triposr.remove(triposr_key)
                            remaining_sf3d.remove(sf3d_key)
                            break
        
        print(f"\n=== CONVERSION RESULTS ===")
        print(f"Successfully converted: {len(converted_state_dict)} / {len(triposr_keys)} TripoSR parameters")
        print(f"Used from SF3D: {len(used_sf3d_keys)} / {len(sf3d_keys)} SF3D parameters")
        
        unused_sf3d = sf3d_keys - used_sf3d_keys
        missing_triposr = triposr_keys - set(converted_state_dict.keys())
        
        if unused_sf3d:
            print(f"Unused SF3D keys: {len(unused_sf3d)}")
            print(f"  Examples: {list(unused_sf3d)[:5]}")
        
        if missing_triposr:
            print(f"Missing TripoSR keys: {len(missing_triposr)}")
            print(f"  Examples: {list(missing_triposr)[:5]}")
            
            # Group missing keys by component
            missing_by_component = {}
            for key in missing_triposr:
                component = key.split('.')[0]
                missing_by_component[component] = missing_by_component.get(component, 0) + 1
            
            print("Missing keys by component:")
            for component, count in missing_by_component.items():
                print(f"  {component}: {count} parameters")
        
        # Load converted weights
        print(f"\n=== LOADING WEIGHTS ===")
        missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
        
        print(f"PyTorch loading results:")
        print(f"  Missing keys: {len(missing_keys)}")
        print(f"  Unexpected keys: {len(unexpected_keys)}")
        
        if missing_keys:
            print("  Missing examples:", missing_keys[:3])
        if unexpected_keys:
            print("  Unexpected examples:", unexpected_keys[:3])
        
        # Initialize missing parameters
        if missing_keys:
            print(f"\n=== INITIALIZING MISSING PARAMETERS ===")
            
            initialization_stats = {
                'image_tokenizer': 0,
                'backbone': 0, 
                'decoder': 0,
                'post_processor': 0,
                'other': 0
            }
            
            with torch.no_grad():
                for key in missing_keys:
                    param = dict(model.named_parameters())[key]
                    
                    # Determine component
                    component = 'other'
                    for comp in ['image_tokenizer', 'backbone', 'decoder', 'post_processor']:
                        if comp in key:
                            component = comp
                            break
                    
                    initialization_stats[component] += 1
                    
                    # Initialize based on parameter type
                    if 'weight' in key:
                        if len(param.shape) >= 2:
                            torch.nn.init.xavier_uniform_(param)
                        else:
                            torch.nn.init.normal_(param, std=0.02)
                    elif 'bias' in key:
                        torch.nn.init.zeros_(param)
                    elif 'embeddings' in key or 'embed' in key:
                        torch.nn.init.normal_(param, std=0.02)
                    elif 'norm' in key:
                        if 'weight' in key:
                            torch.nn.init.ones_(param)
                        else:
                            torch.nn.init.zeros_(param)
            
            print("Initialized parameters by component:")
            for component, count in initialization_stats.items():
                if count > 0:
                    print(f"  {component}: {count} parameters")
        
        print(f"\n=== LOADING COMPLETE ===")
        print("✓ Model successfully loaded with SF3D checkpoint")
        
        if missing_keys:
            print(f"⚠️  Warning: {len(missing_keys)} parameters were randomly initialized")
            print("   Consider fine-tuning or using a more compatible checkpoint")
        
        return model
    
    
    # Helper function to debug HuggingFace repository
    def debug_hf_repository(repo_id):
        """Debug helper to inspect HuggingFace repository contents"""
        from huggingface_hub import list_repo_files
        
        try:
            files = list_repo_files(repo_id)
            print(f"Files in {repo_id}:")
            for file in sorted(files):
                print(f"  {file}")
            return files
        except Exception as e:
            print(f"Error accessing repository {repo_id}: {e}")
            return []
