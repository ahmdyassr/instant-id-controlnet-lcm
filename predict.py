# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import time
import subprocess

import cv2
import torch
import numpy as np
from typing import List
from cog import BasePredictor, Input, Path

import PIL
from PIL import Image

import diffusers
from diffusers import (
    LCMScheduler
)
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from model_util import get_torch_device
from insightface.app import FaceAnalysis
from controlnet_util import openpose, get_depth_map, get_canny_image

from pipeline_stable_diffusion_xl_instantid_full import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps
)

# GPU global variables
DEVICE = get_torch_device()
DTYPE = torch.float16 if str(DEVICE).__contains__("cuda") else torch.float32

# CACHED REFERENES
CHECKPOINTS_URL='https://weights.replicate.delivery/default/InstantID/checkpoints.tar'
CHECKPOINTS_CACHE='./checkpoints'

LCM_LORA_URL="https://weights.replicate.delivery/default/InstantID/models--latent-consistency--lcm-lora-sdxl.tar"
LCM_LORA_CACHE='./checkpoints/models--latent-consistency--lcm-lora-sdxl'

SDXL_URL="https://weights.replicate.delivery/default/InstantID/models--stabilityai--stable-diffusion-xl-base-1.0.tar"
SDXL_CACHE="./checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0"

MODELS_URL='https://weights.replicate.delivery/default/InstantID/models.tar'
MODELS_CACHE='./checkpoints/models'

# CONTENTNET
CONTROLNET_CANNY_URL = "https://weights.replicate.delivery/default/InstantID/models--diffusers--controlnet-canny-sdxl-1.0.tar"
CONTROLNET_DEPTH_URL = "https://weights.replicate.delivery/default/InstantID/models--diffusers--controlnet-depth-sdxl-1.0-small.tar"
CONTROLNET_OPENPOSE_URL = "https://weights.replicate.delivery/default/InstantID/models--thibaud--controlnet-openpose-sdxl-1.0.tar"

CONTROLNET_CANNY_CACHE = './checkpoints/models--diffusers--controlnet-canny-sdxl-1.0'
CONTROLNET_DEPTH_CACHE = './checkpoints/models--diffusers--controlnet-depth-sdxl-1.0-small'
CONTROLNET_OPENPOSE_CACHE = './checkpoints/models--thibaud--controlnet-openpose-sdxl-1.0'

def download_weights(url, dest):
    start = time.time()
    print("⛳️ downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", "-v", url, dest], close_fds=False)
    print("✅ downloading took: ", time.time() - start)

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=PIL.Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = (
            np.array(input_image)
        )
        input_image = Image.fromarray(res)
    return input_image

def get_image_size(input_image):
    MAX_SIDE = 4096
    
    with Image.open(input_image) as img:
        width, height = img.size
    
			# Check if resizing is necessary
        if width > MAX_SIDE or height > MAX_SIDE:
            # Calculate the aspect ratio
            aspect_ratio = width / height
            # Determine the new dimensions
            if aspect_ratio > 1:
                new_width = min(width, MAX_SIDE)
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = min(height, MAX_SIDE)
                new_width = int(new_height * aspect_ratio)
                
						# assign new width and height
            width = new_width
            height = new_height 
               
            return width, height
        
    return width, height

class Predictor(BasePredictor):
    
    def setup_lcm(self):
        
        if not os.path.exists(LCM_LORA_CACHE):
          download_weights(LCM_LORA_URL, LCM_LORA_CACHE)
                
				# SETUP LCM
        self.pipe.load_lora_weights(
					"latent-consistency/lcm-lora-sdxl",
					cache_dir=CHECKPOINTS_CACHE,
					local_files_only=True,
					weight_name="pytorch_lora_weights.safetensors",
					adapter_name="lora"
        )
                        
        self.pipe.to("cuda", dtype=DTYPE)

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # CUSTOM LORA WEIGHTS
        self.tuned_model = False
        self.tuned_weights = None
        
        if not os.path.exists(CHECKPOINTS_CACHE):
          download_weights(CHECKPOINTS_URL, CHECKPOINTS_CACHE)
        
        if not os.path.exists(MODELS_CACHE):
          download_weights(MODELS_URL, MODELS_CACHE)
          
        if not os.path.exists(SDXL_CACHE):
          download_weights(SDXL_URL, SDXL_CACHE)

        # SETUP FACE ANALYSIS
        self.width, self.height = 640, 640
        self.app = FaceAnalysis(
            name="antelopev2",
            root=CHECKPOINTS_CACHE,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(self.width, self.height))
        
        # Path to InstantID models
        self.face_adapter = f"{CHECKPOINTS_CACHE}/ip-adapter.bin"
        controlnet_path = f"{CHECKPOINTS_CACHE}/ControlNetModel"

        # Load pipeline face ControlNetModel
        self.controlnet_identitynet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=DTYPE,
            cache_dir=CHECKPOINTS_CACHE,
            local_files_only=True,
        )
        self.setup_extra_controlnets()
        
        # SETUP PIPELINE
        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0',
            controlnet=[self.controlnet_identitynet],
           	torch_dtype=DTYPE,
            cache_dir=CHECKPOINTS_CACHE,
            use_safetensors=True,
            local_files_only=True,
        ).to("cuda")
                
        self.pipe.load_ip_adapter_instantid(self.face_adapter)
        
				# SETUP LCM
        self.setup_lcm()
        
        self.pipe.to("cuda")

    def setup_extra_controlnets(self):
        print(f"[~] Seting up pose, canny, depth ControlNets")
        
        if not os.path.exists(CONTROLNET_CANNY_CACHE):
          download_weights(CONTROLNET_CANNY_URL, CONTROLNET_CANNY_CACHE)
        
        if not os.path.exists(CONTROLNET_DEPTH_CACHE):
          download_weights(CONTROLNET_DEPTH_URL, CONTROLNET_DEPTH_CACHE)
        
        if not os.path.exists(CONTROLNET_OPENPOSE_CACHE):
          download_weights(CONTROLNET_OPENPOSE_URL, CONTROLNET_OPENPOSE_CACHE)

        controlnet_pose_model = "thibaud/controlnet-openpose-sdxl-1.0"
        controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0"
        controlnet_depth_model = "diffusers/controlnet-depth-sdxl-1.0-small"

        controlnet_pose = ControlNetModel.from_pretrained(
            controlnet_pose_model,
            torch_dtype=DTYPE,
            cache_dir=CHECKPOINTS_CACHE,
            local_files_only=True,
        ).to(DEVICE)
        
        controlnet_canny = ControlNetModel.from_pretrained(
            controlnet_canny_model,
            torch_dtype=DTYPE,
            cache_dir=CHECKPOINTS_CACHE,
            local_files_only=True,
        ).to(DEVICE)
        
        controlnet_depth = ControlNetModel.from_pretrained(
            controlnet_depth_model,
            torch_dtype=DTYPE,
            cache_dir=CHECKPOINTS_CACHE,
            local_files_only=True,
        ).to(DEVICE)

        self.controlnet_map = {
            "pose": controlnet_pose,
            "canny": controlnet_canny,
            "depth": controlnet_depth,
        }
        self.controlnet_map_fn = {
            "pose": openpose,
            "canny": get_canny_image,
            "depth": get_depth_map,
        }

    def generate_image(
        self,
        face_image_path,
        pose_image_path,
        prompt,
        negative_prompt,
        num_steps,
        identitynet_strength_ratio,
        adapter_strength_ratio,
        pose_strength,
        canny_strength,
        depth_strength,
        controlnet_selection,
        guidance_scale,
        seed,
        enhance_face_region
    ):
        # SETUP LCM
        self.pipe.enable_lora()
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        if face_image_path is None:
            raise Exception(
                f"Cannot find any input face `image`! Please upload the face `image`"
            )

        face_image = load_image(face_image_path)
        face_image = resize_img(face_image)
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        # Extract face features
        face_info = self.app.get(face_image_cv2)

        if len(face_info) == 0:
            raise Exception(
                "Face detector could not find a face in the `image`. Please use a different `image` as input."
            )

        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],
        )[
            -1
        ]  # only use the maximum face
        face_emb = face_info["embedding"]
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])

        img_controlnet = face_image
        if pose_image_path is not None:
            pose_image = load_image(pose_image_path)
            pose_image = resize_img(pose_image, max_side=1024)
            img_controlnet = pose_image
            pose_image_cv2 = convert_from_image_to_cv2(pose_image)

            face_info = self.app.get(pose_image_cv2)

            if len(face_info) == 0:
                raise Exception(
                    "Face detector could not find a face in the `pose_image`. Please use a different `pose_image` as input."
                )

            face_info = face_info[-1]
            face_kps = draw_kps(pose_image, face_info["kps"])

            width, height = face_kps.size

        if enhance_face_region:
            control_mask = np.zeros([height, width, 3])
            x1, y1, x2, y2 = face_info["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            control_mask[y1:y2, x1:x2] = 255
            control_mask = Image.fromarray(control_mask.astype(np.uint8))
        else:
            control_mask = None

        if len(controlnet_selection) > 0:
            controlnet_scales = {
                "pose": pose_strength,
                "canny": canny_strength,
                "depth": depth_strength,
            }
            self.pipe.controlnet = MultiControlNetModel(
                [self.controlnet_identitynet]
                + [self.controlnet_map[s] for s in controlnet_selection]
            )
            control_scales = [float(identitynet_strength_ratio)] + [
                controlnet_scales[s] for s in controlnet_selection
            ]
            control_images = [face_kps] + [
                self.controlnet_map_fn[s](img_controlnet).resize((width, height))
                for s in controlnet_selection
            ]
        else:
            self.pipe.controlnet = self.controlnet_identitynet
            control_scales = float(identitynet_strength_ratio)
            control_images = face_kps

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        print("Start inference...")
        print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

        self.pipe.set_ip_adapter_scale(adapter_strength_ratio)

        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=control_images,
            control_mask=control_mask,
            controlnet_conditioning_scale=control_scales,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator
        ).images

        return images

    def predict(
        self,
        image: Path = Input(
            description="Input face image",
        ),
        prompt: str = Input(
            description="Input prompt",
            default="a person",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="nfsw",
        ),
        num_inference_steps: int = Input(
            description="Number of LCM denoising steps",
            default=8,
            ge=1,
            le=10,
        ),
        guidance_scale: float = Input(
            description="Scale for LCM classifier-free guidance",
            default=0,
            ge=0,
            le=1,
        ),
        ip_adapter_scale: float = Input(
            description="Scale for image adapter strength (for detail)",  # adapter_strength_ratio
            default=0.8,
            ge=0,
            le=1.5,
        ),
        controlnet_conditioning_scale: float = Input(
            description="Scale for IdentityNet strength (for fidelity)",  # identitynet_strength_ratio
            default=0.8,
            ge=0,
            le=1.5,
        ),
        enable_pose_controlnet: bool = Input(
            description="Enable Openpose ControlNet, overrides strength if set to false",
            default=True,
        ),
        pose_strength: float = Input(
            description="Openpose ControlNet strength, effective only if `enable_pose_controlnet` is true",
            default=0.4,
            ge=0,
            le=1,
        ),
        enable_canny_controlnet: bool = Input(
            description="Enable Canny ControlNet, overrides strength if set to false",
            default=True,
        ),
        canny_strength: float = Input(
            description="Canny ControlNet strength, effective only if `enable_canny_controlnet` is true",
            default=0.3,
            ge=0,
            le=1,
        ),
        enable_depth_controlnet: bool = Input(
            description="Enable Depth ControlNet, overrides strength if set to false",
            default=True,
        ),
        depth_strength: float = Input(
            description="Depth ControlNet strength, effective only if `enable_depth_controlnet` is true",
            default=0.5,
            ge=0,
            le=1,
        ),
        enhance_nonface_region: bool = Input(
            description="Enhance non-face region", 
            default=True
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None,
        )
    ) -> List[Path]:
        """Run a single prediction on the model"""

        # If no seed is provided, generate a random seed
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if self.tuned_model:
            # consistency with fine-tuning API
            for k, v in self.token_map.items():
                prompt = prompt.replace(k, v)
        print(f"Prompt: {prompt}")

        # Resize the output if the provided dimensions are different from the current ones
        width, height = get_image_size(input_image = image)
        self.width = width
        self.height = height
        self.app.prepare(ctx_id=0, det_size=(self.width, self.height))
        print(f"Got image size: {self.width} x {self.height}!")

        # Set up ControlNet selection and their respective strength values (if any)
        controlnet_selection = []
        if pose_strength > 0 and enable_pose_controlnet:
            controlnet_selection.append("pose")
        if canny_strength > 0 and enable_canny_controlnet:
            controlnet_selection.append("canny")
        if depth_strength > 0 and enable_depth_controlnet:
            controlnet_selection.append("depth")

        # Generate
        images = self.generate_image(
            face_image_path=str(image),
            pose_image_path=str(image),
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=num_inference_steps,
            identitynet_strength_ratio=controlnet_conditioning_scale,
            adapter_strength_ratio=ip_adapter_scale,
            pose_strength=pose_strength,
            canny_strength=canny_strength,
            depth_strength=depth_strength,
            controlnet_selection=controlnet_selection,
            guidance_scale=guidance_scale,
            seed=seed,
            enhance_face_region=enhance_nonface_region,
        )

        # Save the generated images
        output_paths = []
        for i, output_image in enumerate(images):
            output_path = f"/tmp/out_{i}.png"
            output_image.save(output_path)
            output_paths.append(Path(output_path))
        return output_paths
