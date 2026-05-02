import os
import argparse
import pandas as pd
import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import warnings

# VideoMAE imports are deferred until the extractor is requested so users who
# only need CLIP don't have to install the heavier transformers extras.

# Ignore routine transformers warnings
warnings.filterwarnings("ignore")

class BaseVideoExtractor:
    """Base class for video feature extraction. All new models should inherit from this."""
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.setup_model()

    def setup_model(self):
        raise NotImplementedError("Subclasses must implement the model loading logic.")

    def extract_features(self, frames: list) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement the feature extraction logic.")

class CLIPExtractor(BaseVideoExtractor):
    """CLIP model feature extractor."""
    def setup_model(self):
        print(f"Loading CLIP model from {self.model_path}...")
        self.processor = CLIPProcessor.from_pretrained(self.model_path)
        self.model = CLIPModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, frames: list) -> np.ndarray:
        # frames is a list of PIL Images
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        
        # Get only visual features ([CLS] token representation) and normalize
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # Return numpy array of shape (T, D), e.g., (T, 512) for CLIP-ViT-B/32
        return image_features.cpu().numpy()

class VideoMAEExtractor(BaseVideoExtractor):
    """VideoMAE feature extractor.

    VideoMAE expects a fixed 16-frame clip. This extractor uniformly samples
    16 frames from the input frame list, runs them through VideoMAE, and
    returns the mean-pooled hidden state -> shape (1, hidden_size).
    """

    NUM_REQUIRED_FRAMES = 16

    def setup_model(self):
        from transformers import VideoMAEImageProcessor, VideoMAEModel
        print(f"Loading VideoMAE model from {self.model_path}...")
        self.processor = VideoMAEImageProcessor.from_pretrained(self.model_path)
        self.model = VideoMAEModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, frames: list) -> np.ndarray:
        # frames: list of PIL images. Resample to exactly 16 frames.
        if len(frames) == 0:
            raise ValueError("Empty frame list passed to VideoMAEExtractor")
        if len(frames) != self.NUM_REQUIRED_FRAMES:
            indices = np.linspace(0, len(frames) - 1, self.NUM_REQUIRED_FRAMES).astype(int)
            frames = [frames[i] for i in indices]

        inputs = self.processor(frames, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # last_hidden_state: (1, num_patches, hidden_size). Mean-pool over patches.
        pooled = outputs.last_hidden_state.mean(dim=1)
        # Return shape (1, D) for consistency with CLIPExtractor's (T, D).
        return pooled.cpu().numpy()


class DinoV2Extractor(BaseVideoExtractor):
    """DINOv2 image encoder (per-frame). Uses pooler_output (CLS token)
    L2-normalized; output shape (T, D)."""

    def setup_model(self):
        from transformers import AutoImageProcessor, AutoModel
        print(f"Loading DINOv2 model from {self.model_path}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, frames: list) -> np.ndarray:
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        feat = outputs.pooler_output  # (T, D)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.cpu().numpy()


class SigLIPExtractor(BaseVideoExtractor):
    """SigLIP image encoder (per-frame). Uses get_image_features(),
    L2-normalized; output shape (T, D)."""

    def setup_model(self):
        from transformers import AutoProcessor, AutoModel
        print(f"Loading SigLIP model from {self.model_path}...")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, frames: list) -> np.ndarray:
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        feat = self.model.get_image_features(**inputs)  # (T, D)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.cpu().numpy()


class TimeSformerExtractor(BaseVideoExtractor):
    """TimeSformer video transformer.

    Expects 8 input frames (default for k400 finetune). Returns mean-pooled
    last_hidden_state -> shape (1, D).
    """

    NUM_REQUIRED_FRAMES = 8

    def setup_model(self):
        from transformers import AutoImageProcessor, TimesformerModel
        print(f"Loading TimeSformer model from {self.model_path}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = TimesformerModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, frames: list) -> np.ndarray:
        if len(frames) != self.NUM_REQUIRED_FRAMES:
            indices = np.linspace(0, len(frames) - 1, self.NUM_REQUIRED_FRAMES).astype(int)
            frames = [frames[i] for i in indices]
        inputs = self.processor(frames, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return pooled.cpu().numpy()


# ================= Model Factory =================
EXTRACTOR_REGISTRY = {
    "clip": CLIPExtractor,
    "clip-base": CLIPExtractor,
    "clip-large": CLIPExtractor,
    "videomae-base": VideoMAEExtractor,
    "videomae-large": VideoMAEExtractor,
    "dinov2-base": DinoV2Extractor,
    "dinov2-large": DinoV2Extractor,
    "siglip-base": SigLIPExtractor,
    "timesformer-k400": TimeSformerExtractor,
}

def get_extractor(feature_name: str, model_path: str, device: str) -> BaseVideoExtractor:
    if feature_name not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unsupported feature model: {feature_name}. Currently supported: {list(EXTRACTOR_REGISTRY.keys())}")
    return EXTRACTOR_REGISTRY[feature_name](model_path, device)

# ================= Video Processing Tools =================
def extract_frames_uniformly(video_path: str, num_frames: int) -> list:
    """Uniformly sample a specified number of frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError(f"No frames found in video: {video_path}")

    # Generate uniformly distributed frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV reads in BGR by default, convert to RGB, then to PIL Image for transformers
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        else:
            # If a frame fails to read, pad with the last successfully read frame
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                raise RuntimeError(f"Failed to read the first frame of video: {video_path}")
                
    cap.release()
    return frames

# ================= Main Flow =================
def main():
    parser = argparse.ArgumentParser(description="NeuroMM 2026 Batch Video Feature Extraction Tool")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the main manifest file (.csv)")
    parser.add_argument("--raw-video-dir", type=str, required=True, help="Root directory of raw videos")
    parser.add_argument("--output-dir", type=str, required=True, help="Root directory for feature output (e.g., processed/features/)")
    parser.add_argument(
        "--feature-name",
        type=str,
        required=True,
        choices=sorted(EXTRACTOR_REGISTRY.keys()),
        help="Type of feature to extract",
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to local model weights")
    parser.add_argument("--num-frames", type=int, default=8, help="Total number of frames to sample per video (default: 8)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    
    args = parser.parse_args()

    # Create output directory for the specific feature
    feature_output_dir = os.path.join(args.output_dir, args.feature_name)
    os.makedirs(feature_output_dir, exist_ok=True)

    # Initialize feature extractor
    extractor = get_extractor(args.feature_name, args.model_path, args.device)

    # Load manifest
    print(f"Loading manifest from {args.manifest}...")
    df = pd.read_csv(args.manifest)
    
    # Filter out invalid records missing raw video paths
    if 'raw_video_relpath' not in df.columns:
        raise ValueError("Manifest file must contain 'raw_video_relpath' column")
        
    df = df.dropna(subset=['raw_video_relpath'])
    
    print(f"Found {len(df)} videos to process.")
    
    success_cnt = 0
    fail_cnt = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {args.feature_name}"):
        sample_id = row['sample_id']
        video_relpath = row['raw_video_relpath']
        video_path = os.path.join(args.raw_video_dir, video_relpath)
        output_npy_path = os.path.join(feature_output_dir, f"{sample_id}.npy")

        # Skip existing features to support resuming from interruptions
        if os.path.exists(output_npy_path):
            success_cnt += 1
            continue

        if not os.path.exists(video_path):
            print(f"\n[Warning] Video file not found, skipping: {video_path}")
            fail_cnt += 1
            continue

        try:
            # 1. Uniformly sample frames
            frames = extract_frames_uniformly(video_path, args.num_frames)
            # 2. Extract features -> shape: (num_frames, feature_dim)
            features = extractor.extract_features(frames)
            # 3. Save as .npy, named only by sample_id
            np.save(output_npy_path, features)
            success_cnt += 1
        except Exception as e:
            print(f"\n[Error] Failed to process video {sample_id}: {str(e)}")
            fail_cnt += 1

    print("\nExtraction Complete!")
    print(f"Target directory: {feature_output_dir}")
    print(f"Success: {success_cnt} | Failed: {fail_cnt} | Output shape per file: ({args.num_frames}, D)")

if __name__ == "__main__":
    main()