"""
VGGT-based Vision-Language-Action Model
Main model that integrates VGGT, language encoder, and action prediction
"""

import torch
import torch.nn as nn
import os
import logging
import sys
from transformers import AutoModel, AutoTokenizer

# Import VGGT - try package import first, fallback to relative import
try:
    from vggt.models.vggt import VGGT
except ImportError:
    # Fallback: add parent directory to path for development
    import sys
    import os
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from vggt.models.vggt import VGGT

from .geometry_encoder import GeometryFeatureExtractor
from .fusion import MultimodalFusion
# from .action_head import ActionHead # 修改
from .flow_matching_action_head import FlowMatchingActionHead

class VGGTVLA(nn.Module):
    """
    VGGT-based Vision-Language-Action Model
    
    Architecture:
    1. VGGT backbone: extracts 3D geometric information from images
    2. Language encoder: processes language instructions (LLaMA 3 encoder)
    3. Geometry encoder: extracts features from VGGT outputs //真的需要这一步吗？
    4. Multimodal fusion: fuses language and geometry features
    5. Action head: predicts robot actions //这个是怎样的一个结构？
    
    Args:
        vggt_model: Pre-trained VGGT model (optional, will load if None)
        lang_encoder_name: HuggingFace model name for language encoder
        freeze_vggt: Whether to freeze VGGT parameters
        freeze_lang_encoder: Whether to freeze language encoder parameters
        geom_output_dim: Output dimension for geometry features
        fusion_hidden_dim: Hidden dimension for fusion module
        action_dim: Action dimension (default 7: 6-DOF pose + gripper)
    """
    
    def __init__(self, 
                 vggt_model=None,
                 lang_encoder_name="meta-llama/Meta-Llama-3-8B",
                 freeze_vggt=True,
                 freeze_lang_encoder=False,
                 geom_output_dim=512, #VGGT的输出维度
                 fusion_hidden_dim=1024, #融合模块的隐藏维度
                 action_dim=7, #动作维度
                 use_pointnet=True, #是否使用点云
                 use_pose=True, #是否使用姿态
                 hf_token=None, #HuggingFace token
                 **kwargs):
        super().__init__()
        
        # 1. VGGT backbone
        if vggt_model is None:
            print("Loading VGGT model from pretrained...")
            self.vggt = VGGT.from_pretrained("facebook/VGGT-1B")
        else:
            self.vggt = vggt_model
            
        self.freeze_vggt = freeze_vggt
        if freeze_vggt:
            # Freeze VGGT parameters
            for param in self.vggt.parameters():
                param.requires_grad = False
            self.vggt.eval()
            print("VGGT model frozen")
        else:
            print("VGGT model trainable")
            
        # 2. Language encoder (LLaMA encoder)
        print(f"Loading language encoder: {lang_encoder_name}")
        
        # Get HuggingFace token: priority: parameter > kwargs > environment variable
        if hf_token is None:
            hf_token = kwargs.get('hf_token')
        if hf_token is None:
            hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
        
        try:
            # Ensure token is set in environment for transformers library
            # Transformers will automatically use HF_TOKEN from environment
            if hf_token:
                os.environ['HF_TOKEN'] = hf_token
                os.environ['HUGGINGFACE_TOKEN'] = hf_token
                logging.info(f"HuggingFace token set in environment for model loading")
                print(f"Using HuggingFace token for authentication", flush=True)
            else:
                # Try to get from environment if not provided
                hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
                if not hf_token:
                    logging.warning("No HuggingFace token found!")
                    print("Warning: No HuggingFace token provided.", flush=True)
                    print("  Trying to load from environment variables or huggingface-cli login...", flush=True)
                else:
                    logging.info(f"Using HF_TOKEN from environment")
            
            # Load model - explicitly pass token
            logging.info(f"Loading language encoder: {lang_encoder_name}")
            print(f"Loading model: {lang_encoder_name}", flush=True)
            sys.stdout.flush()
            
            # Prepare load_kwargs with token
            load_kwargs = {
                'trust_remote_code': True,
            }
            if hf_token:
                load_kwargs['token'] = hf_token
                logging.info(f"Token added to load_kwargs")
            else:
                logging.warning(f"No token available for model loading")
            
            logging.info(f"load_kwargs: {load_kwargs}")
            
            # Try loading config first to debug and ensure token is working
            try:
                from transformers import AutoConfig
                print(f"Attempting to load config with token: {'Yes' if hf_token else 'No'}", flush=True)
                logging.info(f"Attempting to load config with token: {'Yes' if hf_token else 'No'}")
                if hf_token:
                    print(f"Token (first 15 chars): {hf_token[:15]}...", flush=True)
                    logging.info(f"Token for config: {hf_token[:15]}...")
                print(f"load_kwargs keys: {list(load_kwargs.keys())}", flush=True)
                logging.info(f"load_kwargs: {load_kwargs}")
                config_obj = AutoConfig.from_pretrained(lang_encoder_name, **load_kwargs)
                print(f"Config loaded successfully. Model type: {config_obj.model_type}", flush=True)
                logging.info(f"Config loaded successfully. Model type: {config_obj.model_type}")
            except Exception as config_error:
                print(f"ERROR: Failed to load config: {config_error}")
                print(f"Token provided: {'Yes' if hf_token else 'No'}")
                if hf_token:
                    print(f"Token value: {hf_token[:20]}...")
                print(f"Environment HF_TOKEN: {os.environ.get('HF_TOKEN', 'Not set')[:20] + '...' if os.environ.get('HF_TOKEN') else 'Not set'}")
                print(f"load_kwargs: {load_kwargs}")
                raise  # Don't continue if config fails
            
            # Load model with explicit token
            print(f"Loading model with token: {'Yes' if hf_token else 'No'}")
            self.lang_encoder = AutoModel.from_pretrained(
                lang_encoder_name,
                **load_kwargs
            )
            print(f"Loading tokenizer: {lang_encoder_name}")
            self.lang_tokenizer = AutoTokenizer.from_pretrained(
                lang_encoder_name,
                **load_kwargs
            )
            # Set pad token if not exists
            if self.lang_tokenizer.pad_token is None: # 后面check 
                self.lang_tokenizer.pad_token = self.lang_tokenizer.eos_token
                logging.info("Set pad_token = eos_token")
            
            logging.info("Language encoder and tokenizer ready")
            print(f"Language encoder and tokenizer ready", flush=True)
        except Exception as e:
            logging.error(f"Error: Could not load {lang_encoder_name}: {e}")
            print(f"Error: Could not load {lang_encoder_name}: {e}", flush=True)
            print("Please check:", flush=True)
            print("  1. Model name is correct", flush=True)
            print("  2. You have access to the model", flush=True)
            print("  3. HuggingFace token is set (via config or HF_TOKEN env var)", flush=True)
            print(f"  4. Environment HF_TOKEN: {os.environ.get('HF_TOKEN', 'NOT SET')[:20]}...", flush=True)
            raise
            
        self.freeze_lang_encoder = freeze_lang_encoder
        if freeze_lang_encoder:
            # Freeze language encoder parameters
            for param in self.lang_encoder.parameters():
                param.requires_grad = False
            print("Language encoder frozen")
        else:
            print("Language encoder trainable")
            
        # Get language encoder hidden size
        lang_hidden_size = self.lang_encoder.config.hidden_size
        
        # 3. Geometry feature extractor
        self.geom_encoder = GeometryFeatureExtractor(
            token_dim=2048,  # 2 * embed_dim from VGGT
            output_dim=geom_output_dim,
            use_pointnet=use_pointnet,
            use_pose=use_pose
        )
        
        # 4. Multimodal fusion
        # 改进5: 支持attention pooling选项
        use_attention_pooling = kwargs.get("use_attention_pooling", True)
        self.fusion = MultimodalFusion(
            lang_dim=lang_hidden_size,
            geom_dim=geom_output_dim,
            hidden_dim=fusion_hidden_dim,
            use_attention_pooling=use_attention_pooling
        )
        
        # 5. Action prediction head
        # 改进3: 支持四元数表示
        use_quaternion = kwargs.get("use_quaternion", False)
        self.action_head = FlowMatchingActionHead(
            action_dim=7,
            action_horizon=16,  # 预测未来16步
            input_dim=fusion_hidden_dim,
            use_token_context=True,  # 利用token序列而不是单向量
            hidden_dim=256,
            num_layers=4,
            num_inference_steps=10,
            ode_solver="midpoint",  # 比euler更准确
            cfg_scale=1.5,  # Classifier-Free Guidance
        )
        
    def forward(self, images, language_instruction, return_intermediates=False):
        """
        Forward pass
        
        Args:
            images: [B, S, 3, H, W] or [S, 3, H, W] - Input images in range [0, 1]
            language_instruction: List[str] or dict - Language instructions
            return_intermediates: bool - Whether to return intermediate features
            
        Returns:
            dict containing:
                - action: [B, action_dim] - Predicted actions
                - pose: [B, 6] - End-effector pose
                - gripper: [B, 1] - Gripper action
                - (optional intermediates if return_intermediates=True)
        """
        # Handle batch dimension
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, S = images.shape[:2]
        device = images.device
        
        # 1. VGGT forward pass
        if self.freeze_vggt:
            with torch.no_grad():
                # Get aggregated tokens
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images)
                
                # Get VGGT predictions (optional, for geometry features)
                vggt_outputs = self.vggt(images)
        else:
            aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images)
            vggt_outputs = self.vggt(images)
            
        # 2. Extract geometry features
        geom_features = self.geom_encoder(
            aggregated_tokens_list=aggregated_tokens_list,
            world_points=vggt_outputs.get("world_points"), # 那和pointnet 有什么区别
            depth=vggt_outputs.get("depth"),
            pose_enc=vggt_outputs.get("pose_enc")
        )  # [B, S, geom_output_dim]
        
        # 3. Encode language instructions
        if isinstance(language_instruction, list):
            # Tokenize text
            lang_inputs = self.lang_tokenizer(
                language_instruction,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128 # 够用吗？
            ).to(device)
        else:
            # Assume already tokenized
            lang_inputs = language_instruction
            
        if self.freeze_lang_encoder:
            with torch.no_grad():
                lang_outputs = self.lang_encoder(**lang_inputs)
        else:
            lang_outputs = self.lang_encoder(**lang_inputs)
            
        lang_features = lang_outputs.last_hidden_state  # [B, L, lang_hidden_size]
        
        # 4. Multimodal fusion
        fused_features = self.fusion(lang_features, geom_features)  # [B, fusion_hidden_dim]
        
        # 5. Predict actions
        action_outputs = self.action_head(fused_features)
        
        # Prepare return dictionary
        outputs = {
            "action": action_outputs["action"],
            "pose": action_outputs["pose"],
            "gripper": action_outputs["gripper"]
        }
        
        if return_intermediates:
            outputs.update({
                "geometry_features": geom_features,
                "language_features": lang_features,
                "fused_features": fused_features,
                "vggt_outputs": vggt_outputs
            })
            
        return outputs
    
    def encode_language(self, language_instruction):
        """
        Encode language instructions separately (useful for caching)
        
        Args:
            language_instruction: List[str] or dict
            
        Returns:
            lang_features: [B, L, lang_hidden_size]
        """
        if isinstance(language_instruction, list):
            lang_inputs = self.lang_tokenizer(
                language_instruction,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(next(self.parameters()).device)
        else:
            lang_inputs = language_instruction
            
        if self.freeze_lang_encoder:
            with torch.no_grad():
                lang_outputs = self.lang_encoder(**lang_inputs)
        else:
            lang_outputs = self.lang_encoder(**lang_inputs)
            
        return lang_outputs.last_hidden_state
