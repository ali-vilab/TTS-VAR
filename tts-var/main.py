#!/usr/bin/env python
import sys
# Ensure the project path is correct for your environment
sys.path.append("./Infinity") # Or make this an environment variable/argument
print(sys.path)
import os
# Ensure the HF_ENDPOINT is configured if needed for your environment
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
import json
import gc
import argparse # Added for command-line arguments
import random
import torch
import cv2
import numpy as np
from tqdm import tqdm

# Assuming utils_run.py contains necessary functions like load_tokenizer, load_visual_tokenizer, load_transformer,
# gen_one_img, h_div_w_templates, dynamic_resolution_h_w
from utils_run import * 
from transformers.models.t5.modeling_t5 import T5Block
from infinity.models.basic import CrossAttnBlock, SelfAttnBlock
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a single image with Infinity model using command-line arguments.")

    # Core paths
    parser.add_argument('--model_path', type=str, default='./pretrained_models/Infinity/infinity_2b_reg.pth', help='Path to the Infinity model.')
    parser.add_argument('--vae_path', type=str, default='./pretrained_models/Infinity/infinity_vae_d32reg.pth', help='Path to the VAE model.')
    parser.add_argument('--text_encoder_ckpt', type=str, default='./pretrained_models/flan-t5-xl', help='Path to the text encoder checkpoint.')

    # Generation seed
    parser.add_argument('--seed', type=int, default=None, help='Random seed for generation. If None, a random seed will be used.')

    # Generation strategy parameters
    parser.add_argument('--reward_type', type=str, default='ir', choices=['ir', 'aes', 'hps', 'no'], help='Type of reward model to use ("ir", "aes", "hps", "no").')
    parser.add_argument('--cal_type', type=str, default='value', choices=['value', 'max', 'diff', 'sum', 'no'],
                        help='Calculation type for reward-based optimization (e.g., "value", "no").')
    parser.add_argument('--resample_sis', type=int, nargs='*', default=[6,9], help='List of step indices for resampling in optimization.')
    parser.add_argument('--extract_type', type=str, default='pca', choices=['pca', 'pool', 'inception', 'no'],
                        help='Feature extraction type for reward guidance (e.g., "pca", "no").')
    parser.add_argument('--extract_sis', type=int, nargs='*', default=[2,5], help='List of step indices for feature clustering.')
    parser.add_argument('--bs_sis', type=int, nargs='+', default=[8, 8, 6, 6, 6, 4, 2, 2, 2, 1, 1, 1, 1], help='List of batch sizes for different scales of generation.')
    parser.add_argument('--lam', type=float, default=10.0, help='Lambda value for reward-based optimization.')

    # Output
    parser.add_argument('--save_dir', type=str, default='./output', help='Directory to save the generated image.')
    parser.add_argument('--output_filename', type=str, default=None, help='Optional filename for the output image. If None, a name based on prompt and seed will be generated.')

    # Prompt
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for image generation.')

    # Infinity model and generation configuration
    parser.add_argument('--pn', type=str, default='1M', choices=['0.06M', '0.25M', '0.60M', '1M'], help='Model size/parameter count identifier.')
    parser.add_argument('--cfg_insertion_layer', type=int, default=0, help='Layer index for CFG insertion.')
    parser.add_argument('--vae_type', type=int, default=32, help='VAE type.')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=1, help='Whether to add level embedding only to the first block.')
    parser.add_argument('--use_bit_label', type=int, default=1, help='Whether to use bit labels.')
    parser.add_argument('--model_type', type=str, default='infinity_2b', help='Type of the Infinity model.')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, help='Apply RoPE 2D to each self-attention layer.')
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, help='RoPE 2D normalization scheme.')
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, help='Whether to use scale schedule embedding.')
    parser.add_argument('--sampling_per_bits', type=int, default=1, help='Sampling per bits.')
    parser.add_argument('--text_channels', type=int, default=2048, help='Number of text channels.')
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, help='Whether to apply spatial patchify.')
    parser.add_argument('--h_div_w', type=float, default=1.0, help='Aspect ratio (height / width).')
    parser.add_argument('--use_flex_attn', type=int, default=0, help='Whether to use flex attention.')
    parser.add_argument('--cache_dir', type=str, default='./tmp', help='Cache directory for models or features.')
    parser.add_argument('--checkpoint_type', type=str, default='torch', help='Type of checkpoint.')
    parser.add_argument('--bf16', type=int, default=1, help='Use bfloat16 precision (1 for true, 0 for false).')
    parser.add_argument('--enable_model_cache', type=int, default=1, help='Enable model caching (1 for true, 0 for false).')
    parser.add_argument('--batch_size_namespace', type=int, default=None, help='Namespace batch_size, if needed by utilities (gen_one_img takes its own).')


    # Generation hyperparameters
    parser.add_argument('--cfg', type=float, default=3.0, help='Classifier-Free Guidance scale.')
    parser.add_argument('--tau', type=float, default=0.5, help='Tau parameter for sampling.')
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0, 1], help='Enable positive prompt usage (1 for true, 0 for false).')
    
    # Auxiliary model paths (can be hardcoded or made into args if more flexibility is needed)
    parser.add_argument('--aesthetic_predictor_path', type=str, default="./pretrained_models/aesthetic_predictor_v2_5.pth")
    parser.add_argument('--siglip_encoder_path', type=str, default="./pretrained_models/siglip-so400m-patch14-384")
    parser.add_argument('--image_reward_model_path', type=str, default="./pretrained_models/ImageReward.pt")
    parser.add_argument('--dinov2_hub_path', type=str, default="facebookresearch/dinov2")


    args = parser.parse_args()
    
    # Convert int (0/1) to bool for relevant args for convenience if your functions expect bools
    args.bf16 = bool(args.bf16)
    args.enable_model_cache = bool(args.enable_model_cache)
    args.enable_positive_prompt = bool(args.enable_positive_prompt)
    # ... and others if necessary, e.g. use_bit_label etc.
    # The original script seems to use 0/1 integers directly, so this might not be needed.

    # For Namespace compatibility if some functions in utils_run expect args.batch_size
    # The original args.batch_size was None.
    args.batch_size = args.batch_size_namespace

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    return args

def main():
    args = parse_args()

    accelerator = Accelerator()
    # torch.cuda.set_device(accelerator.device) # Accelerator handles device placement

    if accelerator.is_main_process:
        print("--- Configuration ---")
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        print("---------------------")

    # Set seed for reproducibility
    # The way seed was handled in the original script for loops:
    # args.seed = k * accelerator.num_processes + accelerator.process_index * 100
    # For single image generation, we can use a simpler scheme or let user directly control it.
    # gen_one_img takes g_seed. We use args.seed for that.
    # If running with accelerate launch and multiple processes, each might need a unique seed
    # derived from the main seed to avoid identical random streams if not handled by gen_one_img.
    # For now, we pass args.seed directly. gen_one_img might handle DDP seeding.
    current_seed = args.seed + accelerator.process_index # Simple offset for DDP
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    random.seed(current_seed)

    # Load models
    if accelerator.is_main_process: print("Loading text tokenizer and encoder...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    
    if accelerator.is_main_process: print("Loading VAE...")
    vae = load_visual_tokenizer(args) # args contains vae_path, vae_type
    
    if accelerator.is_main_process: print("Loading Infinity transformer model...")
    infinity = load_transformer(vae, args) # args contains model_path and other model config

    # Prepare models with accelerator
    text_encoder, vae, infinity = accelerator.prepare(text_encoder, vae, infinity)

    # Setup generation scale schedule
    # Ensure h_div_w_templates and dynamic_resolution_h_w are available from utils_run
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - args.h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    if accelerator.is_main_process:
        print(f"Using PN: {args.pn}, Aspect Ratio: {args.h_div_w}, Derived Template: {h_div_w_template_}")
        print(f"Scale schedule: {scale_schedule}")

    # Load reward and feature models
    rw_model = None
    aes_preprocessor = None # Specific to aesthetic_predictor
    feature_model = None

    if args.reward_type == 'aes':
        if accelerator.is_main_process: print(f"Loading Aesthetic Predictor model from {args.aesthetic_predictor_path}...")
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        assert args.aesthetic_predictor_path is not None and args.siglip_encoder_path is not None
        rw_model, aes_preprocessor = convert_v2_5_from_siglip(
            predictor_name_or_path=args.aesthetic_predictor_path,
            encoder_model_name=args.siglip_encoder_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True, # Be cautious with this
        )
        rw_model = rw_model.to(torch.bfloat16 if args.bf16 else torch.float32)
    elif args.reward_type == 'ir':
        if accelerator.is_main_process: print(f"Loading ImageReward model from {args.image_reward_model_path}...")
        import ImageReward as RM
        assert args.image_reward_model_path is not None
        rw_model = RM.load(args.image_reward_model_path, device=accelerator.device) # Pass device directly
        
        def inference_rank(self, prompt, generations_list):
            text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
            
            txt_set = []
            for generation in generations_list:
                # image encode
                if isinstance(generation, Image.Image):
                    pil_image = generation
                elif isinstance(generation, str):
                    if os.path.isfile(generation):
                        pil_image = Image.open(generation)
                else:
                    raise TypeError(r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')
                image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                image_embeds = self.blip.visual_encoder(image)
                
                # text encode cross attention with image
                image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
                text_output = self.blip.text_encoder(text_input.input_ids,
                                                        attention_mask = text_input.attention_mask,
                                                        encoder_hidden_states = image_embeds,
                                                        encoder_attention_mask = image_atts,
                                                        return_dict = True,
                                                    )
                txt_set.append(text_output.last_hidden_state[:,0,:])
                
            txt_features = torch.cat(txt_set, 0).float() # [image_num, feature_dim]
            rewards = self.mlp(txt_features) # [image_num, 1]
            rewards = (rewards - self.mean) / self.std
            rewards = torch.squeeze(rewards)
            _, rank = torch.sort(rewards, dim=0, descending=True)
            _, indices = torch.sort(rank, dim=0)
            indices = indices + 1
            
            return indices.detach().cpu().numpy().tolist(), rewards.float().detach().cpu().numpy().tolist()
        rw_model.inference_rank = inference_rank.__get__(rw_model)
        rw_model = rw_model.to(torch.bfloat16 if args.bf16 else torch.float32)
    else:
        rw_model = None
        aes_preprocessor = None


    if args.extract_type in ['pca', 'pool']:
        if accelerator.is_main_process: print(f"Loading DINOv2 model from {args.dinov2_hub_path}...")
        assert args.dinov2_hub_path is not None
        feature_model = torch.hub.load(
            args.dinov2_hub_path,
            'dinov2_vitl14_reg', # Or make this configurable
            source="local",
        )
        feature_model.eval()
        feature_model.cuda()
        feature_model = feature_model.to(torch.bfloat16 if args.bf16 else torch.float32)
    elif args.extract_type == 'inception':
        if accelerator.is_main_process: print("Loading InceptionV3 model...")
        from torchvision.models import inception_v3
        feature_model = inception_v3(weights=Inception_V3_Weights.DEFAULT) # Use updated API for pretrained
        feature_model.eval()
        feature_model.cuda()
        feature_model.fc = torch.nn.Identity()

    # Clean prompt
    # prompt_text = args.prompt[:-1] if args.prompt.endswith('.') else args.prompt
    prompt_text = args.prompt.strip() # Remove trailing spaces
    if accelerator.is_main_process: print(f"Generating image for prompt: '{prompt_text}' with seed: {args.seed}")

    # Actual generation call
    # Note: batch_size in gen_one_img is per process. bs_sis contains global batch sizes for steps.
    per_process_batch_size_step0 = args.bs_sis[0] // accelerator.num_processes
    if per_process_batch_size_step0 == 0 and accelerator.is_main_process:
        print(f"Warning: Calculated per-process batch size for step 0 is 0 (bs_sis[0]={args.bs_sis[0]}, num_processes={accelerator.num_processes}). This might cause issues. Ensure bs_sis[0] is >= num_processes.")
        # Adjust if this is problematic, e.g. ensure it's at least 1, or error out.
        # For now, let gen_one_img handle it. Some samplers might support generating 0 images per process on some ranks.
        # Or more robustly:
        if args.bs_sis[0] < accelerator.num_processes and accelerator.is_main_process:
             print(f"Warning: bs_sis[0] ({args.bs_sis[0]}) is less than num_processes ({accelerator.num_processes}). Some processes might have batch_size 0.")
        # Ensure batch_size is at least 1 for processes that are supposed to do work
        # This logic depends on how gen_one_img distributes work based on batch_size and bs_sis
        # A common pattern is that only rank 0 of a group might get a batch if total < num_processes.
        # For simplicity, we pass it as calculated.
        

    with torch.no_grad(): # Typically generation is done under no_grad for inference
        _, img_dict, score_record = gen_one_img(
            infinity_test=infinity, # Renamed for clarity, ensure gen_one_img uses this name or adjust
            vae=vae,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            prompt=prompt_text,
            g_seed=args.seed, # Global seed
            gt_leak=0, # Assuming default, make arg if needed
            gt_ls_Bl=None, # Assuming default, make arg if needed
            cfg_list=args.cfg, # Original script implies cfg_list could take a scalar and handle it. If it must be a list, this is correct.
            tau_list=args.tau, # Same as cfg_list
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[args.cfg_insertion_layer], # Original script wraps it in a list
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=args.enable_positive_prompt,
            batch_size=per_process_batch_size_step0, 
            
            accelerator=accelerator,
            bs_sis=args.bs_sis,
            rw_model=rw_model,
            aes_preprocessor=aes_preprocessor,
            cal_type=args.cal_type,
            reward=args.reward_type,
            resample_sis=args.resample_sis,
            feature_model=feature_model,
            extract_sis=args.extract_sis,
            extract_type=args.extract_type,
            _lambda=args.lam,
        )

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        if img_dict and score_record is not None:
            final_scores_values = score_record[:, -1]
            top_1_indices = final_scores_values.argsort()[-1]
            # print(f"top 1: {top_1_indices}, top 4: {top_4_indices}, top 8: {top_8_indices}")
            
            # only save best image
            generated_image = img_dict[list(img_dict.keys())[-1]][top_1_indices]
            # Assuming score_record has shape (num_candidates_in_batch, num_scores_or_steps)
            # And img_dict contains the final batch of images for the last scale
            final_images_batch = img_dict[list(img_dict.keys())[-1]]
            
            if args.output_filename:
                base_filename = args.output_filename
                if not base_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    base_filename += ".png"
            else:
                # Sanitize prompt for filename
                sane_prompt = "".join(c if c.isalnum() else "_" for c in prompt_text[:50]).rstrip('_')
                base_filename = f"{sane_prompt}_{args.seed}.png"
            
            save_file_path = os.path.join(args.save_dir, base_filename)
            
            cv2.imwrite(save_file_path, generated_image.cpu().numpy())
            print(f"Image saved to {save_file_path}")
            
        else:
            print("Image generation did not return expected image dictionary or score record on the main process.")

    accelerator.wait_for_everyone()

    # Cleanup
    if accelerator.is_main_process: print("Cleaning up resources...")
    del infinity, vae, text_encoder, rw_model, feature_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if accelerator.is_main_process: print("Done.")

if __name__ == "__main__":
    # Need to import Inception_V3_Weights for the default value in feature_model loading if used
    from torchvision.models.inception import Inception_V3_Weights
    main()
