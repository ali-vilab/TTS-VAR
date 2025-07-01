import math
import random
import time
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any
import gc
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from timm.models import register_model
from torch.utils.checkpoint import checkpoint
from PIL import Image
import numpy as np
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import broadcast

import infinity.utils.dist as dist
from infinity.utils.dist import for_visualize
from infinity.models.basic import flash_attn_func, flash_fused_op_installed, AdaLNBeforeHead, CrossAttnBlock, SelfAttnBlock, CrossAttention, FastRMSNorm, precompute_rope2d_freqs_grid
from infinity.utils import misc
from infinity.models.flex_attn import FlexAttn
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

try:
    from infinity.models.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None

def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

@torch.no_grad()
def cus_autoregressive_infer_cfg_dino(
    model,
    vae=None,
    scale_schedule=None,
    label_B_or_BLT=None,
    B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
    g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,
    returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,
    cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],
    vae_type=0, softmax_merge_topk=-1, ret_img=False,
    trunk_scale=1000,
    gt_leak=0, gt_ls_Bl=None,
    inference_mode=False,
    save_img_path=None,
    sampling_per_bits=1,
    
    prompt=None,
    accelerator: Accelerator = None,
    bs_sis=[16, 16, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4],
    rw_model=None,
    aes_preprocessor=None,
    cal_type='value',
    reward='ir',
    resample_sis=[6, 8, 10],
    feature_model=None,
    extract_sis=[2, 5],
    extract_type="pca",
    _lambda=10,
):   # returns List[idx_Bl]
    if g_seed is None: rng = None
    else: model.rng.manual_seed(g_seed); rng = model.rng
    assert len(cfg_list) >= len(scale_schedule)
    assert len(tau_list) >= len(scale_schedule)

    # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify, 
    # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
    if model.apply_spatial_patchify:
        vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
    else:
        vae_scale_schedule = scale_schedule
    
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
    if any(np.array(cfg_list) != 1):
        bs = 2*B
        if not negative_label_B_or_BLT:
            kv_compact_un = kv_compact.clone()
            total = 0
            # print(f'kv_compact_un shape: {kv_compact_un.shape}, lens: {lens}, cu_seqlens_k: {cu_seqlens_k}, max_seqlen_k: {max_seqlen_k}, model.cfg_uncond: {model.cfg_uncond}')
            for le in lens:
                kv_compact_un[total:total+le] = (model.cfg_uncond)[:le]
                total += le
            kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
            cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
        else:
            kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
            kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
            cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
            max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
    else:
        bs = B

    kv_compact = model.text_norm(kv_compact)
    sos = cond_BD = model.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) # sos shape: [2, 4096]
    # sos = cond_BD = model.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).repeat_interleave(B, dim=0)
    kv_compact = model.text_proj_for_ca(kv_compact) # kv_compact shape: [304, 4096]
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + model.pos_start.expand(bs, 1, -1)

    with torch.amp.autocast('cuda', enabled=False):
        cond_BD_or_gss = model.shared_ada_lin(cond_BD.float()).float().contiguous()
    accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
    idx_Bl_list, idx_Bld_list = [], []

    if inference_mode:
        for b in model.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
    else:
        assert model.num_block_chunks > 1
        for block_chunk_ in model.block_chunks:
            for module in block_chunk_.module.module:
                (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(True)
    
    abs_cfg_insertion_layers = []
    add_cfg_on_logits, add_cfg_on_probs = False, False
    leng = len(model.unregistered_blocks)
    for item in cfg_insertion_layer:
        if item == 0: # add cfg on logits
            add_cfg_on_logits = True
        elif item == 1: # add cfg on probs
            add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
        elif item < 0: # determine to add cfg at item-th layer's output
            assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={model.num_block_chunks}'
            abs_cfg_insertion_layers.append(leng+item)
        else:
            raise ValueError(f'cfg_insertion_layer: {item} is not valid')
    
    num_stages_minus_1 = len(scale_schedule)-1
    summed_codes = 0
    
    img_dict = {}
    scores_record = torch.zeros((bs_sis[-1] // accelerator.num_processes, len(scale_schedule))).cuda()
    
    progress_bar = tqdm(total=len(scale_schedule), desc='Generating images')
    for si, pn in enumerate(scale_schedule):   # si: i-th segment
        # accelerator.print(f"\tsi: {si}, pn: {pn}")
        
        cfg = cfg_list[si]
        if si >= trunk_scale:
            break
        cur_L += np.array(pn).prod()

        need_to_pad = 0
        attn_fn = None
        if model.use_flex_attn:
            # need_to_pad = (model.pad_to_multiplier - cur_L % model.pad_to_multiplier) % model.pad_to_multiplier
            # if need_to_pad:
            #     last_stage = F.pad(last_stage, (0, 0, 0, need_to_pad))
            attn_fn = model.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

        # assert model.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(model.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {model.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
        layer_idx = 0
        for block_idx, b in enumerate(model.block_chunks):
            # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
            if model.add_lvl_embeding_only_first_block and block_idx == 0:
                last_stage = model.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
            if not model.add_lvl_embeding_only_first_block: 
                last_stage = model.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
            
            for m in b.module:
                last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=model.rope2d_freqs_grid, scale_ind=si)
                if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                    # print(f'add cfg={cfg} on {layer_idx}-th layer output')
                    last_stage = cfg * last_stage[:B] + (1-cfg) * last_stage[B:]
                    last_stage = torch.cat((last_stage, last_stage), 0)
                layer_idx += 1
        # accelerator.print(f"Step {si} CUDA Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, Released: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
                
        if (cfg != 1) and add_cfg_on_logits:
            # print(f'add cfg on add_cfg_on_logits')
            logits_BlV = model.get_logits(last_stage, cond_BD).mul(1/tau_list[si])
            logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
        else:
            logits_BlV = model.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
        
        if model.use_bit_label:
            tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
            logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
            idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or model.top_k, top_p=top_p or model.top_p, num_samples=1)[:, :, 0]
            idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
        else:
            idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or model.top_k, top_p=top_p or model.top_p, num_samples=1)[:, :, 0]
            
        assert returns_vemb
        if si < gt_leak:
            idx_Bld = gt_ls_Bl[si]
        else:
            assert pn[0] == 1
            idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d]
            if model.apply_spatial_patchify: # unpatchify operation
                idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
                idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
                idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
            idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

        idx_Bld_list.append(idx_Bld)
        codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
        if si != num_stages_minus_1:
            # print(si, summed_codes.shape if not isinstance(summed_codes, int) else summed_codes, codes.shape)
            summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
            last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
            last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
            if model.apply_spatial_patchify: # patchify operation
                last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
            last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
            last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
        else:
            summed_codes += codes
        # accelerator.print(f"BS {summed_codes.shape}")
        # accelerator.print(f"Step {si} CUDA Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, Released: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
          
        with torch.inference_mode():
            max_split = 8
            if B <= max_split:
                scale_img = vae.decode(summed_codes.squeeze(-3))
                scale_img = (scale_img + 1) / 2
                scale_img = scale_img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
            else:
                scale_imgs = []
                for i in range(0, B, max_split):
                    scale_imgs.append(vae.decode(summed_codes[i:i+max_split].squeeze(-3)))
                scale_img = torch.cat(scale_imgs, dim=0)
                scale_img = (scale_img + 1) / 2
                scale_img = scale_img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
            
            # gc.collect()
            # torch.cuda.empty_cache()
            
        # accelerator.print(f"Step {si} CUDA Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, Released: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
        
        # dino + kmeans
        if accelerator and si in extract_sis and extract_type != 'no':
            assert bs_sis[si-1] > bs_sis[si]
            
            all_imgs = accelerator.gather(scale_img)
            
            assert bs_sis[si] % accelerator.num_processes == 0
            local_bs = bs_sis[si] // accelerator.num_processes
            all_replace_ids = torch.zeros(bs_sis[si], dtype=torch.long, device=accelerator.device)
            
            if accelerator.is_main_process:
                all_replace_ids = extract_k_ids(
                    feature_model=feature_model,
                    imgs=all_imgs.flip(dims=(3, )),
                    source_num=bs_sis[si-1],
                    target_num=bs_sis[si],
                    patch_size=16,
                    extract_type=extract_type,
                )
                # accelerator.split_between_processes
            accelerator.wait_for_everyone()
            all_replace_ids = broadcast(all_replace_ids.to(accelerator.device))
            
            all_replace_ids = all_replace_ids.cpu()
            # print(accelerator.process_index, all_replace_ids)
            
            replace_ids = all_replace_ids[accelerator.process_index*local_bs:(accelerator.process_index+1)*local_bs]
            
            all_summed_codes = accelerator.gather(summed_codes) # [tot_B, d, 1, h, w]
            all_last_stage = accelerator.gather(last_stage) # [tot_B, d, h, w] or [tot_B, d, 2h, 2w]
            
            # accelerator.print(f"Step {si} ori: \n\tB: {B}, \n\tbs: {bs}, \n\tlast_stage_shape: {last_stage.shape}, \n\tsummed_code_shape: {summed_codes.shape}, \n\tcond_BD_shape: {cond_BD.shape}, \n\tcond_shape: {cond_BD_or_gss.shape}, \n\tca_kv_0_shape: {ca_kv[0].shape}, \n\tca_kv_1_shape: {ca_kv[1]}, \n\tca_kv_2: {ca_kv[2]}")
            summed_codes = all_summed_codes[replace_ids]
            last_stage = all_last_stage[replace_ids]
            
            local_kv_cache = sync_kv_cache(
                    model, sum_kv_cache(model), replace_ids=replace_ids, accelerator=accelerator,
            )
            set_kv_cache(model, local_kv_cache)
            # accelerator.print("kv cache synced")
            
            B, bs, cond_BD, cond_BD_or_gss, ca_kv = rebatch(
                local_bs, B, bs, cond_BD, cond_BD_or_gss, ca_kv, replace_ids=replace_ids
            )
            # accelerator.print(f"Step {si}: \n\tB: {B}, \n\tbs: {bs}, \n\tlast_stage_shape: {last_stage.shape}, \n\tsummed_code_shape: {summed_codes.shape}, \n\tcond_BD_shape: {cond_BD.shape}, \n\tcond_shape: {cond_BD_or_gss.shape}, \n\tca_kv_0_shape: {ca_kv[0].shape}, \n\tca_kv_1_shape: {ca_kv[1]}, \n\tca_kv_2: {ca_kv[2]}\n")
        
        # reward resample
        if accelerator and si in resample_sis and cal_type != 'no':
            all_summed_codes = accelerator.gather(summed_codes) # [tot_B, d, 1, h, w]
            all_last_stage = accelerator.gather(last_stage) # [tot_B, d, h, w] or [tot_B, d, 2h, 2w]
            all_scores_record = accelerator.gather(scores_record) # [tot_B, len(scale_schedule)]
            
            assert cal_type not in ['max', 'sum', 'diff', 'topk']
            score, replace_rate = calculate_replace_rate(
                rw_model=rw_model,
                aes_preprocessor=aes_preprocessor,
                imgs=scale_img.flip(dims=(3, )), # [tot_B, h, w, 3]
                scores_record=scores_record,
                cur_idx=si,
                cal_type=cal_type,
                prompt=prompt,
                reward=reward,
                _lambda=_lambda,
            ) # [tot_B]
            # all_scores_record[:, si] = accelerator.gather(score)
            all_scores_si = accelerator.gather(score)
            all_replace_rate = accelerator.gather(replace_rate)
            
            # random replace batches of summed_codes and last_stage according to replace_rate
            local_bs = bs_sis[si] // accelerator.num_processes
            
            if torch.sum(all_replace_rate) <= 0:
                # all_replace_rate = torch.ones_like(all_replace_rate)
                all_replace_rate = torch.where(all_replace_rate > 0, torch.ones_like(all_replace_rate), torch.zeros_like(all_replace_rate))
            if torch.any(torch.isnan(all_replace_rate)):
                all_replace_rate = torch.where(torch.isnan(all_replace_rate), torch.zeros_like(all_replace_rate), torch.ones_like(all_replace_rate))
            if torch.any(torch.isinf(all_replace_rate)):
                all_replace_rate = torch.where(torch.isinf(all_replace_rate), torch.ones_like(all_replace_rate), torch.zeros_like(all_replace_rate))
            if torch.sum(all_replace_rate) <= 0:
                all_replace_rate = torch.ones_like(all_replace_rate)
            
            replace_ids = torch.multinomial(all_replace_rate, num_samples=local_bs, replacement=True, generator=rng)
            # accelerator.print(f"si: {si}, replace_ids: {replace_ids}, all_summed_codes.shape: {all_summed_codes.shape}, summed_codes.shape: {summed_codes.shape}, last_stage.shape: {last_stage.shape}")
            summed_codes = all_summed_codes[replace_ids]
            last_stage = all_last_stage[replace_ids]
            # scores_record[:, si] = all_scores_si[replace_ids]
            
            # 同步kv cache
            local_kv_cache = sync_kv_cache(
                model, sum_kv_cache(model), replace_ids=replace_ids, accelerator=accelerator,
            )
            set_kv_cache(model, local_kv_cache)
            
            if local_bs != B:
                B, bs, cond_BD, cond_BD_or_gss, ca_kv = rebatch(
                    local_bs, B, bs, cond_BD, cond_BD_or_gss, ca_kv, replace_ids=replace_ids
                )
                
        elif accelerator and si == num_stages_minus_1:
            all_scores_record = accelerator.gather(scores_record) # [tot_B, len(scale_schedule)]
            all_imgs = accelerator.gather(scale_img) # [tot_B, 3, h, w]
            img_dict[si] = all_imgs
            
            score, _ = calculate_replace_rate(
                rw_model=rw_model,
                aes_preprocessor=aes_preprocessor,
                imgs=scale_img.flip(dims=(3, )), # [tot_B, 3, h, w]
                scores_record=all_scores_record,
                cur_idx=si,
                cal_type='value',
                prompt=prompt,
                reward=reward,
                _lambda=_lambda,
            ) # [tot_B]
            all_scores_record[:, si] = accelerator.gather(score)
            # accelerator.print(f"all final scores {all_scores_record[:, si]}")
            
        else:
            all_scores_record = None
            img_dict[si] = scale_img
        
        if si != num_stages_minus_1:
            last_stage = model.word_embed(model.norm0_ve(last_stage))
            last_stage = last_stage.repeat(bs//B, 1, 1)
        # accelerator.print(f"Step {si} CUDA Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, Released: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

        progress_bar.update(1)

    if inference_mode:
        for b in model.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
    else:
        assert model.num_block_chunks > 1
        for block_chunk_ in model.block_chunks:
            for module in block_chunk_.module.module:
                (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(False)

    if not ret_img:
        return ret, idx_Bl_list, []
    
    if vae_type != 0:
        img = vae.decode(summed_codes.squeeze(-3))
    else:
        img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)

    img = (img + 1) / 2
    img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
    
    return ret, idx_Bl_list, img, img_dict, all_scores_record


from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
import hpsv2

@torch.no_grad()
def calculate_replace_rate(
    rw_model,
    aes_preprocessor,
    imgs: torch.Tensor, 
    scores_record: torch.Tensor,
    cur_idx: int,
    cal_type='value',
    _lambda=10,
    prompt=None,
    reward='aes',
):
    
    img_num = imgs.shape[0]
    
    if cal_type == 'fake':
        print("Fake rate now, with same chosen rate")
        score = torch.zeros((img_num,)).cuda()
        rate = torch.FloatTensor([1/img_num for _ in range(img_num)]).cuda()
        # rate = rate.view(img_num, 1)
    else:
        if reward == 'aes':
            pixel_values = (
                aes_preprocessor(images=imgs, return_tensors="pt")
                .pixel_values.to(torch.bfloat16)
                .cuda()
            )
            with torch.inference_mode():
                score = rw_model(pixel_values).logits.squeeze().float() # [img_num]
                score = score / 10.0
                # print(f"Aesthetics score: {score} at {cur_idx}")
        elif reward == 'hps':
            assert prompt is not None
            
            imgs = [Image.fromarray(imgs[i].byte().cpu().numpy()) for i in range(img_num)]
            with torch.inference_mode():
                score = hpsv2.score(imgs, prompt, hps_version="v2.1")
                score = torch.FloatTensor(score).cuda()
                # print(f"HPS score: {score} at {cur_idx}")
        elif reward == 'ir':
            assert prompt is not None
            
            imgs = [Image.fromarray(imgs[i].byte().cpu().numpy()) for i in range(img_num)]
            with torch.inference_mode():
                score = rw_model.score(prompt, imgs)
                if isinstance(score, float):
                    score = torch.FloatTensor([score]).cuda()
                else:
                    score = torch.FloatTensor(score).cuda()
            
        if cal_type == 'value' or cur_idx == 0:
            rate = torch.exp(_lambda * score)
        elif cal_type == 'diff':
            rate = torch.exp(_lambda * (score - scores_record[:, cur_idx-1].cuda()))
        elif cal_type == 'sum':
            rate = torch.exp(_lambda * (score + torch.sum(scores_record[:, :cur_idx].cuda(), dim=1)))
        elif cal_type == 'max':
            rate = torch.exp(_lambda * (torch.max(
                torch.cat((score.unsqueeze(1), scores_record[:, :cur_idx].cuda()), dim=1), dim=1)[0]
            ))
        elif "topk" in cal_type:
            k_num = int(cal_type.split("_")[-1])

            # 合并当前 score 与历史记录
            # merged = torch.cat((score.unsqueeze(1), scores_record[:, :cur_idx].cuda()), dim=1)
            merged = torch.cat((scores_record[:, :cur_idx].cuda(), score.unsqueeze(1)), dim=1)
            # 获取 topk 值（自动处理不足 k 个的情况）
            if k_num > merged.shape[1]:
                k_num = merged.shape[1]

            topk_values = torch.topk(merged, k_num, dim=1, largest=True, sorted=False).values
            # 聚合方式：sum
            topk_sum = topk_values.sum(dim=1)
            rate = torch.exp(_lambda * topk_sum)
            
    return score, rate

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def extract_k_ids(
    feature_model,
    imgs,
    source_num,
    target_num,
    patch_size=16,
    extract_type="pca",
):
    img_num = imgs.shape[0]
    assert source_num == img_num and target_num < len(imgs)
    
    imgs = [Image.fromarray(imgs[i].byte().cpu().numpy()) for i in range(img_num)]
    transform = transforms.Compose([
        transforms.Resize(
            (patch_size * 14, patch_size * 14) if extract_type != "inception" else (299, 299)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    imgs_tensor = torch.stack([transform(img) for img in imgs]).cuda() # [B, C, H, W]

    
    if extract_type == "pca":
        with torch.inference_mode() and torch.no_grad():
            features_dict = feature_model.forward_features(imgs_tensor)
            features = features_dict['x_norm_patchtokens']  # [B, L, D]
        # PCA处理：每个样本独立进行PCA投影
        B, L, D = features.shape
        scores = []
        for i in range(B):
            x_i = features[i]  # [L, D]
            x_centered = x_i - x_i.mean(dim=0, keepdim=True)
            U, S, V = torch.svd(x_centered.float())  # 计算SVD
            principal = V[:, 0]  # 第一主成分
            score_i = (x_centered @ principal)  # [L]
            scores.append(score_i)
        scores = torch.stack(scores, dim=0)  # [B, L]
    elif extract_type == "pca_all":
        with torch.inference_mode() and torch.no_grad():
            features_dict = feature_model.forward_features(imgs_tensor)
            features = features_dict['x_norm_patchtokens']  # [B, L, D]
        # 全局PCA处理：所有样本合并后进行PCA投影
        B, L, D = features.shape
        flattened = features.reshape(-1, D)  # 合并所有样本 [B*L, D]
        global_mean = flattened.mean(dim=0, keepdim=True)  # 全局均值 [1, D]
        centered = flattened - global_mean  # 全局中心化 [B*L, D]
        
        # 计算全局主成分
        U, S, V = torch.svd(centered.float())  # V形状为[D, D]
        principal = V[:, 0]  # 第一主成分 [D]
        
        # 投影并还原分组
        scores_flat = centered @ principal  # 全局投影 [B*L]
        scores = scores_flat.reshape(B, L)  # 还原为[B, L]
    elif extract_type == "pool":
        with torch.inference_mode() and torch.no_grad():
            features_dict = feature_model.forward_features(imgs_tensor)
            features = features_dict['x_norm_patchtokens']  # [B, L, D]
        # 平均池化处理通道维度
        scores = features.mean(dim=1)  # [B, D]
    elif extract_type == "inception":
        with torch.inference_mode() and torch.no_grad():
            scores = feature_model(imgs_tensor) # [B, D]
    else:
        raise ValueError(f"Unsupported extract_type: {extract_type}")

    # 转换scores为numpy数组
    scores_np = scores.float().cpu().numpy()

    # K-means聚类
    kmeans = KMeans(n_clusters=target_num, random_state=0)
    kmeans.fit(scores_np)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    selected_ids = []
    for cluster_id in range(target_num):
        # 获取当前簇的所有样本索引
        cluster_indices = np.where(labels == cluster_id)[0]
        # 获取这些样本的特征
        cluster_points = scores_np[cluster_indices]
        # 计算每个样本到簇中心的距离
        distances = np.linalg.norm(cluster_points - cluster_centers[cluster_id], axis=1)
        # 找到最近的样本索引
        closest_idx = np.argmin(distances)
        # 添加原始索引
        selected_id = cluster_indices[closest_idx]
        selected_ids.append(selected_id)

    # 转换为tensor，[B']
    selected_ids = torch.tensor(selected_ids, dtype=torch.long)
    return selected_ids

def rebatch(
    local_bs, 
    B, 
    bs, 
    cond_BD,
    cond_BD_or_gss, 
    ca_kv, 
    replace_ids,
):
    assert local_bs == replace_ids.shape[0]
    
    new_B = local_bs
    new_bs = local_bs * (bs//B)
    
    
    new_cond_BD = cond_BD[
        torch.cat([
            torch.Tensor([0] * local_bs).long(), torch.Tensor([B] * local_bs).long()
        ])
    ]
    new_cond_BD_or_gss = cond_BD_or_gss[
        torch.cat([
            torch.Tensor([0] * local_bs).long(), torch.Tensor([B] * local_bs).long()
        ])
    ]
    
    ca_kv_0 = ca_kv[0].reshape(bs, -1, ca_kv[0].shape[-1])
    new_ca_kv_0 = ca_kv_0[
        torch.cat([
            torch.Tensor([0] * local_bs).long(), torch.Tensor([B] * local_bs).long()
        ])
    ].reshape(-1, ca_kv_0.shape[-1])
    new_ca_kv = new_ca_kv_0, ca_kv[1][:new_bs+1], ca_kv[2]
    
    return new_B, new_bs, new_cond_BD, new_cond_BD_or_gss, new_ca_kv

def set_kv_cache(model, kv_cache_dict):
    for ib, b in enumerate(model.unregistered_blocks):
        if isinstance(b, CrossAttnBlock):
            del b.sa.cached_k, b.sa.cached_v
            
            b.sa.cached_k = kv_cache_dict["cached_k"][ib].clone()
            b.sa.cached_v = kv_cache_dict["cached_v"][ib].clone()
        else:
            del b.attn.cached_k, b.attn.cached_v
            
            b.attn.cached_k = kv_cache_dict["cached_k"][ib].clone()
            b.attn.cached_v = kv_cache_dict["cached_v"][ib].clone()
    
    gc.collect()
    torch.cuda.empty_cache()
    # return kv_cache_dict
                
def sum_kv_cache(model):
    kv_cache_dict = {
        "cached_k": [None for _ in range(len(model.unregistered_blocks))],
        "cached_v": [None for _ in range(len(model.unregistered_blocks))],
    }
    
    for ib, b in enumerate(model.unregistered_blocks):
        if isinstance(b, CrossAttnBlock):
            kv_cache_dict['cached_k'][ib] = b.sa.cached_k.clone()
            kv_cache_dict['cached_v'][ib] = b.sa.cached_v.clone()
        else:
            kv_cache_dict['cached_k'][ib] = b.attn.cached_k.clone()
            kv_cache_dict['cached_v'][ib] = b.attn.cached_v.clone()
            
    return kv_cache_dict
            
def sync_kv_cache(model, kv_cache_dict, replace_ids, accelerator):
    assert len(kv_cache_dict['cached_k']) == len(kv_cache_dict['cached_v'])
    all_replace_ids = None
    # b_repeat = None
    
    for ib, b in enumerate(model.unregistered_blocks):
        all_kv_cache_k = accelerator.gather(kv_cache_dict['cached_k'][ib])
        all_kv_cache_v = accelerator.gather(kv_cache_dict['cached_v'][ib])
        # print(f'before sync: {kv_cache_dict["cached_k"][ib].shape}, {kv_cache_dict["cached_v"][ib].shape}')
        # print(f'all kv cache: {all_kv_cache_k.shape}, {all_kv_cache_v.shape}')
        
        if all_replace_ids is None:
            local_length = kv_cache_dict['cached_k'][ib].shape[0]
            all_length = all_kv_cache_k.shape[0]
            B = local_length // 2
            cond_ids = (replace_ids // B) * local_length + torch.remainder(replace_ids, B)
            uncond_ids = (replace_ids // B) * local_length + B + torch.remainder(replace_ids, B)
            all_replace_ids = torch.cat((cond_ids, uncond_ids))
            # accelerator.print(replace_ids, all_replace_ids)
            # b_repeat = local_length//all_replace_ids.shape[0]

        # accelerator.print(f'all replace ids: {all_replace_ids}, {local_length//all_replace_ids.shape[0]}')
        # accelerator.print(all_kv_cache_k.shape, all_kv_cache_v.shape)
        
        kv_cache_dict['cached_k'][ib] = all_kv_cache_k[all_replace_ids]
        kv_cache_dict['cached_v'][ib] = all_kv_cache_v[all_replace_ids]
        # kv_cache_dict['cached_k'][ib] = all_kv_cache_k[all_replace_ids].repeat(b_repeat, 1, 1, 1)
        # kv_cache_dict['cached_v'][ib] = all_kv_cache_v[all_replace_ids].repeat(b_repeat, 1, 1, 1)
        # print(f'after sync: {kv_cache_dict["cached_k"][ib].shape}, {kv_cache_dict["cached_v"][ib].shape}')
        
    return kv_cache_dict
        
def mv_cache(model, kv_cache_dict, device):
    for ib, b in enumerate(model.unregistered_blocks):
        kv_cache_dict["cached_k"][ib] = kv_cache_dict["cached_k"][ib].to(device)
        kv_cache_dict["cached_v"][ib] = kv_cache_dict["cached_v"][ib].to(device)
    return kv_cache_dict