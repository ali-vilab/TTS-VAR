mkdir -p pretrained_models
git clone https://huggingface.co/google/flan-t5-xl pretrained_models/flan-t5-xl
git clone https://huggingface.co/google/siglip-so400m-patch14-384 pretrained_models/siglip-so400m-patch14-384
wget https://github.com/discus0434/aesthetic-predictor-v2-5/raw/refs/heads/main/models/aesthetic_predictor_v2_5.pth -O pretrained_models/aesthetic_predictor_v2_5.pth
wget https://huggingface.co/THUDM/ImageReward/resolve/main/ImageReward.pt -O pretrained_models/ImageReward.pt
wget https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_2b_reg.pth -O pretrained_models/infinity_2b_reg.pth
wget https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_vae_d32reg.pth -O pretrained_models/infinity_vae_d32reg.pth