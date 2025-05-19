python models/timesfm_torch/convert_torch_ckpt.py
echo "TimeSFM ckpt downloaded"

mkdir -p models/units/ckpt/

wget -O models/units/ckpt/units_x128_pretrain_checkpoint.pth https://github.com/mims-harvard/UniTS/releases/download/ckpt/units_x128_pretrain_checkpoint.pth
echo "Units x128 ckpt downloaded"

wget -O models/units/ckpt/units_x32_pretrain_checkpoint.pth https://github.com/mims-harvard/UniTS/releases/download/ckpt/units_x32_pretrain_checkpoint.pth
echo "Units x32 ckpt downloaded"