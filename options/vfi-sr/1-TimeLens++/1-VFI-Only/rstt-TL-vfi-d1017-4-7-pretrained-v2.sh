export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/1-TimeLens++/1-VFI-Only/rstt-TL-vfi-d1017-4-7-pretrained-v2.yaml" \
  --log_dir="./log/vfi-sr/1-TimeLens++/1-VFI-Only/rstt-TL-vfi-d1017-4-7-pretrained-v2/" \
  --alsologtostderr=True \
  --RESUME_PATH="./log/vfi-sr/1-TimeLens++/1-VFI-Only/rstt-TL-vfi-d1017-4-7-pretrained-v2/checkpoint.pth.tar" \
  --RESUME_SET_EPOCH=True
