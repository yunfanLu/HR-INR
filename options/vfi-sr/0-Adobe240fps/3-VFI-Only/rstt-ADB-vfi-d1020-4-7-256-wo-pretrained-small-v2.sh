export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/3-VFI-Only/rstt-ADB-vfi-d1020-4-7-256-wo-pretrained-small-v2.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/3-VFI-Only/rstt-ADB-vfi-d1020-4-7-256-wo-pretrained-small-v2/" \
  --alsologtostderr=True

