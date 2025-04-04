export CUDA_VISIBLE_DEVICES="7"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/3-VFI-Only/rstt-ADB-vfi-d1020-4-7-wo-pretrained-small-v1.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/3-VFI-Only/rstt-ADB-vfi-d1020-4-7-wo-pretrained-small-v1/" \
  --alsologtostderr=True

