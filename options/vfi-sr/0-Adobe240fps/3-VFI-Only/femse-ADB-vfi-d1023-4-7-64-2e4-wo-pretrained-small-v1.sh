export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/3-VFI-Only/femse-ADB-vfi-d1023-4-7-64-2e4-wo-pretrained-small-v1.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/3-VFI-Only/femse-ADB-vfi-d1023-4-7-64-2e4-wo-pretrained-small-v1/" \
  --alsologtostderr=True

