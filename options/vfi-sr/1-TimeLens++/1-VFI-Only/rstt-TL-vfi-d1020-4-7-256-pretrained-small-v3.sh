export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/1-TimeLens++/1-VFI-Only/rstt-TL-vfi-d1020-4-7-256-pretrained-small-v3.yaml" \
  --log_dir="./log/vfi-sr/1-TimeLens++/1-VFI-Only/rstt-TL-vfi-d1020-4-7-256-pretrained-small-v3/" \
  --alsologtostderr=True

