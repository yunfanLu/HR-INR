export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/1-TimeLens++/1-VFI-Only/rstt-TL-vfi-d1021-4-7-320-pretrained-small-v1.yaml" \
  --log_dir="./log/vfi-sr/1-TimeLens++/1-VFI-Only/rstt-TL-vfi-d1021-4-7-320-pretrained-small-v1/" \
  --alsologtostderr=True

