export CUDA_VISIBLE_DEVICES="6"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/1-RandomSTSR/3-rstt_event_adapter/rstt-adobe240fps-vfisr-d1014-v1.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/1-RandomSTSR/3-rstt_event_adapter/rstt-adobe240fps-vfisr-d1014-v1/" \
  --alsologtostderr=True
