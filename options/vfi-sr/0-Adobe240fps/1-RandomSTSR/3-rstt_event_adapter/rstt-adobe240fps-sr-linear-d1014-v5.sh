export CUDA_VISIBLE_DEVICES="5"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/1-RandomSTSR/3-rstt_event_adapter/rstt-adobe240fps-sr-linear-d1014-v5.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/1-RandomSTSR/3-rstt_event_adapter/rstt-adobe240fps-sr-linear-d1014-v5/" \
  --alsologtostderr=True
