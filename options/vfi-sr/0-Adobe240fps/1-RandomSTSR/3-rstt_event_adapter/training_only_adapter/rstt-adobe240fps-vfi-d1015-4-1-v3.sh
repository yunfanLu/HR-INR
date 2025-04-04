export CUDA_VISIBLE_DEVICES="4"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/1-RandomSTSR/3-rstt_event_adapter/training_only_adapter/rstt-adobe240fps-vfi-d1015-4-1-v3.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/1-RandomSTSR/3-rstt_event_adapter/training_only_adapter/rstt-adobe240fps-vfi-d1015-4-1-v3/" \
  --alsologtostderr=True
