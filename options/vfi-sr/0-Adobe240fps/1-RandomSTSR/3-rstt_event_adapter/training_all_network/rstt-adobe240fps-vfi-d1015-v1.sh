export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/1-RandomSTSR/3-rstt_event_adapter/training_all_network/rstt-adobe240fps-vfi-d1015-v1.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/1-RandomSTSR/3-rstt_event_adapter/training_all_network/rstt-adobe240fps-vfi-d1015-v1/" \
  --alsologtostderr=True
