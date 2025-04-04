export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/1-RandomSTSR/sptrans-adobe240fps-vfisr-v1012.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/1-RandomSTSR/sptrans-adobe240fps-vfisr-v1012/" \
  --alsologtostderr=True
