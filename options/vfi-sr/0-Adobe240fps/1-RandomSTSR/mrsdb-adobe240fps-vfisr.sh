export CUDA_VISIBLE_DEVICES="2"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/1-RandomSTSR/mrsdb-adobe240fps-vfisr.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/1-RandomSTSR/mrsdb-adobe240fps-vfisr/" \
  --alsologtostderr=True
