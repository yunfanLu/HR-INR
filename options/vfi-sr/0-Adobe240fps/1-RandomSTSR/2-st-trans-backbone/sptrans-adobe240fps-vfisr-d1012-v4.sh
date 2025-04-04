export CUDA_VISIBLE_DEVICES="5"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/1-RandomSTSR/2-st-trans-backbone/sptrans-adobe240fps-vfisr-d1012-v4.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/1-RandomSTSR/2-st-trans-backbone/sptrans-adobe240fps-vfisr-d1012-v4/" \
  --alsologtostderr=True
