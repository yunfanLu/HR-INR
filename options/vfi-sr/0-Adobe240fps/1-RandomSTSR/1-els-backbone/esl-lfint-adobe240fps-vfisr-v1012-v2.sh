export CUDA_VISIBLE_DEVICES="1"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/1-RandomSTSR/1-els-backbone/esl-lfint-adobe240fps-vfisr-d1012-v2.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/1-RandomSTSR/1-els-backbone/esl-lfint-adobe240fps-vfisr-d1012-v2/" \
  --alsologtostderr=True
