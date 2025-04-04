export CUDA_VISIBLE_DEVICES="7"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/vfi-sr/0-Adobe240fps/3-VFI-Only/sptrans-adobe240fps-vfi-d1012-v6.yaml" \
  --log_dir="./log/vfi-sr/0-Adobe240fps/3-VFI-Only/sptrans-adobe240fps-vfi-d1012-v6/" \
  --alsologtostderr=True

