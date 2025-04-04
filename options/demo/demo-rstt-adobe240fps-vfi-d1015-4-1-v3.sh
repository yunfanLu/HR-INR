export CUDA_VISIBLE_DEVICES="0,7"
export PYTHONPATH="./3rdparty/":$PYTHONPATH
export PYTHONPATH="./":$PYTHONPATH

python egrsdb/main.py \
  --yaml_file="options/demo/demo-rstt-adobe240fps-vfi-d1015-4-1-v3.yaml" \
  --log_dir="./log/demo/demo-rstt-adobe240fps-vfi-d1015-4-1-v3/" \
  --alsologtostderr=True
