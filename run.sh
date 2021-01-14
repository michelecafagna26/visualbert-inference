export PYTHONPATH=$PYTHONPATH:visualbert/

CUDA_VISIBLE_DEVICES=0 python inference.py -folder logs  -config visualbert/configs/coco-inference.json
