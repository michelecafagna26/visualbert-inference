export PYTHONPATH=$PYTHONPATH:visualbert/
#export PYTHONPATH=$PYTHONPATH:XX/

#CUDA_VISIBLE_DEVICES=0 python XX/train.py -folder XX/logs  -config XX/visualbert/configs/vqa/fine-tune.json

CUDA_VISIBLE_DEVICES=0 python inference.py -folder logs  -config visualbert/configs/vqa/coco-pre-train.json
