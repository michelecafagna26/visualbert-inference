export PYTHONPATH=$PYTHONPATH:XX/visualbert/
#export PYTHONPATH=$PYTHONPATH:XX/

#CUDA_VISIBLE_DEVICES=0 python XX/train.py -folder XX/logs  -config XX/visualbert/configs/vqa/fine-tune.json

CUDA_VISIBLE_DEVICES=0 python XX/inference.py -folder XX/logs  -config XX/visualbert/configs/vqa/coco-pre-train.json
