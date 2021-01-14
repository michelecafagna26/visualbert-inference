"""
Training script. Should be pretty adaptable to whatever.
"""
import argparse
import os

import shutil
from copy import deepcopy

import multiprocessing
import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm

from allennlp.nn.util import device_mapping

from visualbert.utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint, restore_checkpoint_flexible, load_state_dict_flexible, compute_score_with_logits

from visualbert.dataloaders.vcr import VCR, VCRLoader
from pytorch_pretrained_bert.optimization import BertAdam

try:
    from visualbert.dataloaders.coco_dataset import COCODataset
except:
    print("Import COCO dataset failed.")

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

from allennlp.models import Model
from visualbert.models.model_wrapper import ModelWrapper
from visualbert.models import model

#################################
from attrdict import AttrDict

parser = argparse.ArgumentParser(description='train')

parser.add_argument(
    '-folder',
    dest='folder',
    help='folder location',
    type=str,
)
parser.add_argument(
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)

parser.add_argument(
    '-config',
    dest='config',
    help='config location',
    type=str,
)

args = parser.parse_args()

args = ModelWrapper.read_and_insert_args(args, args.config)

##################################################### 

if os.path.exists(args.folder):
    create_flag = 0
else:
    create_flag = 1
    print("Making directories")
    os.makedirs(args.folder, exist_ok=True)

import sys
run_log_counter = 0

while(os.path.exists(args.folder + '/run_{}.log'.format(run_log_counter))):
    run_log_counter += 1

file_log = open(args.folder + '/run_{}.log'.format(run_log_counter),'w')  # File where you need to keep the logs
file_log.write("")
class Unbuffered:
    def __init__(self, stream):
       self.stream = stream
    def write(self, data):
       self.stream.write(data)
       self.stream.flush()
       file_log.write(data)    # Write the data of stdout here to a text file as well
    def flush(self):
        pass

sys.stdout = Unbuffered(sys.stdout)

NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = multiprocessing.cpu_count()
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")

def _to_gpu(td):
    if args.get("fp16", False):
        _to_fp16(td)
nvidi
    if NUM_GPUS > 1:
        return td
    for k in td:
        if k != 'metadata':
            if td[k] is not None:
                td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(non_blocking=True)
    return td
def _to_fp16(td):
    for k in td:
        if isinstance(td[k], torch.FloatTensor):
            td[k] = td[k].to(dtype=torch.float16)

num_workers = args.get("num_workers", 2)
val_workers = args.get("val_workers", 0)

TEST_DATA_READING = False
if TEST_DATA_READING:
    num_workers = 0

print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': args.train_batch_size // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}

def get_dataset_loader(args, dataset_name):
    # The VCR approach toward

    if dataset_name == "coco":
        train, val, test = COCODataset.splits(args)
    else:
        assert(0)

    loader_params = {'batch_size': args.train_batch_size // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}
    train_loader_params = deepcopy(loader_params)
    val_loader_params = deepcopy(loader_params)
    val_loader_params["num_workers"] = val_workers
    test_loader_params = deepcopy(loader_params)
    test_loader_params["num_workers"] = val_workers
    
    train_loader = VCRLoader.from_dataset(train, **train_loader_params)
    val_loader = VCRLoader.from_dataset(val, **val_loader_params)
    test_loader = VCRLoader.from_dataset(test, **test_loader_params)
    train_set_size = len(train)

    return train_loader, val_loader, test_loader, train_set_size


train_loader, val_loader, test_loader, train_set_size = get_dataset_loader(args, args.dataset)


train_model = ModelWrapper(args, train_set_size)

#Loading from pre-trained model
if args.restore_bin:
    train_model.restore_checkpoint_pretrained(args.restore_bin)

#Loading from previous checkpoint
if create_flag == 0:
    start_epoch, val_metric_per_epoch = train_model.restore_checkpoint(serialization_dir=args.folder, epoch_to_load = args.get("epoch_to_load", None))
    if val_metric_per_epoch is None:
        val_metric_per_epoch = []
else:
    create_flag = 1
    start_epoch, val_metric_per_epoch = 0, []

shutil.copy2(args.config, args.folder) # Always copy the config

if args.get("freeze_detector", True):
    train_model.freeze_detector()

param_shapes = print_para(train_model.model)

print("########### Starting from {}".format(start_epoch))

try:
    ### This is the inference part

    train_model.eval()

    for i, batch in enumerate(val_loader):
        with torch.no_grad():

            tokens = [val_loader.dataset.tokenizer.ids_to_tokens[i] for i in batch['bert_input_ids'][0].tolist()]
            batch = _to_gpu(batch)
            output_dict = train_model.step(batch, eval_mode = True)
            print(output_dict.keys())

            index = val_loader.dataset.items[i]['image_id']
            outfile_name = "COCO_val2014_000000{}.out.npz".format(index)

            output_dict['attention_weights'] = [ m.cpu().detach().numpy() for m in output_dict['attention_weights']]

            np.savez("visualbert-inference/out_attn/" + outfile_name, tokens = tokens, attention_weights = output_dict['attention_weights'] )

except KeyboardInterrupt:

    print("Something Went Wrong with Evaluation. Stopped.")
    assert(0)
except Exception as e:
    print(e)
    print("Something Went Wrong with Evaluation. Ignored.")
    if args.get("skip_training", False):
        assert(0)
        