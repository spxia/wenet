# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.init_model import init_model

import numpy as np
from ctc_segmentation import (
    CtcSegmentationParameters,
    ctc_segmentation,
    determine_utterance_segments,
    prepare_token_list
)

def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='asr result file')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--revise_time',
                        type=float,
                        default=0.2, # v1.12 滞后了0.2s,这个值需要根据模型去调整
                        help='根据模型调整修正的时间差')


    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)


    # Load dict
    symbol_table = read_symbol_table(args.dict)
    keys_list = list(symbol_table.keys())
    char_dict = {v: k for k, v in symbol_table.items()}

    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['perturb_wav'] = False
    test_conf['save_audio'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           non_lang_syms,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    model = init_model(configs)
    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    config = CtcSegmentationParameters()
    # Parameters for timing
    config.subsampling_factor = 4
    config.frame_duration_ms = 10
    # Parameters for text preparation
    config.char_list = keys_list[:-1]

    model.eval()
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            encoder_out, _ = model._forward_encoder(
            feats, feats_lengths, args.decoding_chunk_size, args.num_decoding_left_chunks, args.simulate_streaming)
            lpz = model.ctc.log_softmax(encoder_out)
            lpz = lpz[0].cpu().detach()

            # prepare label
            text_char = []
            for i in target[0]:
                text_char.append(char_dict[i.item()])
            tokens_array = []
            for text in text_char:
                tokens_array.append(np.array([symbol_table[c] for c in text]))

            ground_truth_mat, utt_begin_indices = prepare_token_list(config, tokens_array)
            timings, char_probs, state_list = ctc_segmentation(
                config, lpz.numpy(), ground_truth_mat
            )

            # Obtain list of utterances with time intervals and confidence score
            segments = determine_utterance_segments(
                config, utt_begin_indices, char_probs, timings, text_char
            )
            
            revise_time = args.revise_time
            for i, boundary in enumerate(segments):
                start_time = boundary[0] - revise_time
                if start_time < 0.0:
                    start_time = 0.0
                if (i + 1)  == len(segments):
                    end_time = boundary[1]
                else:
                    end_time = boundary[1] - revise_time
                utt_segment = (
                    f" {text_char[i]} {start_time:.2f}"
                    f" {end_time:.2f} {boundary[2]:.9f}"
                )
                logging.info('{}{}'.format(keys[0], utt_segment))
                fout.write('{}{}\n'.format(keys[0], utt_segment))


if __name__ == '__main__':
    main()
