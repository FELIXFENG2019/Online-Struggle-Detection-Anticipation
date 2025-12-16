# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
from tqdm import tqdm

import torch
import numpy as np
import pickle as pkl

from rekognition_online_action_detection.datasets import build_dataset
from rekognition_online_action_detection.evaluation import compute_result


def do_perframe_det_batch_inference(cfg, model, device, logger):
    # Setup model to test mode
    model.eval()

    data_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg, phase='test', tag='BatchInference'),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE ,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    # Collect scores and targets
    pred_scores = {}
    gt_targets = {}

    # ADDED
    # import time
    # counter_iter = 0
    # all_iteration_runtimes_ms = [] # To store the runtime for each measured iteration
    # if device == 'cuda':
    #     starter = torch.cuda.Event(enable_timing=True)
    #     ender = torch.cuda.Event(enable_timing=True)
    # start_runtime_measure = False # Flag to start measuring runtime after 5 iterations
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='BatchInference')
        for batch_idx, data in enumerate(pbar, start=1):
            # import pdb; pdb.set_trace()
            # if counter_iter >= 100:
            #     start_runtime_measure = True
            
            # if start_runtime_measure:
            #     # ADDED
            #     # --- Start Timer for this iteration ---
            #     if device == 'cuda':
            #         starter.record()
            #     else: # CPU timing
            #         start_time_cpu = time.time()

            target = data[-4]
            score = model(*[x.to(device) for x in data[:-4]])
            score = score.softmax(dim=-1).cpu().numpy()
            
            # if start_runtime_measure:
            # # --- End Timer for this iteration ---
            #     if device == 'cuda':
            #         ender.record()
            #         torch.cuda.synchronize() # THIS IS CRUCIAL: Wait for all GPU operations to finish
            #         iteration_runtime_ms = starter.elapsed_time(ender)
            #     else: # CPU timing
            #         end_time_cpu = time.time()
            #         iteration_runtime_ms = (end_time_cpu - start_time_cpu) * 1000 # Convert to milliseconds

            #     all_iteration_runtimes_ms.append(iteration_runtime_ms)
            
            # counter_iter += 1
            # if counter_iter >= 1000:
            #     mean_total_runtime_ms = sum(all_iteration_runtimes_ms) / len(all_iteration_runtimes_ms)
            #     std_total_runtime_ms = (sum([(x - mean_total_runtime_ms) ** 2 for x in all_iteration_runtimes_ms]) / len(all_iteration_runtimes_ms)) ** 0.5
            #     print(f'Average runtime per iteration: {mean_total_runtime_ms:.2f} ms, Std Dev: {std_total_runtime_ms:.2f} ms')
            #     import pdb; pdb.set_trace()

            """
            # for random baseline 
            score = []
            # target reshape (b, 25, 2) -> (b*25, 2)
            target = target.reshape(-1, cfg.DATA.NUM_CLASSES)
            # get probability from target
            background, struggle = target.sum(axis=0).tolist()
            struggle_prob = struggle / (background + struggle)
            # struggle_prob = 0.5 # for purely random baseline

            for i in range(target.shape[0]):
                per_frame_score = np.random.random_sample()
                if per_frame_score < struggle_prob:
                    per_frame_score = 1
                score.append([1 - per_frame_score, per_frame_score])
            score = np.array(score) # shape (b*25, 2)
            # print(score)
            # (b*25, 2) -> (b, 25, 2)
            score = score.reshape(-1, 49, 2)
            # print(score.shape) # shape (b, 25, 2)
            """

            for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                if session not in pred_scores:
                    pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                if session not in gt_targets:
                    gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                # import pdb; pdb.set_trace()
                if query_indices[0] == 0:
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        pred_scores[session][query_indices[-cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:]] = score[bs][-cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:]
                        gt_targets[session][query_indices[-cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:]] = target[bs][-cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:]
                    else:
                        pred_scores[session][query_indices] = score[bs]
                        gt_targets[session][query_indices] = target[bs]
                else:
                    pred_scores[session][query_indices[-1]] = score[bs][-1]
                    gt_targets[session][query_indices[-1]] = target[bs][-1]

    # Save scores and targets
    pkl.dump({
        'cfg': cfg,
        'perframe_pred_scores': pred_scores,
        'perframe_gt_targets': gt_targets,
    }, open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '-anticipation'+ '.pkl', 'wb'))
    # import pdb; pdb.set_trace()
    # Compute results
    result = compute_result['perframe'](
        cfg,
        np.concatenate(list(gt_targets.values()), axis=0),
        np.concatenate(list(pred_scores.values()), axis=0),
    )
    logger.info('Action detection perframe m{}: {:.5f}'.format(
        cfg.DATA.METRICS, result['mean_AP']
    ))
