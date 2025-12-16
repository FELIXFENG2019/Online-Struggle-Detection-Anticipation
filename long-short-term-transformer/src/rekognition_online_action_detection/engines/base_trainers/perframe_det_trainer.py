# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from tqdm import tqdm

import torch
import torch.nn as nn

from rekognition_online_action_detection.evaluation import compute_result


def do_perframe_det_train(cfg,
                          data_loaders,
                          model,
                          criterion,
                          optimizer,
                          scheduler,
                          device,
                          checkpointer,
                          logger):
    # Setup model on multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    import wandb
    if cfg.SOLVER.ENABLE_WANDB:
        wandb.login()
        run = wandb.init(
            project="lstr-online-struggle-detection",
            # Track hyperparameters and run metadata
            config=cfg
        )
    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        # Reset
        det_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        det_pred_scores = []
        det_gt_targets = []

        start = time.time()
        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            model.train(training)

            with torch.set_grad_enabled(training):
                pbar = tqdm(data_loaders[phase],
                            desc='{}ing epoch {}'.format(phase.capitalize(), epoch))
                for batch_idx, data in enumerate(pbar, start=1):
                    # import pdb; pdb.set_trace()
                    batch_size = data[0].shape[0]
                    det_target = data[-1].to(device)

                    det_score = model(*[x.to(device) for x in data[:-1]])
                    det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                    det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                    # import pdb; pdb.set_trace()
                    det_loss = criterion['Focal'](det_score, det_target)
                    det_losses[phase] += det_loss.item() * batch_size

                    # Output log for current batch
                    pbar.set_postfix({
                        'lr': '{:.7f}'.format(scheduler.get_last_lr()[0]),
                        'det_loss': '{:.5f}'.format(det_loss.item()),
                    })

                    if training:
                        optimizer.zero_grad()
                        det_loss.backward()
                        optimizer.step()
                        scheduler.step()
                    else:
                        # for anticipation
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            det_score = det_score.reshape(batch_size, -1, cfg.DATA.NUM_CLASSES)
                            det_score = det_score[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                            det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)

                            det_target = det_target.reshape(batch_size, -1, cfg.DATA.NUM_CLASSES)
                            det_target = det_target[:, -cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:, :]
                            det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
            
                        # Prepare for evaluation (for normal online action detection setting)
                        det_score = det_score.softmax(dim=1).cpu().tolist()
                        det_target = det_target.cpu().tolist()
                        det_pred_scores.extend(det_score)
                        det_gt_targets.extend(det_target)
        end = time.time()

        # Output log for current epoch
        log = []
        log.append('Epoch {:2}'.format(epoch))
        log.append('train det_loss: {:.5f}'.format(
            det_losses['train'] / len(data_loaders['train'].dataset),
        ))
        if 'test' in cfg.SOLVER.PHASES:
            # Compute result
            det_result = compute_result['perframe'](
                cfg,
                det_gt_targets,
                det_pred_scores,
            )
            log.append('test det_loss: {:.5f} det_mAP: {:.5f}'.format(
                det_losses['test'] / len(data_loaders['test'].dataset),
                det_result['mean_AP'],
            ))
        log.append('running time: {:.2f} sec'.format(
            end - start,
        ))
        logger.info(' | '.join(log))

        if cfg.SOLVER.ENABLE_WANDB:
            run.log({
                'train_det_loss': det_losses['train'] / len(data_loaders['train'].dataset),
                'test_det_loss': det_losses['test'] / len(data_loaders['test'].dataset),
                'test_det_mAP': det_result['mean_AP'],
                'lr': scheduler.get_last_lr()[0], 
                'epoch': epoch,
            })

        # Save checkpoint for model and optimizer
        checkpointer.save(epoch, model, optimizer)

        # Shuffle dataset for next epoch
        data_loaders['train'].dataset.shuffle()
    
    if cfg.SOLVER.ENABLE_WANDB:
        run.finish()

