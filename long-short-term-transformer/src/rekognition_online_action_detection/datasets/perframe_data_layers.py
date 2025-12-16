# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import os.path as osp
from bisect import bisect_right

import torch
import torch.utils.data as data
import numpy as np

from .datasets import DATA_LAYERS as registry


@registry.register('LSTRTHUMOS')
@registry.register('LSTRTVSeries')
class LSTRDataLayer(data.Dataset):

    def __init__(self, cfg, phase='train'):
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.training = phase == 'train'

        self._init_dataset()

    def shuffle(self):
        self._init_dataset()

    def _init_dataset(self):
        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
            seed = np.random.randint(self.work_memory_length) if self.training else 0
            for work_start, work_end in zip(
                range(seed, target.shape[0], self.work_memory_length),
                range(seed + self.work_memory_length, target.shape[0], self.work_memory_length)):
                self.inputs.append([
                    session, work_start, work_end, target[work_start: work_end],
                ])

    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, target = self.inputs[index]

        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')

        # Get target
        target = target[::self.work_memory_sample_rate]

        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            if self.training:
                long_indices = self.segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = self.uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1 # Finding the last zero index
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target
        else:
            return fusion_visual_inputs, fusion_motion_inputs, target

    def __len__(self):
        return len(self.inputs)


@registry.register('LSTRBatchInferenceTHUMOS')
@registry.register('LSTRBatchInferenceTVSeries')
class LSTRBatchInferenceDataLayer(data.Dataset):

    def __init__(self, cfg, phase='test'):
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES

        assert phase == 'test', 'phase must be `test` for batch inference, got {}'

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
            for work_start, work_end in zip(
                range(0, target.shape[0] + 1),
                range(self.work_memory_length, target.shape[0] + 1)):
                self.inputs.append([
                    session, work_start, work_end, target[work_start: work_end], target.shape[0]
                ])

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, target, num_frames = self.inputs[index]

        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')

        # Get target
        target = target[::self.work_memory_sample_rate]

        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            long_indices = self.uniform_sampler(
                long_start,
                long_end,
                self.long_memory_num_samples,
                self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target,
                    session, work_indices, num_frames)
        else:
            return (fusion_visual_inputs, fusion_motion_inputs, target,
                    session, work_indices, num_frames)

    def __len__(self):
        return len(self.inputs)


@registry.register('LSTRStruggle')
class LSTRDataLayer_Struggle(data.Dataset):

    def __init__(self, cfg, phase='train'):
        self.data_root = cfg.DATA.DATA_ROOT
        # self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        # self.motion_feature = cfg.INPUT.MOTION_FEATURE
        # self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET') # now is 'activity-vidname' per entity
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH # LONG_MEMORY_SECONDS 512*3.125=1600
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE # 4
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES # 1600 // 4 = 400
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH # WORK_MEMORY_SECONDS 8*3.125=25
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE # 1
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES # 25 // 1 = 25
        self.training = phase == 'train'
        self.num_classes = cfg.DATA.NUM_CLASSES
        self.features_fps = cfg.DATA.FPS # 3.125
        self.annotation_path = cfg.DATA.DATA_SPLIT_PATH 
        
        # The Anticipation choice and Future choice are added for the video anticipation task
        # Anticipation choice
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH # default 0, set 0 to disable
        self.anticipation_sample_rate = cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE
        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES

        self._init_dataset()

    def shuffle(self):
        self._init_dataset()

    def _init_dataset(self):
        self.inputs = []
        import json
        with open(self.annotation_path, 'r') as f:
            data_info = json.load(f)
        for session in self.sessions:
            # target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
            activity_name, video_name = session.split('-')
            features = np.load(osp.join(self.data_root, 'extracted_features', 'slowfast_features', activity_name, video_name + '.npy'), mmap_mode='r')
            target = np.zeros((features.shape[0], self.num_classes), dtype=np.float32) # initial target shape (L, 2)
            target[:, 0] = 1.0 # background
            
            try:
                annotations = data_info['database'][video_name]['annotations']
            except KeyError:
                try:
                    annotations = data_info['database'][activity_name + '-' + video_name]['annotations']
                except KeyError:
                    # Handle the case where neither key exists
                    print(f"Could not find annotations for {video_name} or {activity_name + '-' + video_name}")
                    annotations = None

            # print('--------------------------------------------')
            if len(annotations) == 0:
                # no struggle action, all background 
                continue
            else:
                for annotation in annotations:
                    if annotation['label'] == 'Struggle':
                        # struggle action
                        start = int(annotation['segment'][0] * self.features_fps)
                        end = int(annotation['segment'][1] * self.features_fps)
                        # print(start, end)
                        # print(target.shape)
                        target[start:end, 1] = 1.0
                        target[start:end, 0] = 0.0
                    else:
                        ValueError(f"Unknown action {annotation['label']} for video {video_name}")
            
            seed = np.random.randint(self.work_memory_length) if self.training else 0
            for work_start, work_end in zip(
                range(seed, target.shape[0], self.work_memory_length + self.anticipation_length),
                range(seed + self.work_memory_length, target.shape[0] - self.anticipation_length, self.work_memory_length + self.anticipation_length)):
                self.inputs.append([
                    session, work_start, work_end, target, 
                ]) # work_end - work_start = self.work_memory_length, target contains all the frames

    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, target = self.inputs[index]
        activity_name, video_name = session.split('-')
        visual_inputs = np.load(
            osp.join(self.data_root, 'extracted_features', 'slowfast_features', activity_name, video_name + '.npy'), mmap_mode='r')[:, :2048]
        motion_inputs = np.load(
            osp.join(self.data_root, 'extracted_features', 'slowfast_features', activity_name, video_name + '.npy'), mmap_mode='r')[:, 2048:]

        # Get target
        # target = target[work_start: work_end][::self.work_memory_sample_rate]

        # Get work memory
        if self.anticipation_num_samples > 0:
            # Anticipation Setting
            target = target[work_start: work_end + self.anticipation_length]
            target = np.concatenate((target[:self.work_memory_length:self.work_memory_sample_rate],
                                     target[self.work_memory_length::self.anticipation_sample_rate]),
                                    axis=0)
            work_indices = np.arange(work_start, work_end).clip(0)
            work_indices = work_indices[::self.work_memory_sample_rate]
            work_visual_inputs = visual_inputs[work_indices]
            work_motion_inputs = motion_inputs[work_indices]
        else:
            # Normal Online Action Detection Setting
            target = target[work_start: work_end][::self.work_memory_sample_rate]
            work_indices = np.arange(work_start, work_end).clip(0)
            work_indices = work_indices[::self.work_memory_sample_rate]
            work_visual_inputs = visual_inputs[work_indices]
            work_motion_inputs = motion_inputs[work_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            if self.training:
                long_indices = self.segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = self.uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1 # Finding the last zero index
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target
        else:
            return fusion_visual_inputs, fusion_motion_inputs, target

    def __len__(self):
        return len(self.inputs)


@registry.register('LSTRBatchInferenceStruggle')
class LSTRBatchInferenceDataLayer_Struggle(data.Dataset):

    def __init__(self, cfg, phase='test'):
        self.data_root = cfg.DATA.DATA_ROOT
        # self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        # self.motion_feature = cfg.INPUT.MOTION_FEATURE
        # self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET') # now is 'activity-vidname' per entity
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH # LONG_MEMORY_SECONDS 512*3.125=1600
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE # 4
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES # 1600 // 4 = 400
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH # WORK_MEMORY_SECONDS 8*3.125=25
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE # 1
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES # 25 // 1 = 25
        self.num_classes = cfg.DATA.NUM_CLASSES
        self.features_fps = cfg.DATA.FPS # 3.125
        self.annotation_path = cfg.DATA.DATA_SPLIT_PATH 

        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH # default 0, set 0 to disable
        self.anticipation_sample_rate = cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE
        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES

        assert phase == 'test', 'phase must be `test` for batch inference, got {}'

        self.inputs = []
        import json
        with open(self.annotation_path, 'r') as f:
            data_info = json.load(f)
        for session in self.sessions:
            # target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
            activity_name, video_name = session.split('-')
            features = np.load(osp.join(self.data_root, 'extracted_features', 'slowfast_features', activity_name, video_name + '.npy'), mmap_mode='r')
            # print(features.shape)
            target = np.zeros((features.shape[0], self.num_classes), dtype=np.float32) # initial target shape (L, 2)
            target[:, 0] = 1.0 # background
            # print('--------------------------------------------')
            try:
                annotations = data_info['database'][video_name]['annotations']
            except KeyError:
                try:
                    annotations = data_info['database'][activity_name + '-' + video_name]['annotations']
                except KeyError:
                    # Handle the case where neither key exists
                    print(f"Could not find annotations for {video_name} or {activity_name + '-' + video_name}")
                    annotations = None
            
            if len(annotations) == 0:
                # no struggle action, all background 
                continue
            else:
                for annotation in annotations:
                    if annotation['label'] == 'Struggle':
                        # struggle action
                        start = int(annotation['segment'][0] * self.features_fps)
                        end = int(annotation['segment'][1] * self.features_fps)
                        # print(start, end)
                        # print(target.shape)
                        target[start:end, 1] = 1.0
                        target[start:end, 0] = 0.0
                        # print(target)   
                    else:
                        ValueError(f"Unknown action {annotation['label']} for video {video_name}")
            # print(target)
            for work_start, work_end in zip(
                range(0, target.shape[0]),
                range(self.work_memory_length, target.shape[0] - self.anticipation_length)):
                self.inputs.append([
                    session, work_start, work_end, target, target.shape[0]
                ])

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)
    
    def future_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((indices, np.full(padding, indices[-1])))[:num_samples]
        assert num_samples == indices.shape[0], f"{indices.shape}"
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, target, num_frames = self.inputs[index]

        activity_name, video_name = session.split('-')
        visual_inputs = np.load(
            osp.join(self.data_root, 'extracted_features', 'slowfast_features', activity_name, video_name + '.npy'), mmap_mode='r')[:, :2048]
        motion_inputs = np.load(
            osp.join(self.data_root, 'extracted_features', 'slowfast_features', activity_name, video_name + '.npy'), mmap_mode='r')[:, 2048:]

        # Get target
        # total_target = copy.deepcopy(target)
        # target = target[work_start: work_end][::self.work_memory_sample_rate]
        if self.anticipation_num_samples > 0:
            # Anticipation Setting
            target = target[work_start: work_end + self.anticipation_length]
            target = np.concatenate((target[:self.work_memory_length:self.work_memory_sample_rate],
                                     target[self.work_memory_length::self.anticipation_sample_rate]),
                                    axis=0)
        else:
            # Normal Online Action Detection Setting
            target = target[work_start: work_end][::self.work_memory_sample_rate]
        
        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]

        if self.anticipation_num_samples > 0:
            anticipation_indices = np.arange(work_end, work_end + self.anticipation_length).clip(0)
            anticipation_indices = anticipation_indices[::self.anticipation_sample_rate]
            work_indices = np.concatenate((work_indices, anticipation_indices))

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            long_indices = self.uniform_sampler(
                long_start,
                long_end,
                self.long_memory_num_samples,
                self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target,
                    session, work_indices, num_frames)
        else:
            return (fusion_visual_inputs, fusion_motion_inputs, target,
                    session, work_indices, num_frames)

    def __len__(self):
        return len(self.inputs)
    