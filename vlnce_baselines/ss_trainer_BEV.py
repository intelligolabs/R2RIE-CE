import os
import random
import warnings
from collections import defaultdict
from typing import Dict
import jsonlines
import wandb

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.stats import norm
import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import (
    construct_envs,
    is_slurm_batch_job,
)
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST

from .utils import get_camera_orientations12
from vlnce_baselines.common.utils import gather_list_and_concat
from habitat_extensions.measures import NDTW
from fastdtw import fastdtw


import torch.distributed as distr
import gzip
import json
import cv2
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
from vlnce_baselines.models.bev_utils import transfrom3D, bevpos_polar, PointCloud
from vlnce_baselines.models.graph_utils import heading_from_quaternion
import torch.nn.functional as F
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from colorama import Fore, Style, Back
from colorama import init as init_colorama
init_colorama(autoreset=True)

from sklearn.metrics import roc_auc_score

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    device = tensors[0].device
    output = torch.zeros(*size, dtype=dtype).to(device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


@baseline_registry.register_trainer(name="SS-BEV")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(
            config.IL.max_traj_len
        )  #  * 0.97 transfered gt path got 0.96 spl

    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        if not self.config.MODEL.TRAIN_TRAJECTORY_MATCHING.train: raise NotImplementedError("Training trajectory matching model is the only option supported for now")
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.trajectory_matching_optimizer.state_dict() if self.config.MODEL.TRAIN_TRAJECTORY_MATCHING.train else self.optimizer.state_dict(),
                "iteration": iteration,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}_tm.pth"),
        )

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT

        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[
            self.config.local_rank
        ]
        self.config.use_pbar = not is_slurm_batch_job()
        """ if choosing image """
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = (
            self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        )
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = (
            crop_config
        )
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.0
            orient_dict = {
                "Back": [0, math.pi + shift, 0],  # Back
                "Down": [-math.pi / 2, 0 + shift, 0],  # Down
                "Front": [0, 0 + shift, 0],  # Front
                "Right": [0, math.pi / 2 + shift, 0],  # Right
                "Left": [0, 3 / 2 * math.pi + shift, 0],  # Left
                "Up": [math.pi / 2, 0 + shift, 0],  # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(
                    self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR"
                )
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(
                        self.config.TASK_CONFIG.SIMULATOR,
                        camera_template,
                        camera_config,
                    )
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(
                        camera_template
                    )
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)

        if self.local_rank == 0:
            # init wandb
            if self.config.MODEL.WANDB.use:
                name = self.config.MODEL.WANDB.run_name
                run = wandb.init(
                    project="VLN uncertainty",
                    save_code=True,
                    name=name,
                    notes=self.config.MODEL.WANDB.notes,
                )
                artifact = wandb.Artifact(name="configuration", type="config")
                artifact.add_dir(local_path="run_r2r")
                run.log_artifact(artifact)

        if self.world_size > 1:
            distr.init_process_group(backend="nccl", init_method="env://")
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME), auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(
            f"LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}"
        )
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _build_projector(self):
        self.bev_dim = 11
        self.bev_res = 1
        self.projector = PointCloud(
            math.radians(90),
            1,
            feature_map_height=14,
            feature_map_width=14,
            map_dim=self.bev_dim,
            map_res=self.bev_res,
            world_shift_origin=torch.FloatTensor([0, 0, 0]).to(self.device),
            z_clip_threshold=0.5,
            device=self.device,
        )

        self.bev_pos = bevpos_polar(self.bev_dim).to(self.device)
        self.bev_pos = self.bev_pos[None, :, :, :]  # 1 x 11 x 11 x 3

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        """ initialize the waypoint predictor here """
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM

        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = (
            "data/wp_pred/check_cwp_bestdist_hfov63"
            if self.config.MODEL.task_type == "rxr"
            else "data/wp_pred/check_cwp_bestdist_hfov90"
        )
        self.waypoint_predictor.load_state_dict(
            torch.load(cwp_fn, map_location=torch.device("cpu"))["predictor"][
                "state_dict"
            ]
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers
        self._build_projector()

        if self.config.MODEL.TRAIN_TRAJECTORY_MATCHING.train:
            print(Fore.YELLOW + "Training trajectory matching model")
            
        if self.config.GPU_NUMBERS > 1:
            print("Using", self.config.GPU_NUMBERS, "GPU!")
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(
                self.policy.net.to(self.device),
                device_ids=[self.device],
                output_device=self.device,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
        if self.config.MODEL.TRAIN_TRAJECTORY_MATCHING.train:
            if self.config.GPU_NUMBERS > 1:
                self.trajectory_matching_optimizer = torch.optim.AdamW(
                    self.policy.net.module.vln_bert.trajectory_matching_original_bev.parameters(), lr=1e-4
                )
            else:
                self.trajectory_matching_optimizer = torch.optim.AdamW(
                    self.policy.net.vln_bert.trajectory_matching_original_bev.parameters(), lr=1e-4
                )
            
        else:
            policy_parameters = [
                param for name, param in self.policy.named_parameters() 
                if 'trajectory_matching_original_bev' not in name
            ]
            self.optimizer = torch.optim.AdamW(
                policy_parameters, lr=self.config.IL.lr
            )
           
        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob

                ckpt_list = list(
                    filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*"))
                )
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            start_iter = ckpt_dict["iteration"]

            if (
                "module" in list(ckpt_dict["state_dict"].keys())[0]
                and self.config.GPU_NUMBERS == 1
            ):
                self.policy.net = torch.nn.DataParallel(
                    self.policy.net.to(self.device),
                    device_ids=[self.device],
                    output_device=self.device,
                )
                try:
                    self.policy.load_state_dict(ckpt_dict["state_dict"], strict=True)
                except RuntimeError as e:
                    # the traj moduel can be  trained from the optimal policy (without trajectory matching) - check if it is the case
                    if not "train" in  self.config.TASK_CONFIG.DATASET.SPLIT: raise e

                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(
                    self.waypoint_predictor.to(self.device),
                    device_ids=[self.device],
                    output_device=self.device,
                )
            else:
                if self.config.GPU_NUMBERS > 1:
                    # due to DDPm and data parallel, keys can contain "module." if trained with 2 gpus (DDP)
                    key = list(ckpt_dict["state_dict"].keys())[0]
                    if "net.module" not in key:
                        state_dict = {k.replace("net.", "net.module.", 1): v for k, v in ckpt_dict["state_dict"].items()} # replace only the first occurence of net
                        ckpt_dict["state_dict"] = state_dict
                try:
                    self.policy.load_state_dict(ckpt_dict["state_dict"], strict=True)
                except RuntimeError as e:
                    # the traj moduel can be  trained from the optimal policy (without trajectory matching) - check if it is the case
                    
                    if not "train" in  self.config.TASK_CONFIG.DATASET.SPLIT: 
                        print(Fore.RED + f"Error loading checkpoint {ckpt_path}")
                        raise e
            logger.info(
                f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}"
            )

        if self.config.MODEL.TRAIN_TRAJECTORY_MATCHING.train:
            # put require grad to none to the rest of the model
            for name, param in self.policy.named_parameters():
                if "trajectory_matching_original_bev" not in name:
                    param.requires_grad = False 
        
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        logger.info(
            f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB."
        )
        logger.info("Finished setting up policy.")

        return start_iter

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == "r2r":
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(
                        j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k}
                    )
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == "rxr":
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append(
                    {
                        "ref_path": self.gt_data[str(current_episodes[i].episode_id)][
                            "locations"
                        ],
                        "angles": batch_angles[i],
                        "distances": batch_distances[i],
                        "candidate_length": candidate_lengths[i],
                    }
                )
            oracle_cand_idx = self.envs.call(
                ["get_cand_idx"] * self.envs.num_envs, kargs
            )
            return oracle_cand_idx

    def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left):
        teacher_actions = []
        cur_episodes = self.envs.current_episodes()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(
            zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)
        ):
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal")
            if curr_dis_to_goal < 1.5:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy == "spl":
                    ghost_vp_pos = [
                        (vp, random.choice(pos))
                        for vp, pos in gmap.ghost_real_pos.items()
                    ]
                    ghost_dis_to_goal = [
                        self.envs.call_at(i, "point_dist_to_goal", {"pos": p[1]})
                        for p in ghost_vp_pos
                    ]
                    target_ghost_vp = ghost_vp_pos[np.argmin(ghost_dis_to_goal)][0]
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                elif self.config.IL.expert_policy == "ndtw":
                    ghost_vp_pos = [
                        (vp, random.choice(pos))
                        for vp, pos in gmap.ghost_real_pos.items()
                    ]
                    target_ghost_vp = self.envs.call_at(
                        i,
                        "ghost_dist_to_ref",
                        {
                            "ghost_vp_pos": ghost_vp_pos,
                            "ref_path": self.gt_data[str(cur_episodes[i].episode_id)][
                                "locations"
                            ],
                        },
                    )
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                else:
                    raise NotImplementedError

        return torch.tensor(teacher_actions).cuda()

    def _vp_feature_variable(self, obs):
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []

        for i in range(self.envs.num_envs):
            rgb_fts, dep_fts, loc_fts, nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=np.bool)
            cand_idxes[obs["cand_img_idxes"][i]] = True
            # cand
            rgb_fts.append(obs["cand_rgb"][i])
            dep_fts.append(obs["cand_depth"][i])
            loc_fts.append(obs["cand_angle_fts"][i])
            nav_types += [1] * len(obs["cand_angles"][i])
            # non-cand
            rgb_fts.append(obs["pano_rgb"][i][~cand_idxes])
            dep_fts.append(obs["pano_depth"][i][~cand_idxes])
            loc_fts.append(obs["pano_angle_fts"][~cand_idxes])
            nav_types += [0] * (12 - np.sum(cand_idxes))

            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))
        # collate
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            "rgb_fts": batch_rgb_fts,
            "dep_fts": batch_dep_fts,
            "loc_fts": batch_loc_fts,
            "nav_types": batch_nav_types,
            "view_lens": batch_view_lens,
        }

    def lift(self, cur_pos, cur_ori, rgb_grid, depth_grid):
        # feature and height of depht mpa -> 14x14
        """unproject rgbs and depths to pointcloud in world coord"""
        bs = self.envs.num_envs
        num_view = 12

        xyzhe = np.zeros([bs, num_view, 5])
        for i, (pos_i, ori_i) in enumerate(zip(cur_pos, cur_ori)):
            x, y, z = pos_i
            xyzhe[i, :, 0] = x
            xyzhe[i, :, 1] = y
            xyzhe[i, :, 2] = z
            xyzhe[i, :, 3] = -np.arange(12) * np.radians(30) + heading_from_quaternion(
                ori_i
            )
            xyzhe[i, :, 4] = np.pi
        T = transfrom3D(xyzhe.reshape(-1, 5))  # bs * NUM_VIEW, 4, 4
        T = torch.from_numpy(T).cuda()  # bs * NUM_VIEW, 4, 4

        depths = depth_grid.reshape(-1, 1, 14, 14).cuda() * 10
        pc, pc_mask = self.projector.forward(depths, T)
        pc = pc.reshape(bs, -1, 3)
        pc_mask = pc_mask.reshape(bs, -1)

        rgbs = rgb_grid.reshape(-1, 14, 14, 768).cuda()
        pc_feat = rgbs.reshape(bs, -1, 768)

        return pc, pc_mask, pc_feat
        # return pc.cpu(), pc_mask.cpu(), pc_feat.cpu()

    def splat(self, cur_pos, cur_ori, pc, pc_mask, pc_feat):
        """
        1. transform pointcloud to ego coord
        2. project to bev
        """
        bs = self.envs.num_envs

        S = []
        for i, pos_i in enumerate(cur_pos):
            x, y, z = pos_i
            S.append([np.array([x, y, z])])
        S = np.vstack(S).astype(np.float32)  # bs, 3
        S = torch.from_numpy(S).cuda()
        xyzhe = np.zeros([bs, 5])
        for i, ori_i in enumerate(cur_ori):
            xyzhe[i, 3] = -heading_from_quaternion(ori_i)
        T = torch.from_numpy(transfrom3D(xyzhe)).cuda()  # bs, 4, 4

        # transform to ego coord
        pc = pc - S[:, None, :]
        ones = torch.ones(pc.shape[:2]).unsqueeze(-1).cuda()
        pc1 = torch.cat([pc, ones], dim=-1)  # bs, N, 4
        pc1 = torch.matmul(pc1, T.transpose(1, 2))  # bs, N, 4
        pc = pc1[:, :, :3]  # bs, N, 3

        # project to bev
        # viz = False
        bev_fts, bev_masks = self.projector.project_bev(pc, pc_mask, pc_feat)
        # if viz:
        #     feat_masks = []
        #     for ob, bev_mask in zip(obs, bev_masks):
        #         cand_pos = self._map_cand_to_bev(ob)
        #         bev_mask = bev_mask.cpu().numpy()[:,:,None] * np.array([255,255,255])[None,None,:]
        #         bev_mask = bev_mask.astype(np.uint8)
        #         for p in cand_pos:
        #             bev_mask[p[1], p[0], :] = np.array([0,255,0]).astype(np.uint8)
        #         feat_masks.append(bev_mask)
        #     feat_masks = np.concatenate(feat_masks, axis=1)
        #     cv2.imwrite('feat_masks.png', feat_masks)

        #     bev_imgs = [draw_ob(ob) for ob in obs]
        #     bev_imgs = np.concatenate(bev_imgs, axis=0)
        #     cv2.imwrite('bev_imgs.png', bev_imgs)

        if not self.config.VIDEO_OPTION:
            bev_masks = torch.ones_like(bev_masks)
        bev_fts = bev_fts.reshape(bs, -1, 768)
        bev_masks = bev_masks.reshape(bs, -1)
        bev_pos_fts = self.bev_pos.expand(bs, -1, -1, -1).reshape(bs, -1, 3)

        return bev_fts, bev_masks, bev_pos_fts

    def _discretize_polar_relpos(self, cands_relpos):
        cx = cy = int((self.bev_dim - 1) // 2)
        cands_x = (
            cx
            + (cands_relpos[:, 1] * np.sin(cands_relpos[:, 0]) / self.bev_res).round()
        )
        cands_y = (
            cy
            - (cands_relpos[:, 1] * np.cos(cands_relpos[:, 0]) / self.bev_res).round()
        )
        cands_xy = np.concatenate([cands_x[:, None], cands_y[:, None]], axis=1)
        cands_xy[cands_xy < 0] = 0
        cands_xy[cands_xy >= self.bev_dim] = self.bev_dim - 1

        return cands_xy.astype(np.int)

    def _nav_bev_variable(self, cur_vp, cur_pos, cur_ori):
        batch_pc = []
        batch_pc_mask = []
        batch_pc_feat = []

        batch_bev_nav_masks = []
        batch_bev_cand_vpids = []
        batch_bev_cand_idxs = []
        batch_gmap_pos_fts = []

        for i, gmap in enumerate(self.gmaps):
            pc, pc_mask, pc_feat = gmap.gather_node_pc(cur_vp[i], order=1)
            batch_pc.append(pc)
            batch_pc_mask.append(pc_mask)
            batch_pc_feat.append(pc_feat)
            # map first-order candidate to bev, including cur_vp
            # cands_relpos: (ang, dis), clockwise
            cands_vp, cands_relpos = gmap.get_neighbors(
                cur_vp[i], cur_pos[i], cur_ori[i]
            )
            cands_idx_2d = self._discretize_polar_relpos(cands_relpos)
            cands_idx = cands_idx_2d[:, 1] * self.bev_dim + cands_idx_2d[:, 0]
            bev_nav_masks = np.zeros(self.bev_dim * self.bev_dim).astype(np.bool)
            bev_nav_masks[cands_idx] = True
            batch_bev_cand_vpids.append(cands_vp)
            batch_bev_nav_masks.append(torch.from_numpy(bev_nav_masks))
            batch_bev_cand_idxs.append(torch.from_numpy(cands_idx))
            # global relative pos
            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], ["0"]
            )  # '0' is the start node
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))

        # collate
        batch_pc = pad_sequence(batch_pc, batch_first=True).cuda()
        batch_pc_mask = pad_sequence(
            batch_pc_mask, batch_first=True, padding_value=True
        ).cuda()  # no depth mask
        batch_pc_feat = pad_sequence(batch_pc_feat, batch_first=True).cuda()
        batch_bev_nav_masks = pad_tensors(batch_bev_nav_masks).cuda()
        batch_bev_cand_idxs = pad_tensors(batch_bev_cand_idxs)
        batch_gmap_pos_fts = (
            pad_tensors(batch_gmap_pos_fts)
            .expand(-1, self.bev_dim * self.bev_dim, -1)
            .cuda()
        )

        bev_fts, bev_masks, bev_pos_fts = self.splat(
            cur_pos, cur_ori, batch_pc, batch_pc_mask, batch_pc_feat
        )
        bev_pos_fts = torch.cat([batch_gmap_pos_fts, bev_pos_fts], dim=-1)
        if self.config.VIDEO_OPTION:
            bev_masks_vis = bev_masks[:, :, None].expand(-1, -1, 3).int() * 255
            bev_masks_vis[batch_bev_nav_masks] = torch.IntTensor([0, 0, 255]).cuda()
            bev_masks_vis = bev_masks_vis.cpu().numpy()
            bev_masks_vis = bev_masks_vis.reshape(self.envs.num_envs, 11, 11, 3)
            bev_masks_vis = [bev_masks_vis[i] for i in range(self.envs.num_envs)]
            bev_masks_vis = np.concatenate(bev_masks_vis, axis=1)
            cv2.imwrite("bev_masks.png", bev_masks_vis)

        return {
            "bev_fts": bev_fts,
            "bev_pos_fts": bev_pos_fts,
            "bev_masks": bev_masks,
            "bev_nav_masks": batch_bev_nav_masks,
            "bev_cand_idxs": batch_bev_cand_idxs,
            "bev_cand_vpids": batch_bev_cand_vpids,
        }

    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = (
                [0]
                + [gmap.node_stepId[vp] for vp in node_vp_ids]
                + [0] * len(ghost_vp_ids)
            )
            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + [
                gmap.get_node_embeds(vp) for vp in ghost_vp_ids
            ]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros(
                (len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32
            )
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j + 1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith("g") and not vp2.startswith("g"):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith("g") and vp2.startswith("g"):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith("g") and vp2.startswith("g"):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = (
                            front_dis1
                            + gmap.shortest_dist[front_vp1][front_vp2]
                            + front_dis2
                        )
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST

            batch_gmap_vp_ids.append(gmap_vp_ids)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))

        # collate
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(
            batch_gmap_visited_masks, batch_first=True
        ).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[
                i, : batch_gmap_lens[i], : batch_gmap_lens[i]
            ] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            "gmap_vp_ids": batch_gmap_vp_ids,
            "gmap_step_ids": batch_gmap_step_ids,
            "gmap_img_fts": batch_gmap_img_fts,
            "gmap_pos_fts": batch_gmap_pos_fts,
            "gmap_masks": batch_gmap_masks,
            "gmap_visited_masks": batch_gmap_visited_masks,
            "gmap_pair_dists": gmap_pair_dists,
            "no_vp_left": batch_no_vp_left,
        }

    def _history_variable(self, obs):
        batch_size = obs["pano_rgb"].shape[0]
        hist_rgb_fts = obs["pano_rgb"][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs["pano_rgb"].cuda()
        hist_pano_ang_fts = (
            obs["pano_angle_fts"].unsqueeze(0).expand(batch_size, -1, -1).cuda()
        )

        return hist_rgb_fts, hist_pano_rgb_fts, hist_pano_ang_fts

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            for k, v in batch.items():
                batch[k] = v[state_index]

        return envs, batch

    def train(self):
        self._set_config()

        if self.config.MODEL.task_type == "rxr":
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ),
                    "rt",
                ) as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )
        
        if self.config.MODEL.TRAIN_TRAJECTORY_MATCHING.train:
            self.trajectory_matching_scaler = GradScaler()
        else:
            self.scaler = GradScaler()

        total_iter = self.config.IL.iters
        log_every = self.config.IL.log_every
        writer = TensorboardWriter(
            self.config.TENSORBOARD_DIR if self.local_rank < 1 else None
        )

        
        logger.info("Traning Starts... GOOD LUCK!")
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter - idx, 0))
            cur_iter = idx + interval

            sample_ratio = self.config.IL.sample_ratio ** (
                idx // self.config.IL.decay_interval + 1
            )
            # sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval)
            logs = self._train_interval(
                interval, self.config.IL.ml_weight, sample_ratio
            )

            if self.local_rank < 1:
                loss_str = f"iter {cur_iter}: "
                wandb_dict = {}
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f"{k}: {logs[k]:.3f}, "
                    writer.add_scalar(f"loss/{k}", logs[k], cur_iter)
                    wandb_dict[k] = logs[k]
                if self.config.MODEL.WANDB.use:
                    wandb.log({"train": {**wandb_dict}}, step=cur_iter, commit=True)
                logger.info(loss_str)
                self.save_checkpoint(cur_iter)

    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.train()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()
        self.waypoint_predictor.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        for idx in pbar:
            
            if self.config.MODEL.TRAIN_TRAJECTORY_MATCHING.train:
                self.trajectory_matching_optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
            self.loss = 0.0
            self.loss_trajectory_matching = 0.0

            with autocast():
                self.rollout("train", ml_weight, sample_ratio)
            if self.config.MODEL.TRAIN_TRAJECTORY_MATCHING.train:
                self.trajectory_matching_scaler.scale(self.loss_trajectory_matching).backward()
                self.trajectory_matching_scaler.step(self.trajectory_matching_optimizer)
                self.trajectory_matching_scaler.update()
            else:
                self.scaler.scale(self.loss).backward()  # self.loss.backward()
                self.scaler.step(self.optimizer)  # self.optimizer.step()
                self.scaler.update()

            if self.local_rank < 1:
                pbar.set_postfix({"iter": f"{idx+1}/{interval}"})

        return deepcopy(self.logs)

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        iteration = checkpoint_path.split("/")[-1].split(".")[1]
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.0
            orient_dict = {
                "Back": [0, math.pi + shift, 0],  # Back
                "Down": [-math.pi / 2, 0 + shift, 0],  # Down
                "Front": [0, 0 + shift, 0],  # Front
                "Right": [0, math.pi / 2 + shift, 0],  # Right
                "Left": [0, 3 / 2 * math.pi + shift, 0],  # Left
                "Up": [math.pi / 2, 0 + shift, 0],  # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(
                    self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR"
                )
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(
                        self.config.TASK_CONFIG.SIMULATOR,
                        camera_template,
                        camera_config,
                    )
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(
                        camera_template
                    )
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{iteration}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(
                self.config.EVAL.CKPT_PATH_DIR
            ):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::5]
            if self.config.EVAL.fast_eval
            else self.traj,
            auto_reset_done=False,  # unseen: 11006
        )
        print("Number for each scene:", (self.envs.number_of_episodes))
        dataset_length = sum(self.envs.number_of_episodes)
        print("local rank:", self.local_rank, "|", "dataset length:", dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(
                self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes)
            )
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval)


        while len(self.stat_eps) < eps_to_eval:
            self.rollout("eval")
        self.envs.close()

        if self.world_size > 1:
            distr.barrier()

        # since ATD metric is calculate for all the episode, but not every episode has this metric, we need to calculate it manually
        atd_metric_tmp = 0
        counter_atd = 0
        for ep_id, metric_values in self.stat_eps.items():
            if "ATD" in metric_values.keys():
                counter_atd += 1
                atd_metric_tmp += metric_values["ATD"]
                del metric_values["ATD"]


        try:
            atd_metric = atd_metric_tmp / counter_atd # divided by the number of episodes
        except:
            print(Fore.RED + "we did not find any ATD metric in the episodes")
            atd_metric = -1
        aggregated_states = {"ATD": atd_metric}

        # Not consider this metric, since we calculated them with sklearn
        task_1_metric_keys = ["instruction_contains_an_error_prediction", "instruction_contains_an_error_gt"]
       
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            if stat_key in task_1_metric_keys: continue

            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        ##### calculate task 1 metric res - ROC-AUC_scor
        instruction_contains_an_error_gt = []
        instruction_contains_an_error_prediction = []
        for ep_id, metric_values in self.stat_eps.items():
            instruction_contains_an_error_gt.append(metric_values["instruction_contains_an_error_gt"])
            instruction_contains_an_error_prediction.append(metric_values["instruction_contains_an_error_prediction"])



        ######### ROC AUC calculation
        instruction_contains_an_error_gt = {self.local_rank : instruction_contains_an_error_gt}
        instruction_contains_an_error_prediction = {self.local_rank : instruction_contains_an_error_prediction}

        output_list_contains_an_error_gt = [torch.zeros_like(torch.empty(1)) for _ in range(self.world_size)]
        output_list_contains_an_error_prediction = [torch.zeros_like(torch.empty(1)) for _ in range(self.world_size)]

        if self.world_size > 1:
            distr.all_gather_object(output_list_contains_an_error_gt, instruction_contains_an_error_gt)
            distr.all_gather_object(output_list_contains_an_error_prediction, instruction_contains_an_error_prediction)

            if self.local_rank == 0:
                list_contains_an_error_gt = []
                list_contains_an_error_prediction =  []
                for el in output_list_contains_an_error_gt:
                    _rank_labels = list(el.keys())[0] #0 since its dickt keys
                    list_contains_an_error_gt += list(el.values())[0]

                    for preds in output_list_contains_an_error_prediction:
                        _rank_preds = list(preds.keys())[0]
                        if _rank_labels == _rank_preds: # take the element with the same rank, so we know the are ordered
                            list_contains_an_error_prediction += list(preds.values())[0]
                            break
                assert len(list_contains_an_error_gt) == len(list_contains_an_error_prediction), "dimension of list does not match"
                if len(np.unique(np.array(list_contains_an_error_gt))) == 1:
                    # we don't have error, roc score defined only when at least one class is present:
                    aggregated_states['ROC-AUC_score_instr_contains_an_error'] = -1
                else: 
                    aggregated_states['ROC-AUC_score_instr_contains_an_error'] = roc_auc_score(list_contains_an_error_gt, list_contains_an_error_prediction)
                    aggregated_states['ROC-AUC_score_num_episodes'] = len(list_contains_an_error_prediction)
        else: # we have only one rank
            if len(np.unique(np.array(list_contains_an_error_gt))) == 1:
                aggregated_states['ROC-AUC_score_instr_contains_an_error'] = -1
            else:
                aggregated_states['ROC-AUC_score_instr_contains_an_error'] = roc_auc_score(instruction_contains_an_error_gt[0], instruction_contains_an_error_prediction[0])
                aggregated_states['ROC-AUC_score_num_episodes'] = len(instruction_contains_an_error_prediction[0])

        #########

        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total, dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(
                f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}"
            )
            for k, v in aggregated_states.items():
                if k == 'ATD': continue
                if "ROC-AUC_score" in k: continue # we have already calculated in a  better way
            
                v = torch.tensor(v * num_episodes).cuda()
                cat_v = gather_list_and_concat(v, self.world_size)
                v = (sum(cat_v) / total).item()
                aggregated_states[k] = v

        # save also the ckpt name
        aggregated_states["ckpt_name"] = checkpoint_path
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                if isinstance(v, float):
                    logger.info(f"Average episode {k}: {v:.6f}")
                else:
                    logger.info(f"Average episode {k}: {v}")
                #writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ["POSITION_INFER"]
        self.config.TASK_CONFIG.TASK.SENSORS = [
            s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s
        ]
        self.config.SIMULATOR_GPU_IDS = [
            self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        ]
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = (
            self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        )
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = (
            crop_config
        )
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend="nccl", init_method="env://")
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_infer_traj()

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(
                self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes)
            )
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}  # transfer submit format
        self.pbar = tqdm.tqdm(total=eps_to_infer)

        while len(self.path_eps) < eps_to_infer:
            self.rollout("infer")
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict

        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(
                f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}"
            )
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k, v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]:
                        path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(
                self.config.INFERENCE.PREDICTIONS_FILE, mode="w"
            ) as writer:
                writer.write_all(preds)
            logger.info(
                f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}"
            )

    def get_pos_ori(self):
        pos_ori = self.envs.call(["get_pos_ori"] * self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    
    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        if mode == "train":
            feedback = "sample"
        elif mode == "eval" or mode == "infer":
            feedback = "argmax"
        else:
            raise NotImplementedError

        if self.config.MODEL.USE_GT_TRAJECTORY.use:
            print(Back.RED + Fore.WHITE + "You are now using the Ground Truth actions!")

        self.envs.resume_all()
        observations = self.envs.reset()
        instr_max_len = self.config.IL.max_text_len  # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == "rxr" else 0
        observations = extract_instruction_tokens(
            observations,
            self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            max_length=instr_max_len,
            pad_id=instr_pad_id,
        )
        
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        

        if mode == "eval":
            env_to_pause = [
                i
                for i, ep in enumerate(self.envs.current_episodes())
                if ep.episode_id in self.stat_eps
            ]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0:
                return
        if mode == "infer":
            env_to_pause = [
                i
                for i, ep in enumerate(self.envs.current_episodes())
                if ep.episode_id in self.path_eps
            ]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0:
                return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == "rxr":
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        # encode instructions
        all_txt_ids = batch["instruction"]
        all_episode_contains_an_error_gt = batch["error_sensor_contains_error"]
        all_token_position_error_gt = batch['error_sensor_token_swapped_error']
        all_txt_masks = all_txt_ids != instr_pad_id
        all_txt_embeds = self.policy.net(
            mode="language",
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )

        
        batch_size = all_txt_ids.size()[0]

        all_node_features_mask = torch.zeros((batch_size, 16), dtype=torch.bool, device=self.device)
        all_node_features_embeds = torch.zeros((batch_size, 16, 768), device=self.device)


        loss = 0.0
        total_actions = 0.0
        not_done_index = list(range(self.envs.num_envs))

        have_real_pos = mode == "train" or self.config.VIDEO_OPTION or self.config.MODEL.USE_GT_TRAJECTORY.use
        ghost_aug = self.config.IL.ghost_aug if mode == "train" else 0
        self.gmaps = [
            GraphMap(
                have_real_pos,
                self.config.IL.loc_noise,
                self.config.MODEL.merge_ghost,
                ghost_aug,
            )
            for _ in range(self.envs.num_envs)
        ]

        # save ep id into gmaps
        curr_eps = self.envs.current_episodes()
        for i, gmap in enumerate(self.gmaps):
            gmap.set_ep_id(curr_eps[i].episode_id)
            gmap.set_index_for_history_information(i)

        prev_vp = [None] * self.envs.num_envs

        for stepk in range(self.max_len):
            if stepk > 0 and mode  == 'eval':
                reverse = {ix: v for v,ix in  HabitatSimActions._known_actions.items()} 

            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]


            # cand waypoint prediction
            wp_outputs = self.policy.net(
                mode="waypoint",
                waypoint_predictor=self.waypoint_predictor,
                observations=batch,
                in_train=(mode == "train" and self.config.IL.waypoint_aug),
            )

            # graph representation
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update(
                {
                    "mode": "panorama",
                }
            )
            pano_embeds, pano_masks = self.policy.net(
                **vp_inputs
            )  # panoramic images with different type of embeddings
            avg_pano_embeds = torch.sum(
                pano_embeds * pano_masks.unsqueeze(2), 1
            ) / torch.sum(pano_masks, 1, keepdim=True)


            # get vp_id, vp_pos of cur_node and cand_ndoe
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i],
                    cur_ori[i],
                    wp_outputs["cand_angles"][i],
                    wp_outputs["cand_distances"][i],
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)

            # pointcloud representation -  # NEW IN BEVBERT
            pc, pc_mask, pc_feat = self.lift(
                cur_pos,
                cur_ori,
                wp_outputs["pano_rgb_grid"],
                wp_outputs["pano_depth_grid"],
            )

            if mode == "train" or self.config.VIDEO_OPTION or self.config.MODEL.USE_GT_TRAJECTORY.use:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(
                            i, "get_cand_real_pos", {"angle": ang, "forward": dis}
                        )
                        for ang, dis in zip(
                            wp_outputs["cand_angles"][i],
                            wp_outputs["cand_distances"][i],
                        )
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs["nav_types"][i] == 1]
                self.gmaps[i].update_graph(
                    prev_vp[i],
                    stepk + 1,
                    cur_vp[i],
                    cur_pos[i],
                    cur_embeds,
                    cand_vp[i],
                    cand_pos[i],
                    cand_embeds,
                    cand_real_pos[i],
                )
                self.gmaps[i].update_node_pc(cur_vp[i], pc[i], pc_mask[i], pc_feat[i])

            all_node_features_mask[not_done_index, stepk] = True
            all_node_features_embeds[not_done_index, stepk] = avg_pano_embeds # don't use not done index, since dones env are deleted


            # navigation variable
            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            nav_inputs.update(
                self._nav_bev_variable(cur_vp, cur_pos, cur_ori)
            )  
            nav_inputs.update(
                {
                    "mode": "navigation",
                    "txt_embeds": txt_embeds,
                    "txt_masks": txt_masks,
                    "hist_embeds": None,
                    "hist_mask": None,
                }
            )
            no_vp_left = nav_inputs.pop("no_vp_left")

            (
                nav_outs,
                _,
                _,
                _,
                _,
                _,
            ) = self.policy.net(**nav_inputs)

            nav_logits = nav_outs["fused_logits"] 
            if self.config.MODEL.USE_GT_TRAJECTORY.use:
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
                nav_logits[torch.arange(teacher_actions.shape[0]), teacher_actions] = 10 # high value for softmax
                
            nav_probs = F.softmax(nav_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()


            if mode == "train" or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(
                    nav_inputs["gmap_vp_ids"], no_vp_left
                )
            if mode == "train":
                loss += F.cross_entropy(
                    nav_logits, teacher_actions, reduction="sum", ignore_index=-100
                )

            # determine action
            if feedback == "sample":
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample().detach()
                a_t = torch.where(
                    torch.rand_like(a_t, dtype=torch.float) <= sample_ratio,
                    teacher_actions,
                    a_t,
                )
            elif feedback == "argmax":
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            cpu_a_t = a_t.cpu().numpy()

            # make equiv action
            env_actions = []
            use_tryout = (
                self.config.IL.tryout
                and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING
            )
            for i, gmap in enumerate(self.gmaps):
                
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    # stop at node with max stop_prob
                    vp_stop_scores = [
                        (vp, stop_score)
                        for vp, stop_score in gmap.node_stop_scores.items()
                    ]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]

                   
                    stop_pos = gmap.node_pos[stop_vp]
                    if self.config.IL.back_algo == "control":
                        back_path = [
                            (vp, gmap.node_pos[vp])
                            for vp in gmap.shortest_path[cur_vp[i]][stop_vp]
                        ]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                        "nodes": list(gmap.node_pos.values()),
                        "ghosts": list(gmap.ghost_aug_pos.values()),
                        "predict_ghost": stop_pos,
                        "step_k": stepk
                    }
                    env_actions.append(
                        {
                            "action": {
                                "act": 0,
                                "cur_vp": cur_vp[i],
                                "stop_vp": stop_vp,
                                "stop_pos": stop_pos,
                                "back_path": back_path,
                                "tryout": use_tryout,
                            },
                            "vis_info": vis_info,
                        }
                    )
                else:  # continue navigation to the node with max score
                    ghost_vp = nav_inputs["gmap_vp_ids"][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[
                                nav_inputs["gmap_vp_ids"][i][teacher_action_cpu]
                            ]
                        vis_info = {
                            "nodes": list(gmap.node_pos.values()),
                            "ghosts": list(gmap.ghost_aug_pos.values()),
                            "predict_ghost": ghost_pos,
                            "teacher_ghost": teacher_ghost,
                            "step_k": stepk
                        }
                    else:
                        vis_info =  None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == "control":
                        back_path = [
                            (vp, gmap.node_pos[vp])
                            for vp in gmap.shortest_path[cur_vp[i]][front_vp]
                        ]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            "action": {
                                "act": 4,
                                "cur_vp": cur_vp[i],
                                "front_vp": front_vp,
                                "front_pos": front_pos,
                                "ghost_vp": ghost_vp,
                                "ghost_pos": ghost_pos,
                                "back_path": back_path,
                                "tryout": use_tryout,
                            },
                            "vis_info": vis_info,
                            
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]


            # calculate metric
            if mode == "eval":
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]["locations"]).astype(
                        np.float
                    )
                    gmap = self.gmaps[i]
                    EP_ID_TO_INDEX_MAPPING = gmap.episode_id_to_history_index # we use this, since when env is done they are deleted
                   
                    pred_path = np.array(info["position"]["position"])
                    distances = np.array(info["position"]["distance"])
                    metric = {}
                    
                    traj_matcher_input = {
                                "mode": "trajectory_matcher",
                                "txt_embeds": all_txt_embeds[EP_ID_TO_INDEX_MAPPING].unsqueeze(0),
                                "txt_masks": all_txt_masks[EP_ID_TO_INDEX_MAPPING].unsqueeze(0),
                                "hist_embeds": all_node_features_embeds[EP_ID_TO_INDEX_MAPPING].unsqueeze(0),
                                "hist_mask":  all_node_features_mask[EP_ID_TO_INDEX_MAPPING].unsqueeze(0),
                                }
                    cls_logits, token_position_error_hat = self.policy.net(**traj_matcher_input)
                    cls_logits = cls_logits.squeeze(0)

                     
                    # save prediction for "Does the instruction contains an error?"
                    instruction_contains_an_error_gt_data = all_episode_contains_an_error_gt[EP_ID_TO_INDEX_MAPPING].item()
                    metric["instruction_contains_an_error_prediction"] = torch.sigmoid(cls_logits).item()
                    metric["instruction_contains_an_error_gt"] = instruction_contains_an_error_gt_data

                
                    # SAVE PREDICTION FOR THE ATD METRIC
                    ground_truth_token_swapped_positions = all_token_position_error_gt[EP_ID_TO_INDEX_MAPPING] # can also be more than one
                    ground_truth_token_swapped_positions = ground_truth_token_swapped_positions[:, 1] # take the position of the token swapped, not the token ids
                    
                    T = (ground_truth_token_swapped_positions != -1).sum().item() # number of token swapped
                  

                    
                    if instruction_contains_an_error_gt_data:
                            assert ground_truth_token_swapped_positions[0] != -1, "instruction contains error, but no token swapped position is provided"
                   
                    if instruction_contains_an_error_gt_data:
                        if self.config.MODEL.VERBOSE.IS_VERBOSE :
                                print(Fore.LIGHTBLUE_EX + f"Instruction contains an error: {instruction_contains_an_error_gt_data}")
                                print(Fore.LIGHTBLUE_EX + f"ground_truth_token_swapped_positions: {ground_truth_token_swapped_positions[0]}")
                        
                        

                        token_position_error_hat = F.softmax(token_position_error_hat, dim=-1)
                        token_position_error_hat = torch.topk(token_position_error_hat, k=T, dim=-1).indices
                        predicted_token_swapped_positions = token_position_error_hat

                        if self.config.MODEL.VERBOSE.IS_VERBOSE :
                            print(Fore.LIGHTBLUE_EX + f"predicted_token_swapped_positions: {predicted_token_swapped_positions}")
                            token_swapped = batch['error_sensor_token_swapped_error'][i].cpu().numpy()
                            print(Fore.LIGHTBLUE_EX + f"token_swapped: {token_swapped}")

                        
                        # sort the token swapped positions, and the predictions
                        # since we may also contain -1, we sort in descending, and then check how many valid token there are. This way, -1 will not be considered
                        predicted_token_swapped_positions = torch.sort(predicted_token_swapped_positions, descending=True).values # sort by index values
                        ground_truth_token_swapped_positions = torch.sort(ground_truth_token_swapped_positions, descending=True).values # sort by index values
                        
                        ATD = 0
                        for tj in range(T):
                            ATD += torch.abs(predicted_token_swapped_positions[0, tj] - ground_truth_token_swapped_positions[tj]).item()
                        
                        metric["ATD"] = ATD / T

                    
                    metric["steps_taken"] = info["steps_taken"]
                    metric["distance_to_goal"] = distances[-1]
                    metric["success"] = 1.0 if distances[-1] <= 3.0 else 0.0
                    metric["oracle_success"] = 1.0 if (distances <= 3.0).any() else 0.0
                    metric["path_length"] = float(
                        np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum()
                    )
                    metric["collisions"] = info["collisions"]["count"] / len(pred_path)
                    gt_length = distances[0]
                    metric["spl"] = (
                        metric["success"]
                        * gt_length
                        / max(gt_length, metric["path_length"])
                    )
                    dtw_distance = fastdtw(
                        pred_path, gt_path, dist=NDTW.euclidean_distance
                    )[0]
                    metric["ndtw"] = np.exp(-dtw_distance / (len(gt_path) * 3.0))
                    metric["sdtw"] = metric["ndtw"] * metric["success"]
                    metric["ghost_cnt"] = self.gmaps[i].ghost_cnt
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == "infer":
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            "position": info["position_infer"]["position"][0],
                            "heading": info["position_infer"]["heading"][0],
                            "stop": False,
                        }
                    ]
                    for p, h in zip(
                        info["position_infer"]["position"][1:],
                        info["position_infer"]["heading"][1:],
                    ):
                        if p != self.path_eps[ep_id][-1]["position"]:
                            self.path_eps[ep_id].append(
                                {"position": p, "heading": h, "stop": False}
                            )
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]["stop"] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                max_length=instr_max_len,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        # when all the env has terminated
        if mode == "train":
            
            loss = ml_weight * loss / total_actions
            self.logs["IL_loss"].append(loss.item())

            self.loss += loss # not actually used, since we do not finetune the policy but the TM model itself

            ## GET THE LOSS ABOUT THE TRAJECTORY MATCHER
            traj_matcher_input = {"mode": "trajectory_matcher",
                                "txt_embeds": all_txt_embeds,
                                "txt_masks": all_txt_masks,
                                "hist_embeds": all_node_features_embeds,
                                "hist_mask": all_node_features_mask,
                                }
            cls_logits, token_position_error_hat = self.policy.net(**traj_matcher_input)
            bce_loss =  F.binary_cross_entropy_with_logits(cls_logits, all_episode_contains_an_error_gt.type(torch.float16), reduction="mean")

            num_errors = all_token_position_error_gt[:, :].size()[1]
            
            final_token_pos_error_loss = 0
            for ix in range(num_errors):
                # tmp loss of error ix
                tmp_token_pos_loss = F.cross_entropy(input=token_position_error_hat, target=all_token_position_error_gt[:, ix, 1], ignore_index=-1, reduction="mean")
                if not torch.isnan(tmp_token_pos_loss):
                    final_token_pos_error_loss += tmp_token_pos_loss
                

            token_position_error_loss = final_token_pos_error_loss / num_errors
            if not torch.is_tensor(obj=token_position_error_loss):
                token_position_error_loss = torch.tensor(token_position_error_loss, device=loss.device)
            
            self.logs["Traj_Matching"].append(bce_loss.item())
            self.logs["TOK_POS_ERROR"].append(token_position_error_loss.item())

            self.loss_trajectory_matching += ( bce_loss + token_position_error_loss)

            
            
