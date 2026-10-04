from collections import defaultdict
from pathlib import Path
import math
import sys
import time

import torch
import numpy as np
import wandb
import torch.nn.functional as F

from config.config import ProjectConfig
from tridi.utils.training import get_optimizer, get_scheduler, TrainState, resume_from_checkpoint, compute_grad_norm
from tridi.data import get_train_dataloader
from tridi.data.batch_data import BatchData
from tridi.model.wrappers.mesh import MeshModel
from tridi.model.wrappers.contact import ContactModel
from tridi.utils.metrics.reconstruction import get_obj_v2v, get_obj_center_distance, get_mpjpe, get_mpjpe_pa
from tridi.model.base import TriDiModelOutput

from logging import getLogger

logger = getLogger(__name__)


class Trainer:
    def __init__(self, cfg: ProjectConfig, model):
        self.cfg = cfg

        model = model.to("cuda")

        optimizer = get_optimizer(cfg, model)
        scheduler = get_scheduler(cfg, optimizer)

        # Resume from checkpoint and create the initial training state
        self.train_state: TrainState = resume_from_checkpoint(cfg, model, optimizer, scheduler)

        # Get dataloaders
        dataloader_train, dataloader_val, canonical_obj_meshes, canonical_obj_keypoints \
            = get_train_dataloader(cfg)

        self.model = model
        self.mesh_model: MeshModel = MeshModel(
            model_path=cfg.env.smpl_folder,
            batch_size=cfg.dataloader.batch_size,
            canonical_obj_meshes=canonical_obj_meshes,
            canonical_obj_keypoints=canonical_obj_keypoints,
            device=self.model.device
        )
        self.model.set_mesh_model(self.mesh_model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        # Set contact model
        contact_model_type = self.cfg.model_conditioning.use_contacts
        self.contact_model = ContactModel(contact_model_type, device=self.model.device)
        if contact_model_type == "encoder_decimated_clip":
            weights_path = Path(cfg.env.assets_folder) / f"{cfg.model_conditioning.contact_model}.pth"
            self.contacts_enc_dec = "model_enc"
        elif contact_model_type == "NONE":
            weights_path = ""
            self.contacts_enc_dec = ""
        else:
            raise ValueError(f"Unsupported contact conditioning {contact_model_type}")
        self.contact_model.set_contact_model(
            contact_model_type, self.contacts_enc_dec, weights_path
        )
        self.model.set_contact_model(self.contact_model)

    def get_outputs(self, batch):
        # get gt sbj vertices and joints
        with torch.no_grad():
            gt_sbj_vertices, gt_sbj_joints = self.mesh_model.get_smpl_th(batch)
            batch.sbj_vertices = gt_sbj_vertices
            batch.sbj_joints = gt_sbj_joints

            # get gt contacts
            sbj_contacts, sbj_contacts_full, captions = self.contact_model.encode_contacts(# 使用采样的接触信息训练神经网络
                batch.sbj_vertices, batch.sbj_contact_indexes, batch.obj_keypoints,
                batch.obj_class, batch.obj_R,
                enc_type=self.contacts_enc_dec, obj_name=batch.obj
            )
            batch.sbj_contacts = sbj_contacts
            batch.sbj_contacts_full = sbj_contacts_full

        # aux_output is (x_0, x_t, noise, x_0_pred)
        denoise_loss, aux_output = self.model(batch, 'train', return_intermediate_steps=True)

        # aux output is (x_0, x_t, noise, x_0_pred)
        output = self.model.split_output(aux_output[3], aux_output)
        sbj_vertices, obj_keypoints, sbj_joints = self.mesh_model.get_meshes_wkpts_th(
            output, batch.obj_class, batch.scale, return_joints=True
        )
        output.sbj_vertices = sbj_vertices
        output.obj_keypoints = obj_keypoints
        output.sbj_joints = sbj_joints

        return denoise_loss, output

    def compute_loss(self, batch: BatchData, output: TriDiModelOutput, denoise_loss):
        wandb_log = dict()
        loss = 0
        THR = self.cfg.train.loss_t_stamp_threshold

        for key, weight in self.cfg.train.losses.items():
            if key in {"smpl_v2v", "obj_v2v", "sbj_contacts"}:
                times = output.timesteps_obj if key == "obj_v2v" else output.timesteps_sbj
                if not (times <= THR).any():
                    wandb_log[f'loss/{key}'] = 0.0
                    continue
            if key == "smpl_v2v":
                mask = (output.timesteps_sbj <= THR)

                gt_sbj_vertices = batch.sbj_vertices.to(output.sbj_vertices.device)
                pred_sbj_vertices = output.sbj_vertices

                # filter based on mask
                gt_sbj_vertices = gt_sbj_vertices[mask]
                pred_sbj_vertices = pred_sbj_vertices[mask]

                loss_i = F.mse_loss(pred_sbj_vertices, gt_sbj_vertices, reduction='none')
            elif key == "obj_v2v":
                mask = (output.timesteps_obj <= THR)

                gt_obj_keypoints = batch.obj_keypoints
                pred_obj_keypoints = output.obj_keypoints

                # filter based on mask
                pred_obj_keypoints = pred_obj_keypoints[mask]
                gt_obj_keypoints = gt_obj_keypoints[mask.cpu()].to(output.obj_keypoints.device)

                loss_i = F.mse_loss(pred_obj_keypoints, gt_obj_keypoints, reduction='none')
            elif key == "sbj_contacts":
                mask = torch.logical_and(output.timesteps_sbj <= THR, output.timesteps_sbj <= THR)

                pred_contact_vertices = output.sbj_vertices
                pred_obj_keypoints = output.obj_keypoints

                # filter based on mask
                pred_contact_vertices = pred_contact_vertices[mask]
                pred_obj_keypoints = pred_obj_keypoints[mask]

                sbj_contact_indexes = batch.sbj_contact_indexes[0].to(output.sbj_vertices.device)
                pred_contact_vertices = pred_contact_vertices[:, sbj_contact_indexes]

                pred_contacts = torch.cdist(
                    pred_contact_vertices,
                    pred_obj_keypoints
                )
                pred_contacts = pred_contacts.min(dim=-1).values
                gt_contacts = batch.sbj_contacts_full[mask.cpu()].to(output.sbj_vertices.device)
                loss_i = F.mse_loss(pred_contacts, gt_contacts, reduction='none')
            elif key.startswith("denoise") or key.startswith("manifold_"):
                loss_i = denoise_loss[key]
            else:
                raise NotImplementedError(f"No implementation for {key} loss.")

            loss_i = loss_i.mean()
            loss += weight * loss_i
            wandb_log[f'loss/{key}'] = loss_i.detach().cpu().item()

        wandb_log["loss/total"] = loss.detach().cpu().item()
        return loss, wandb_log

    def train_step(self, batch):
        self.model.train()
        denoise_loss, output = self.get_outputs(batch)
        loss, wandb_log = self.compute_loss(batch, output, denoise_loss)

        # backward
        loss.backward()
        if self.cfg.optimizer.clip_grad_norm is not None:
            grad_norm_unclipped = compute_grad_norm(self.model.parameters())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optimizer.clip_grad_norm)
            wandb_log['grad_norm'] = grad_norm_unclipped
            wandb_log['grad_norm_clipped'] = compute_grad_norm(self.model.parameters())

        # optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.train_state.step += 1

        return wandb_log

    def save_checkpoint(self):
        checkpoint_dict = {
            'model': (self.model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.train_state.epoch,
            'step': self.train_state.step,
            'relative_step': self.train_state.step - self.train_state.initial_step,
            'cfg': self.cfg
        }

        checkpoint_name = f'checkpoint-step-{self.train_state.step:07d}.pth'
        checkpoint_path = Path(self.cfg.run.path) / 'checkpoints' / checkpoint_name
        torch.save(checkpoint_dict, checkpoint_path)
        logger.info(f'Saved checkpoint to {checkpoint_path}')

    def train(self):  # 控制整个训练过程：重复训练和验证，直到达到最大训练步数。
        total_batch_size = self.cfg.dataloader.batch_size  # 记录当前配置的单批样本数，用于启动日志。

        # Log info
        logger.info(  # 将下面拼接的多行信息写入日志。
            f'***** Starting training *****\n'  # 日志标题；\n 表示换行。
            f'    Number of classes: {len(self.mesh_model.canonical_obj_keypoints.keys()):_}\n'  # 物体类别数。
            f'    Dataset train size: {len(self.dataloader_train.dataset):_}\n'  # 训练集样本数。
            f'    Dataset val size: {len(self.dataloader_val.dataset):_}\n'  # 验证集样本数。
            f'    Dataloader train size: {len(self.dataloader_train):_}\n'  # 每轮训练包含的批次数。
            f'    Dataloader val size: {len(self.dataloader_val):_}\n'  # 每轮验证包含的批次数。
            f'    Batch size per device = {self.cfg.dataloader.batch_size}\n'  # 当前设备每批读取的样本数。
            f'    Total train batch size (w. parallel, dist & accum) = {total_batch_size}\n'  # 这里显示的仍是上面的 batch_size。
            f'    Gradient Accumulation steps = {self.cfg.optimizer.gradient_accumulation_steps}\n'  # 配置中的梯度累积步数。
            f'    Max training steps = {self.cfg.train.max_steps}\n'  # 训练计划执行的最大优化步数。
            f'    Training state = {self.train_state}'  # 当前轮数、步数等训练状态。
        )  # 完成启动日志的写入。

        while True:  # 持续进行训练轮次；达到 max_steps 时由下面的 return 退出。
            self.model.train()  # 将模型切换为训练模式，启用训练时的层行为。
            print(enumerate(self.dataloader_train))  # 打印数据加载器的枚举对象，用于调试；不会遍历批次。
            for i, batch in enumerate(self.dataloader_train):  # 依次读取训练批次，i 从 0 开始。
                # Early stop
                if (self.cfg.train.limit_train_batches is not None) and (i >= self.cfg.train.limit_train_batches):  # 如果设置了每轮训练批次上限，只处理前 limit_train_batches 批。
                    break  # 本轮训练提前结束，接着进入验证阶段。

                # Process one training step
                wandb_log = self.train_step(batch)  # 获取这一训练步的损失和梯度等日志数据。

                # Check if loss was None
                if not math.isfinite(wandb_log["loss/total"]):  # 检查总损失是否为有限数值。
                    logger.error("Loss is {}, stopping training".format(wandb_log["loss/total"]))  # 记录异常损失。
                    sys.exit(1)  # 以错误状态终止程序。

                # Log to wandb
                # 启用 wandb 且当前步达到日志间隔时，上传训练指标。
                if self.cfg.logging.wandb and \
                        self.train_state.step % self.cfg.train.log_step_freq == 0:  # 检查当前步数是否达到训练日志间隔。
                    wandb_log['lr'] = self.optimizer.param_groups[0]['lr']  # 记录第一个参数组的当前学习率。
                    wandb_log['step'] = self.train_state.step  # 记录从训练开始累计的步数。
                    wandb_log['relative_step'] = self.train_state.step - self.train_state.initial_step  # 记录本次启动后执行的步数。
                    wandb.log(wandb_log, step=self.train_state.step)  # 按当前训练步上传日志。

                # Save checkpoint
                if (self.train_state.step % self.cfg.train.checkpoint_freq == 0):  # 每经过 checkpoint_freq 个训练步，保存一次模型与训练状态。
                    self.save_checkpoint()  # 将模型、优化器、调度器等写入检查点文件。

                # Exit training
                if self.train_state.step >= self.cfg.train.max_steps:  # 达到设定的最大训练步数后，结束整个训练函数。
                    logger.info(f'Ending training at with state: {self.train_state}')  # 记录最终训练状态。

                    wandb.finish()  # 结束 wandb 日志会话。
                    time.sleep(5)  # 在退出前等待 5 秒。
                    return  # 退出 train，不再执行本轮后续验证。
            print("VAL")  # 在终端提示开始验证。
            self.model.eval()  # 将模型切换为评估模式，关闭训练时的层行为。
            val_log, val_metrics, val_counters = defaultdict(float), defaultdict(float), defaultdict(int)  # 初始化验证结果的累加容器。
            for i, batch in enumerate(self.dataloader_val):  # 依次处理验证集的每个批次。
                val_losses, tmp_metrics, tmp_counters = self.val_step(batch)  # 计算这一批的损失、评估指标及其计数。

                for k, v in val_losses.items():  # 遍历这一批的各项损失。
                    val_log[k] += v  # 按指标名累计损失。
                for k in tmp_metrics.keys():  # 遍历这一批计算出的评估指标。
                    val_metrics[k] += tmp_metrics[k]  # 累加该指标的数值总和。
                    val_counters[k] += tmp_counters[k]  # 累加该指标对应的样本计数。

            val_log = {f"VAL_{k}": v / len(self.dataloader_val) for k, v in val_log.items()}  # 按验证批次数求各项平均损失。
            val_log["epoch"] = self.train_state.epoch  # 给验证日志附上当前轮数。
            for k in val_metrics.keys():  # 逐项计算验证指标的平均值。
                val_log[f"VAL_{k}"] = val_metrics[k] / (val_counters[k] + 1e-4)  # 用计数归一化；小常数避免除以零。

            if self.cfg.logging.wandb:  # 仅在启用 wandb 时上传验证结果。
                wandb.log(val_log, step=self.train_state.step)  # 将本轮验证结果关联到当前训练步。

            # Epoch complete
            self.train_state.epoch += 1  # 将训练轮数加一。

    @torch.no_grad()
    def compute_val_metrics(self, batch, output):
        METRICS = ["MPJPE", "MPJPE_PA", "OBJ_V2V", "OBJ_CENTER_DISTANCE"]
        tmp_metrics = {metric: 0.0 for metric in METRICS}
        tmp_counters = {metric: 0 for metric in METRICS}

        for i in range(batch.batch_size()):
            mpjpe = get_mpjpe(
                output.sbj_joints[i].detach().cpu().numpy(), batch.sbj_joints[i].detach().cpu().numpy()
            )
            tmp_metrics["MPJPE"] += mpjpe
            tmp_counters["MPJPE"] += 1
            mpjpe_pa = get_mpjpe_pa(
                output.sbj_joints[i].detach().cpu().numpy(), batch.sbj_joints[i].detach().cpu().numpy()
            )
            tmp_metrics["MPJPE_PA"] += mpjpe_pa
            tmp_counters["MPJPE_PA"] += 1
        obj_v2v = get_obj_v2v(
            output.obj_keypoints.detach().cpu().numpy(),
            batch.obj_keypoints.detach().cpu().numpy()
        )
        tmp_metrics["OBJ_V2V"] += np.sum(obj_v2v)
        tmp_counters["OBJ_V2V"] += len(obj_v2v)
        obj_center_distance = get_obj_center_distance(
            output.obj_keypoints.detach().cpu().numpy(),
            batch.obj_keypoints.detach().cpu().numpy()
        )
        tmp_metrics["OBJ_CENTER_DISTANCE"] += np.sum(obj_center_distance)
        tmp_counters["OBJ_CENTER_DISTANCE"] += len(obj_center_distance)

        return tmp_metrics, tmp_counters

    @torch.no_grad()
    def val_step(self, batch):
        denoise_loss, output = self.get_outputs(batch)
        _, wandb_log = self.compute_loss(batch, output, denoise_loss)
        val_metrics, val_counters = self.compute_val_metrics(batch, output)

        return wandb_log, val_metrics, val_counters
