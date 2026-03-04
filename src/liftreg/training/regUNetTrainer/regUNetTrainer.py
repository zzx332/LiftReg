import os
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

from liftreg.models.LiftRegDeformSubspaceBackproj_unet import model 
from liftreg.losses.SubspaceLoss import loss as SubspaceLoss
from liftreg.utils.general import make_dir, get_class
from liftreg.utils import module_parameters as pars
from liftreg.utils.net_utils import resume_train, save_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

class DummyDataGenerator:
    """虚拟数据生成器"""
    
    def __init__(self, img_size=(160, 160, 160), proj_size=(160, 160), 
                 num_projections=40, batch_size=2):
        self.img_size = img_size
        self.proj_size = proj_size
        self.num_projections = num_projections
        self.batch_size = batch_size
        
        # 生成虚拟的 PCA 参数（如果不存在）
        self._prepare_pca_data()
    
    def _prepare_pca_data(self):
        """准备 PCA 数据（如果不存在则创建虚拟数据）"""
        pca_path = "D:/dataset/CTA_DSA/DeepFluoro/pca"
        if not os.path.exists(pca_path):
            os.makedirs(pca_path)
        
        # 如果PCA文件不存在，创建虚拟的PCA参数
        pca_vectors_path = os.path.join(pca_path, "pca_vectors.npy")
        pca_mean_path = os.path.join(pca_path, "pca_mean.npy")
        
        if not os.path.exists(pca_vectors_path):
            print("创建虚拟 PCA vectors...")
            # latent_dim=56, output_dim=3*160*160*160
            D, W, H = self.img_size
            latent_dim = 56
            pca_vectors = np.random.randn(3*D*W*H, latent_dim).astype(np.float32) * 0.01
            np.save(pca_vectors_path, pca_vectors)
            print(f"保存到: {pca_vectors_path}")
        
        if not os.path.exists(pca_mean_path):
            print("创建虚拟 PCA mean...")
            D, W, H = self.img_size
            pca_mean = np.zeros(3*D*W*H, dtype=np.float32)
            np.save(pca_mean_path, pca_mean)
            print(f"保存到: {pca_mean_path}")
    
    def generate_batch(self, device='cuda'):
        """生成一个批次的虚拟数据"""
        B = self.batch_size
        D, W, H = self.img_size
        proj_w, proj_h = self.proj_size
        proj_num = self.num_projections
        
        # 生成随机的3D图像 (归一化到 [-1, 1])
        source = torch.randn(B, 1, D, W, H, device=device) * 0.5
        target = source + torch.randn(B, 1, D, W, H, device=device) * 0.1  # 略有不同
        
        # 生成随机的投影图像
        target_proj = torch.randn(B, proj_num, proj_w, proj_h, device=device).abs() * 0.5
        
        # 生成相机位姿 (X, Y, Z 坐标)
        # Y 轴是光源到探测器的方向，通常为正值
        target_poses = torch.zeros(B, proj_num, 3, device=device)
        target_poses[:, :, 0] = torch.linspace(-50, 50, proj_num)  # X 方向旋转
        target_poses[:, :, 1] = 300.0  # Y 方向（光源距离）
        target_poses[:, :, 2] = torch.linspace(-30, 30, proj_num)  # Z 方向旋转
        
        # 可选：生成分割标签
        source_label = (torch.randn(B, 1, D, W, H, device=device) > 0.5).float()
        target_label = (torch.randn(B, 1, D, W, H, device=device) > 0.5).float()
        
        # 生成 spacing 信息
        spacing = np.array([2.2, 2.2, 2.2])
        
        batch = {
            'source': source,
            'target': target,
            'target_proj': target_proj,
            'target_poses': target_poses,
            'source_label': source_label,
            'target_label': target_label,
            'spacing': spacing
        }
        
        return batch

class regUNetTrainer:
    """自定义训练器"""
    
    def __init__(self, setting, img_size=(160, 160, 160),num_projections=1, device='cuda'):
        self.img_size = img_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.input_img_sz = setting['dataset'][('img_after_resize', None, "image size after resampling")]
        self.gpu_ids = setting['train'][('gpu_ids', 0, 'the gpu id used for network methods')]
        print(f"使用设备: {self.device}, GPU ID: {self.gpu_ids}")
        
        # 创建输出目录
        self.output_path = setting['train']['output_path']
        train_setting = setting['train']
        dataset_setting = setting['dataset']
        self.mode = train_setting[('mode', "train", '\'train\' or \'test\'')]
        self.log_path = os.path.join(self.output_path, "logs")
        self.epochs = train_setting[('epoch', 100, 'num of training epoch')]
        self.val_frequency = train_setting[('val_frequency', 10, 'How many epoch per one validation')]
        self.start_epoch = train_setting[('start_epoch', 0, 'start epoch')]
        # Init dataset and dataloader
        data_path = dataset_setting["data_path"]
        batch_size = train_setting["dataloader"]["batch_size"]
        shuffle = train_setting["dataloader"]["shuffle"]
        workers = train_setting["dataloader"]["workers"]
        
        dataset_class = get_class(dataset_setting["dataset_class"])
        if self.mode == "train":
            self.dataset = {'train': dataset_class(data_path, phase="train", 
                                                option=dataset_setting)}
                            # 'val': dataset_class(data_path, phase='val',
                            #                     option=dataset_setting),
                            # 'debug': dataset_class(data_path, phase='debug',
                            #                     option=dataset_setting)}
            self.dataloaders = {'train': DataLoader(self.dataset["train"],
                                                batch_size=batch_size,
                                                shuffle=shuffle[0],
                                                num_workers=workers[0])}
                            # 'val': DataLoader(self.dataset["val"],
                            #                   batch_size=batch_size,
                            #                   shuffle=shuffle[1],
                            #                   num_workers=workers[1]),
                            # 'debug': DataLoader(self.dataset["debug"],
                            #                    batch_size=batch_size,
                            #                    shuffle=shuffle[2],
                            #                    num_workers=workers[2])}
        elif self.mode == "test":
            self.dataset = {'test': dataset_class(data_path, phase="test",
                                                  option=dataset_setting)}
            self.dataloaders = {"test": DataLoader(self.dataset["test"],
                                                   batch_size=batch_size,
                                                   shuffle=shuffle[3],
                                                   num_workers=workers[3])}
        if self.mode == "train":
            self.writer = SummaryWriter(self.log_path + "/" +datetime.now().strftime("%Y%m%d-%H%M%S"), flush_secs=30)

        # Init model.
        self.model = get_class(train_setting['model_class'])(self.input_img_sz, setting["train"]["model"])
        self.model = self.model.to(self.device)
        self.lr_scheduler = None
        # Init loss.
        self.loss = get_class(train_setting['loss_class'])(setting["train"]["loss"])
        
        # 初始化优化器
        self._init_optim(setting["train"]["optim"])
                # Resume training if specified.
        if self.mode == 'train':
            self.continue_train = train_setting[('continue_train', False,
            "for network training method, continue training the model loaded from model_path")]
            continue_from = train_setting['continue_from']
            continue_train_lr = train_setting[('continue_train_lr', -1,
                                'Used when continue_train=True. The network \
                                will restore the lr from model_load_path if \
                                it is set to -1.')]
            if self.continue_train:
                self.start_epoch, self.global_step = resume_train(continue_from, self.model, self.optimizer, self.lr_scheduler)
                if continue_train_lr > 0:
                    self._update_learning_rate(continue_train_lr)
                    train_setting['optim']['lr'] = train_setting['optim']['lr'] if not self.continue_train else continue_train_lr
                    print("the learning rate has been changed into {} when resuming the training".format(continue_train_lr))
            else:
                self.start_epoch = 0
                self.global_step = {"train":0, "val":0, "debug":0, "test":0}
        elif self.mode == 'test':
            test_from = train_setting['test_from']
            self.start_epoch, self.global_step = resume_train(test_from, self.model, self.optimizer, self.lr_scheduler)
        
        self.cur_epoch = self.start_epoch
        # # 数据生成器
        # self.data_generator = DummyDataGenerator(
        #     img_size=img_size,
        #     proj_size=(160, 160),
        #     num_projections=num_projections,
        #     batch_size=2
        # )
        
        print("初始化完成！")
    
    
    def _init_optim(self, setting, warmming_up=False):
        """
        set optimizers and scheduler

        :param setting: settings on optimizer
        :param network: model with learnable parameters
        :param warmming_up: if set as warmming up
        :return: optimizer, custom scheduler, plateau scheduler
        """
        optimize_name = setting['optim_type']
        lr = setting['lr']
        beta = setting['adam']['beta']
        lr_sched_setting = setting[('lr_scheduler', {},
                            "settings for learning scheduler")]
        self.lr_sched_type = lr_sched_setting['type']

        if optimize_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5, betas=beta)
            # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, eps=1e-5, betas=beta)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.optimizer.zero_grad()

        if self.lr_sched_type == 'custom':
            step_size = lr_sched_setting['custom'][('step_size', 50,
                            "update the learning rate every # epoch")]
            gamma = lr_sched_setting['custom'][('gamma', 0.5,
                            "the factor for updateing the learning rate")]
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                step_size=step_size, gamma=gamma)
        elif self.lr_sched_type == 'plateau':
            patience = lr_sched_setting['plateau']['patience']
            factor = lr_sched_setting['plateau']['factor']
            threshold = lr_sched_setting['plateau']['threshold']
            min_lr = lr_sched_setting['plateau']['min_lr']
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer,
                                                    mode='max',
                                                    patience=patience,
                                                    factor=factor,
                                                    verbose=True,
                                                    threshold=threshold,
                                                    min_lr=min_lr,
                                                    cooldown=lr_sched_setting['plateau']['cooldown'])

        if not warmming_up:
            print(" no warming up the learning rate is {}".format(lr))
        else:
            lr = setting['lr']/10
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr_scheduler.base_lrs = [lr]
            print(" warming up on the learning rate is {}".format(lr))


    def set_input(self, input):
        """
        :param data:
        :param is_train:
        :return:
        """
        _, self.fname_list = input
        prepared_input = {}
        self.moving = input[0]['source']
        # self.target = input[0]['target']
        if 'source_label' in input[0]:
            self.l_moving = input[0]['source_label']
            # self.l_target = input[0]['target_label']
        else:
            self.l_moving = None
            # self.l_target = None

        if self.gpu_ids is not None and self.gpu_ids >= 0:
            for k,v in input[0].items():
                if isinstance(v, torch.Tensor) and len(v.shape) > 3:
                    prepared_input[k] = v.cuda()
                else:
                    prepared_input[k] = v
        else:
            for k,v in input[0].items():
                prepared_input[k] = v
            
        prepared_input['epoch'] = self.cur_epoch

        return prepared_input

    def train_step(self, batch):
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        if hasattr(self.model, 'set_cur_epoch'):
            self.model.set_cur_epoch(self.cur_epoch)
        # 前向传播
        output = self.model(batch)
        output["epoch"] = self.cur_epoch
        
        # 计算损失
        losses = self.loss(output)
        
        # 反向传播
        losses["total_loss"].backward()
        self.optimizer.step()
        
        # 返回损失值
        loss_dict = {k: v.detach().item() if isinstance(v, torch.Tensor) else v 
                     for k, v in losses.items()}
        
        return loss_dict, output['params']
    
    def val_step(self, batch):
        """验证步骤"""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(batch)
            output["epoch"] = self.cur_epoch
            losses = self.loss(output)
        
        loss_dict = {k: v.detach().item() if isinstance(v, torch.Tensor) else v 
                     for k, v in losses.items()}
        
        return loss_dict, output
    
    def train(self):
        """训练循环"""
        print(f"\n开始训练 - {self.epochs} 个 epoch")
        print("=" * 60)
        
        for epoch in tqdm(range(self.start_epoch, self.epochs+1)):
            self.cur_epoch = epoch
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], epoch)
            # 训练阶段
            print(f"\nEpoch [{epoch+1}/{self.epochs}]")
            epoch_losses = []
            reg_losses = []
            sim_losses = []
            
            for i, batch in enumerate(self.dataloaders['train']):
                global_step = epoch * len(self.dataloaders['train']) + i
                # 训练步骤
                loss_dict, disp_field = self.train_step(self.set_input(batch))
                epoch_losses.append(loss_dict['total_loss'])
                reg_losses.append(loss_dict['reg_loss'])
                sim_losses.append(loss_dict['sim_loss'])
                # for k,v in loss_dict.items():
                #     self.writer.add_scalar(f"Train/{k}", v, global_step)        

            
            avg_loss = np.mean(epoch_losses)
            avg_reg_loss = np.mean(reg_losses)
            avg_sim_loss = np.mean(sim_losses)
            print(f"  平均total loss: {avg_loss:.4f}")
            print(f"  平均reg loss: {avg_reg_loss:.4f}")
            print(f"  平均sim loss: {avg_sim_loss:.4f}")
            print(f"  max displacement: {disp_field.max():.4f}")
            print(f"  min displacement: {disp_field.min():.4f}")
            self.writer.add_scalar("Train/total_loss_epoch", avg_loss, epoch)
            self.writer.add_scalar("Train/reg_loss_epoch", avg_reg_loss, epoch)
            self.writer.add_scalar("Train/sim_loss_epoch", avg_sim_loss, epoch)
            # 验证阶段
            if (epoch + 1) % self.val_frequency == 0:
                print("  执行验证...")
                val_loss_dict, output = self.val_step(self.set_input(batch))
                print(f"  验证损失: {val_loss_dict['total_loss']:.4f}")
                
                # 打印输出的形状信息
                print("  输出形状:")
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        print(f"    {key}: {value.shape}")
            
            # 更新学习率
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  当前学习率: {current_lr:.6f}")
            
            # 保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
        
        print("\n训练完成！")
        print("=" * 60)
    
    def save_checkpoint(self, epoch):
        """保存检查点"""
        checkpoint_path = os.path.join(
            self.output_path, 
            "checkpoints", 
            f"model_epoch_{epoch+1}.pth"
        )
        
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
        }, checkpoint_path)
        
        print(f"  保存检查点: {checkpoint_path}")
    
    def test_forward(self):
        """测试前向传播"""
        print("\n测试前向传播...")
        for i in range(10):
            data = next(iter(self.dataloaders['train']))
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(self.set_input(data))
        
        print("输入形状:")
        for key, value in self.set_input(data).items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        print("\n输出形状:")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        print("\n前向传播测试通过！✓")
        return output
