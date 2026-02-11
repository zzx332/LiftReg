import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class regUNetTrainer:
    """自定义训练器"""
    
    def __init__(self, setting, img_size=(160, 160, 160),num_projections=1, device='cuda'):
        self.img_size = img_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        self.output_path = f"./exp_custom/{timestamp}"
        make_dir(self.output_path)
        make_dir(os.path.join(self.output_path, "checkpoints"))
        train_setting = setting['train']
        dataset_setting = setting['dataset']
        self.mode = train_setting[('mode', "train", '\'train\' or \'test\'')]

        # Init dataset and dataloader
        data_path = dataset_setting["data_path"]
        batch_size = train_setting["dataloader"]["batch_size"]
        shuffle = train_setting["dataloader"]["shuffle"]
        workers = train_setting["dataloader"]["workers"]
        
        dataset_class = get_class(dataset_setting["dataset_class"])
        if self.mode == "train":
            self.dataset = {'train': dataset_class(data_path, phase="train", 
                                                option=dataset_setting),
                            'val': dataset_class(data_path, phase='val',
                                                option=dataset_setting),
                            'debug': dataset_class(data_path, phase='debug',
                                                option=dataset_setting)}
            self.dataloaders = {'train': DataLoader(self.dataset["train"],
                                                batch_size=batch_size,
                                                shuffle=shuffle[0],
                                                num_workers=workers[0]),
                            'val': DataLoader(self.dataset["val"],
                                              batch_size=batch_size,
                                              shuffle=shuffle[1],
                                              num_workers=workers[1]),
                            'debug': DataLoader(self.dataset["debug"],
                                               batch_size=batch_size,
                                               shuffle=shuffle[2],
                                               num_workers=workers[2])}
        elif self.mode == "test":
            self.dataset = {'test': dataset_class(data_path, phase="test",
                                                  option=dataset_setting)}
            self.dataloaders = {"test": DataLoader(self.dataset["test"],
                                                   batch_size=batch_size,
                                                   shuffle=shuffle[3],
                                                   num_workers=workers[3])}

        # 初始化模型
        self._init_model(num_projections)
        
        # 初始化损失函数
        self._init_loss()
        
        # 初始化优化器
        self._init_optimizer()
        
        # 数据生成器
        self.data_generator = DummyDataGenerator(
            img_size=img_size,
            proj_size=(160, 160),
            num_projections=num_projections,
            batch_size=2
        )
        
        print("初始化完成！")
    
    def _init_model(self, num_projections=1):
        """初始化模型"""
        print("初始化模型...")
        
        model_opt = {
            "drr_feature_num": num_projections,
            "latent_dim": 56,
            "pca_path": "D:/dataset/CTA_DSA/DeepFluoro/pca",
            "use_pca": False
        }
        
        self.model = LiftRegModel(self.img_size, model_opt)
        self.model = self.model.to(self.device)
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
    
    def _init_loss(self):
        """初始化损失函数"""
        print("初始化损失函数...")
        
      
        loss_opt = pars.ParameterDict()
        loss_opt['initial_reg_factor'] = 0.01
        loss_opt['min_reg_factor'] = 0.01
        loss_opt['reg_factor_decay_from'] = 2
        loss_opt['sim_class'] = 'liftreg.layers.losses.NCCLoss'
        
        self.loss_fn = SubspaceLoss(loss_opt)
    
    def _init_optimizer(self):
        """初始化优化器"""
        print("初始化优化器...")
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=0
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=0.8
        )
    
    def train_step(self, batch):
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        output = self.model(batch)
        output["epoch"] = self.current_epoch
        
        # 计算损失
        losses = self.loss_fn(output)
        
        # 反向传播
        losses["total_loss"].backward()
        self.optimizer.step()
        
        # 返回损失值
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                     for k, v in losses.items()}
        
        return loss_dict
    
    def val_step(self, batch):
        """验证步骤"""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(batch)
            output["epoch"] = self.current_epoch
            losses = self.loss_fn(output)
        
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                     for k, v in losses.items()}
        
        return loss_dict, output
    
    def train(self, num_epochs=10, steps_per_epoch=10, val_frequency=2):
        """训练循环"""
        print(f"\n开始训练 - {num_epochs} 个 epoch, 每个 epoch {steps_per_epoch} 步")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 训练阶段
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            epoch_losses = []
            
            for step in range(steps_per_epoch):
                # 生成虚拟数据
                batch = self.data_generator.generate_batch(device=self.device)
                
                # 训练步骤
                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict['total_loss'])
                
                if step % 5 == 0:
                    print(f"  Step [{step}/{steps_per_epoch}] - Loss: {loss_dict['total_loss']:.4f}")
            
            avg_loss = np.mean(epoch_losses)
            print(f"  平均训练损失: {avg_loss:.4f}")
            
            # 验证阶段
            if (epoch + 1) % val_frequency == 0:
                print("  执行验证...")
                batch = self.data_generator.generate_batch(device=self.device)
                val_loss_dict, output = self.val_step(batch)
                print(f"  验证损失: {val_loss_dict['total_loss']:.4f}")
                
                # 打印输出的形状信息
                print("  输出形状:")
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        print(f"    {key}: {value.shape}")
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  当前学习率: {current_lr:.6f}")
            
            # 保存检查点
            if (epoch + 1) % 5 == 0:
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
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, checkpoint_path)
        
        print(f"  保存检查点: {checkpoint_path}")
    
    def test_forward(self):
        """测试前向传播"""
        print("\n测试前向传播...")
        batch = self.data_generator.generate_batch(device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch)
        
        print("输入形状:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        print("\n输出形状:")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        print("\n前向传播测试通过！✓")
        return output
