import os
import sys
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from stat import S_IREAD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from liftreg.utils.general import make_dir, get_class
from liftreg.utils import module_parameters as pars
from liftreg.utils.utils import set_seed_for_demo

class ExportWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, source_proj, target_proj):
        out = self.net({
            "source_proj": source_proj,
            "target_proj": target_proj,
        })
        return out["phi"], out["warped_moving"], out["target_proj"]  # 固定tuple输出

class RegTrainer2D:
    def __init__(self, setting, device='cuda'):
        self.setting = setting
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        train_setting = setting['train']
        dataset_setting = setting['dataset']
        
        self.output_path = train_setting['output_path']
        self.log_path = os.path.join(self.output_path, "logs")
        self.epochs = train_setting['epoch', 1000]
        self.steps_per_epoch = int(train_setting['steps_per_epoch', 500])
        self.val_frequency = train_setting['val_frequency', 20]
        
        # Data
        dataset_class = get_class(dataset_setting["dataset_class"])
        data_path = dataset_setting["data_path"]
        val_data_path = dataset_setting["val_data_path"]
        batch_size = train_setting["dataloader"]["batch_size"]
        workers = train_setting["dataloader"]["workers"]
        
        self.train_dataset = dataset_class(data_path, phase="train", option=dataset_setting)
        self.val_dataset = dataset_class(val_data_path, phase="val", option=dataset_setting)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers[0])
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers[1])
        
        # Model
        self.model = get_class(train_setting['model_class'])(img_size=dataset_setting['img_after_resize']).to(self.device)
        self.loss = get_class(train_setting['loss_class'])(train_setting['loss']).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_setting['optim']['lr'])
        self.writer = SummaryWriter(self.log_path + "/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.start_epoch = 1

        # Continue training from checkpoint if requested
        self.continue_train = train_setting['continue_train', False]
        self.test_forward = train_setting['test_forward', False]
        self.continue_from = train_setting['continue_from', '']
        self.test_from = train_setting['test_from', '']
        if self.continue_train and len(self.continue_from) > 0:
            self._load_checkpoint(self.continue_from)
        if self.test_forward and len(self.test_from) > 0:
            self._load_checkpoint(self.test_from)

        # Multi-step registration at validation/inference only (train stays single-step)
        self.num_refine_steps = int(train_setting['num_refine_steps', 1])

    def set_input(self, batch):
        ret = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                ret[k] = v.to(self.device)
            else:
                ret[k] = v
        return ret

    def _compose_flow(self, phi_list):
        """Compose per-step displacement fields: phi_total = phi_0 + warp(phi_1, phi_0) + ..."""
        phi_total = phi_list[0]
        for phi_next in phi_list[1:]:
            phi_warped = self.model.transformer(phi_next, phi_total, mode='bilinear')
            phi_total = phi_total + phi_warped
        return phi_total

    def _forward_refine(self, batch):
        """Iterative inference: each step sees warp(original, phi_so_far); final output uses phi_total."""
        if self.num_refine_steps <= 1:
            return self.model(batch)

        original_source = batch['source_proj']
        current_source = original_source
        target = batch['target_proj']
        phi_list = []
        step_batch = dict(batch)
        output = None
        import time
        time_start = time.time()
        for step in range(self.num_refine_steps):
            step_batch['source_proj'] = current_source
            step_batch['target_proj'] = target
            output = self.model(step_batch)
            phi_list.append(output['phi'])
            if step < self.num_refine_steps - 1:
                phi_so_far = self._compose_flow(phi_list)
                current_source = self.model.transformer(original_source, phi_so_far)
        time_end = time.time()
        print(f"Time taken for {self.num_refine_steps}-step refinement: {time_end - time_start:.2f} seconds")
        phi_total = self._compose_flow(phi_list)
        output = dict(output)
        output['phi'] = phi_total
        output['warped_moving'] = self.model.transformer(original_source, phi_total)
        if 'source_label' in batch:
            output['warped_label'] = self.model.transformer(
                batch['source_label'].float(), phi_total, mode='nearest'
            )
        return output

    def export_pt(self, output_path):
        self.model.eval().to(self.device)
        wrapper = ExportWrapper(self.model).eval().to(self.device)
        src = torch.rand(1, 1, 160, 160, device=self.device)
        tgt = torch.rand(1, 1, 160, 160, device=self.device)
        traced = torch.jit.trace(wrapper, (src, tgt), strict=True)
        traced.save(os.path.join(output_path, "model.pt"))
        print("saved:", os.path.join(output_path, "model.pt"))
        print(D)
        return

    def train(self):
        print(f"Starting 2D Training for {self.epochs} epochs.")
        train_iter = iter(self.train_loader)
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            sim_loss_epoch = 0.0
            reg_loss_epoch = 0.0

            pbar = tqdm(
                range(self.steps_per_epoch),
                desc=f"Epoch {epoch}/{self.epochs}",
            )
            for _ in pbar:
                try:
                    batch, _ = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch, _ = next(train_iter)

                batch = self.set_input(batch)
                self.optimizer.zero_grad()

                output = self.model(batch)
                losses = self.loss(output)

                loss_val = losses['total_loss']
                loss_val.backward()
                self.optimizer.step()

                epoch_loss += loss_val.item()
                sim_loss_epoch += losses['sim_loss'].item()
                reg_loss_epoch += losses['reg_loss'].item()

                pbar.set_postfix({'loss': loss_val.item()})
            print(f"max displacement: {output['phi'].max().item():.4f}")
            print(f"min displacement: {output['phi'].min().item():.4f}")
            n_steps = self.steps_per_epoch
            avg_loss = epoch_loss / n_steps
            avg_sim = sim_loss_epoch / n_steps
            avg_reg = reg_loss_epoch / n_steps

            self.writer.add_scalar("Train/Total_Loss", avg_loss, epoch)
            self.writer.add_scalar("Train/Sim_Loss", avg_sim, epoch)
            self.writer.add_scalar("Train/Reg_Loss", avg_reg, epoch)
            
            print(f"Epoch {epoch} | Total Loss: {avg_loss:.4f} | Sim: {avg_sim:.4f} | Reg: {avg_reg:.4f}")
            
            if epoch % self.val_frequency == 0:
                self.validate(epoch)
                
            if epoch % self.setting['train']['save_model_frequency', 50] == 0:
                save_path = os.path.join(self.output_path, "checkpoints", f"model_{epoch}.pth")
                make_dir(os.path.dirname(save_path))
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, save_path)
                print(f"Checkpoint saved to {save_path}")

    def validate(self, epoch, save_path=None):
        self.model.eval()
        if self.num_refine_steps > 1:
            print(f"Validation with {self.num_refine_steps}-step refinement")
        val_loss = 0.0
        with torch.no_grad():
            for batch, identifier in self.val_loader:
                batch = self.set_input(batch)
                output = self._forward_refine(batch)
                if save_path is not None:
                    self.save_output(batch, output, save_path, identifier)
                    continue
                losses = self.loss(output)
                val_loss += losses['total_loss'].item()
        avg_val_loss = val_loss / len(self.val_loader)
        self.writer.add_scalar("Val/Total_Loss", avg_val_loss, epoch)
        print(f"Validation Loss: {avg_val_loss:.4f}")
    
    def save_output(self, input, output, save_path, identifier):
        make_dir(save_path)
        import SimpleITK as sitk
        def save_tensor_arr(tensor_arr, path):
            sitk.WriteImage(sitk.GetImageFromArray(tensor_arr.detach().cpu().numpy()), path)
        for i, case in enumerate(identifier):
            flow = output['phi']
            save_tensor_arr(input['source_proj'][i], os.path.join(save_path, f"{case}_source.nii.gz"))
            save_tensor_arr(output['target_proj'][i], os.path.join(save_path, f"{case}_fixed.nii.gz"))
            save_tensor_arr(output['warped_moving'][i], os.path.join(save_path, f"{case}_warped.nii.gz"))
            if 'source_label' in input:
                save_tensor_arr(input['source_label'][i], os.path.join(save_path, f"{case}_source_seg.nii.gz"))
                if 'warped_label' in output:
                    save_tensor_arr(output['warped_label'][i], os.path.join(save_path, f"{case}_warped_seg.nii.gz"))
                else:
                    source_seg = input['source_label'][i].unsqueeze(0).float()
                    warped_seg = self.model.transformer(source_seg, flow[i:i + 1], mode="nearest")
                    save_tensor_arr(warped_seg[0], os.path.join(save_path, f"{case}_warped_seg.nii.gz"))

    def _load_checkpoint(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # Compatible with both old (state_dict-only) and new checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'epoch' in checkpoint:
                self.start_epoch = int(checkpoint['epoch']) + 1
        else:
            self.model.load_state_dict(checkpoint, strict=False)
            self.start_epoch = 1
        print(f"Resumed from checkpoint: {ckpt_path}, start_epoch={self.start_epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting_path', required=True, type=str, help='Path to deepfluoro_task_setting_2d.json')
    parser.add_argument('-e', '--exp_name', required=False, type=str, default="exp_2d")
    parser.add_argument('--continue_from', required=False, type=str, default=None, help='Path to checkpoint for continue training')
    parser.add_argument('--save_path', required=False, type=str, default=None, help='Path to save the output')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--test_from',required=False, type=str,
                        help='The path to the checkpoint for testing')
    args = parser.parse_args()

    set_seed_for_demo()
    
    setting = pars.ParameterDict()
    setting.load_JSON(args.setting_path)
    if args.test_from is not None:
        setting["train"]["test_forward"] = True
        setting["train"]["test_from"] = args.test_from
    if args.continue_from is not None:
        setting["train"]["continue_train"] = True
        setting["train"]["continue_from"] = args.continue_from

    # Setup directories
    exp_path = os.path.join(setting["train"]["output_path"], args.exp_name, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    setting["train"]["output_path"] = exp_path
    make_dir(exp_path)
    make_dir(os.path.join(exp_path, "logs"))
    make_dir(os.path.join(exp_path, "checkpoints"))
    task_output_path = os.path.join(exp_path, 'cur_task_setting.json')
    setting.write_ext_JSON(task_output_path)
    # Make the setting file read-only
    os.chmod(task_output_path, S_IREAD)

    trainer = RegTrainer2D(setting=setting)
    # trainer.export_pt(exp_path)
    # 测试前向传播
    if args.test:
        _ = trainer.validate(0, save_path = args.save_path)
        del _
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        trainer.train()
