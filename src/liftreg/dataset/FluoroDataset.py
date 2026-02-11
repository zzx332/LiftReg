from __future__ import division, print_function

import os
from multiprocessing import *
from pathlib import Path

import blosc
import numpy as np
import progressbar as pb
import SimpleITK as sitk
import torch
import torchio
from torch.utils.data import Dataset
from torchio import LabelMap, ScalarImage, Subject
from diffdrr.pose import RigidTransform
from torchio.transforms import Resample
blosc.set_nthreads(1)

class FluoroDataset(Dataset):
    """fluoro dataset."""

    def __init__(self, data_path, phase=None, transform=None, option=None):
        """
        the dataloader for registration task, to avoid frequent disk communication, all pairs are compressed into memory
        :param data_path:  string, path to the data
            the data should be preprocessed and saved into txt
        :param phase:  string, 'train'/'val'/ 'test'/ 'debug' ,    debug here means a subset of train data, to check if model is overfitting
        :param transform: function,  apply transform on data
        : seg_option: pars,  settings for segmentation task,  None for segmentation task
        : reg_option:  pars, settings for registration task, None for registration task

        """
        self.data_path = data_path + "/img"
        self.drr_path = data_path + f"/drr"
        self.phase = phase
        self.transform = transform
        ind = ['train', 'val', 'test', 'debug'].index(phase)
        max_num_for_loading=option['max_num_for_loading',(-1,-1,-1,-1),"the max number of pairs to be loaded, set -1 if there is no constraint,[max_train, max_val, max_test, max_debug]"]
        self.max_num_for_loading = max_num_for_loading[ind]
        """ the max number of pairs to be loaded into the memory,[max_train, max_val, max_test, max_debug]"""
        self.has_label = option['use_segmentation_map', False, 'indicates whether to load segmentation map from dataset.']
        self.spacing = option['spacing_to_refer', (1,1,1)]
        self.load_projection_interval = option['load_projection_interval', 2]
        self.apply_hu_clip = option['apply_hu_clip', False]

        self.get_identifier_list()
        print(self.identifier_list)
        self.reg_option = option
        load_training_data_into_memory = option[('load_training_data_into_memory',False,"when train network, load all training sample into memory can relieve disk burden")]
        self.load_into_memory = load_training_data_into_memory if phase == 'train' else False
        self.pair_list = []
        self.proj_list = []
        self.spacing_list = []
        self.target_poses_list = []
        self.init_img_pool()

    def get_identifier_list(self):
        """
        get the all files belonging to data_type from the data_path,
        :return: full file path list, file name list
        """
        self.identifier_list = os.listdir(self.drr_path)
        self.identifier_list = [i[:-7] for i in os.listdir(self.drr_path) if i.endswith(".nii.gz")]

        if self.max_num_for_loading > 0:
            read_num = min(self.max_num_for_loading, len(self.identifier_list))
            self.identifier_list = self.identifier_list[:read_num]

    def transform_hu_to_density(self, volume, bone_attenuation_multiplier):
        # volume can be loaded as int16, need to convert to float32 to use float bone_attenuation_multiplier
        volume = volume.to(torch.float32)
        air = torch.where(volume <= -800)
        soft_tissue = torch.where((-800 < volume) & (volume <= 350))
        bone = torch.where(350 < volume)

        density = torch.empty_like(volume)
        density[air] = volume[soft_tissue].min()
        density[soft_tissue] = volume[soft_tissue]
        density[bone] = volume[bone] * bone_attenuation_multiplier
        density -= density.min()
        density /= density.max()
        return density
    def _resample_image(self, image, new_size, new_spacing, is_label=False):
        """
        重采样图像到指定大小和间距
        
        :param image: SimpleITK Image对象
        :param new_size: 新的图像大小 [x, y, z]
        :param new_spacing: 新的体素间距 [x, y, z]
        :param is_label: 是否为标签图像（使用最近邻插值）
        :return: 重采样后的SimpleITK Image对象
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        new_size_array = np.array(new_size)
        new_spacing_array = np.array(new_spacing)
        direction = np.array(image.GetDirection()).reshape(3, 3)
        
        # 计算新图像的半尺寸（物理空间）
        half_size = new_spacing_array * (new_size_array - 1) / 2.0
        
        # 新原点 = -方向矩阵 * 半尺寸
        # 这使得图像中心位于 (0, 0, 0)
        new_origin = -(direction.dot(half_size))
        resampler.SetOutputOrigin(new_origin.tolist())
        resampler.SetOutputDirection(image.GetDirection())
        
        # 标签使用最近邻插值，图像使用线性插值
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(-1000)
        # 2) 计算输入图像中心（物理坐标）
        dir_in = np.array(image.GetDirection()).reshape(3,3)
        org_in = np.array(image.GetOrigin(), dtype=float)
        sp_in  = np.array(image.GetSpacing(), dtype=float)
        sz_in  = np.array(image.GetSize(), dtype=float)
        # center_in = org_in + dir_in.dot(sp_in * (sz_in - 1) / 2.0)
        # 不确定要不要 - 1
        center_in = org_in + dir_in.dot(sp_in * (sz_in / 2.0))

        # 3) 关键：Transform 是 out->in，所以加一个 +center_in 的平移
        T = sitk.TranslationTransform(3)
        T.SetOffset(center_in.tolist())
        resampler.SetTransform(T)
       
        return resampler.Execute(image)

    def _read_case(self, identifier_list, img_label_dic):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(identifier_list)).start()
        count = 0
        for identifier in identifier_list:
            img_label_np = {}
            # Pay attention here. Flip is used to transform SAR orientation to SPR orientation.
            # This is only applied to synthetic dataset because the real data has already been transformed in preprocessing script.
            # TODO: Should make such change in preprocess script.
            # source_img = np.flip(np.load(os.path.join(self.data_path, identifier.split("_")[0] + "_source.nii.gz")).astype(np.float32), axis=(1))
            source_img = sitk.ReadImage(os.path.join(self.data_path, identifier.split("_")[0] + "_source.nii.gz"))
            source_img = self._resample_image(source_img, [160, 160, 160], [2.2, 2.2, 2.2], is_label=False)
            source_arr = sitk.GetArrayFromImage(source_img)
            # not sure if this is correct
            # source_img = np.flip(source_img, axis=(1))
            source_arr = source_arr.astype(np.float32)
            if self.apply_hu_clip:
                source_arr = self._normalize_intensity(source_arr, linear_clip=True, clip_range=[-100, 0])
            else:
                source_arr = self._normalize_intensity(source_arr, linear_clip=True)
            if self.has_label:
                source_seg = sitk.ReadImage(os.path.join(self.data_path, identifier.split("_")[0] + "_source_seg.nii.gz"))
                source_seg = self._resample_image(source_seg, [160, 160, 160], [2.2, 2.2, 2.2], is_label=True)
                source_seg = sitk.GetArrayFromImage(source_seg)                
                # not sure if this is correct
                # source_seg = np.flip(source_seg, axis=(1))
                source_seg = source_seg.astype(np.float32)
                img_label_np['source_seg'] = blosc.pack_array(source_seg)
            else:
                img_label_np['source_seg'] = None
            img_label_np['source'] = blosc.pack_array(source_arr)

            target_proj = sitk.ReadImage(os.path.join(self.drr_path, identifier+".nii.gz"))
            target_proj = sitk.GetArrayFromImage(target_proj)
            target_proj = target_proj.astype(np.float32)
            target_proj = self._normalize_intensity(target_proj, linear_clip=True, clip_range=(0,6))[::self.load_projection_interval]
            img_label_np['target_proj'] = blosc.pack_array(target_proj)

            # Load geo info
            img_label_np['target_poses'] , *_ = torch.load(os.path.join(self.drr_path,  identifier.split("_")[0] + "_pose.pt"), weights_only=False)["pose"][::self.load_projection_interval]
            img_label_np['target_poses'] = self._convert_extrinsic_to_pose(img_label_np['target_poses'], source_img)
            img_label_np["spacing"] = np.array(self.spacing)
            
            img_label_dic[identifier] = img_label_np
            count += 1
            pbar.update(count)
        pbar.finish()

    def _read_case_polypose(self, identifier_list, img_label_dic, center_volume=True, resample_target=[160, 160, 160], orientation="PA"):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(identifier_list)).start()
        count = 0
        for identifier in identifier_list:
            img_label_np = {}
            # Pay attention here. Flip is used to transform SAR orientation to SPR orientation.
            # This is only applied to synthetic dataset because the real data has already been transformed in preprocessing script.
            # TODO: Should make such change in preprocess script.
            # source_img = np.flip(np.load(os.path.join(self.data_path, identifier.split("_")[0] + "_source.nii.gz")).astype(np.float32), axis=(1))
            source_img = ScalarImage(os.path.join(self.data_path, identifier.split("_")[0] + "_source.nii.gz"))
            if self.has_label:
                source_seg = ScalarImage(os.path.join(self.data_path, identifier.split("_")[0] + "_source_seg.nii.gz"))

            if center_volume:
                source_img = self.canonicalize(source_img)
                source_seg = self.canonicalize(source_seg)
                # Apply resample
            if resample_target is not None:
                transform = torchio.transforms.Compose([
                    torchio.transforms.Resample((2.2, 2.2, 2.2)),
                    torchio.transforms.CropOrPad((160, 160, 160)),
                ])

                source_img = transform(source_img)
                source_seg = transform(source_seg)
            source_arr = source_img.data.detach().cpu().numpy().astype(np.float32).squeeze(0)
            # not sure if this is correct
            # source_img = np.flip(source_img, axis=(1))
            # source_arr = source_arr.astype(np.float32)
            if self.apply_hu_clip:
                source_arr = self._normalize_intensity(source_arr, linear_clip=True, clip_range=[-100, 0])
            else:
                source_arr = self._normalize_intensity(source_arr, linear_clip=True)
            if self.has_label:
                source_seg = source_seg.data.detach().cpu().numpy().astype(np.float32).squeeze(0)
                img_label_np['source_seg'] = blosc.pack_array(source_seg)
            else:
                img_label_np['source_seg'] = None
            img_label_np['source'] = blosc.pack_array(source_arr)

            target_proj = sitk.ReadImage(os.path.join(self.drr_path, identifier+".nii.gz"))
            target_proj = sitk.GetArrayFromImage(target_proj)
            target_proj = target_proj.astype(np.float32)
            target_proj = self._normalize_intensity(target_proj)[::self.load_projection_interval]
                # Frame-of-reference change
            if orientation == "AP":
                # Rotates the C-arm about the x-axis by 90 degrees
                reorient = torch.tensor(
                    [
                        [1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=torch.float32,
                )
            elif orientation == "PA":
                # Rotates the C-arm about the x-axis by 90 degrees
                # Reverses the direction of the y-axis
                reorient = torch.tensor(
                    [
                        [1, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                    ],
                    dtype=torch.float32,
                )
            elif orientation is None:
                # Identity transform
                reorient = torch.tensor(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=torch.float32,
                )
            else:
                raise ValueError(f"Unrecognized orientation {orientation}")
            img_label_np['target_proj'] = blosc.pack_array(target_proj)
            # Load geo info
            img_label_np['target_poses'] , *_ = torch.load(os.path.join(self.drr_path,  identifier.split("_")[0] + "_pose.pt"), weights_only=False)["pose"][::self.load_projection_interval]
            img_label_np['target_poses'] = self._extrinsic_cam2world_to_SOUV(img_label_np['target_poses'], reorient, sdd=1020.0)
            img_label_np["spacing"] = np.array(self.spacing)
            # img_label_np['reorient'] = reorient
            
            img_label_dic[identifier] = img_label_np
            count += 1
            pbar.update(count)
        pbar.finish()

    def canonicalize(self, volume):
        isocenter = volume.get_center()
        Tinv = np.array(
            [
                [1.0, 0.0, 0.0, -isocenter[0]],
                [0.0, 1.0, 0.0, -isocenter[1]],
                [0.0, 0.0, 1.0, -isocenter[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        new_affine = Tinv.dot(volume.affine)
        volume_centered = ScalarImage(
            tensor=volume.data,      # 原始体数据
            affine=new_affine     # 新 affine
        )

        return volume_centered

    def init_img_pool(self):
        """img pool shoudl include following thing:
        img_label_path_dic:{img_name:{'img':img_fp,'label':label_fp,...}
        img_label_dic: {img_name:{'img':img_np,'label':label_np},......}
        pair_name_list:[[pair1_s,pair1_t],[pair2_s,pair2_t],....]
        pair_list [[s_np,t_np,sl_np,tl_np],....]
        only the pair_list need to be used by get_item method
        """
        manager = Manager()
        img_label_dic = manager.dict()
        # num_of_workers = 12
        num_of_workers = 1
        num_of_workers = num_of_workers if len(self.identifier_list )>num_of_workers else len(self.identifier_list )
        split_dict = self.__split_dict(self.identifier_list , num_of_workers)
        procs = []
        for i in range(num_of_workers):
            # p = Process(target=self._read_case, args=(split_dict[i], img_label_dic,))
            p = Process(target=self._read_case_polypose, args=(split_dict[i], img_label_dic))
            p.start()
            print("pid:{} start:".format(p.pid))

            procs.append(p)

        for p in procs:
            p.join()
        
        for case_name in self.identifier_list :
            case = img_label_dic[case_name]
            if self.has_label:
                # self.pair_list.append([case['source'], case['target'], case['source_seg'], case['target_seg']])
                self.pair_list.append([case['source'], case['source_seg']])
            else:
                # self.pair_list.append([case['source'], case['target']])
                self.pair_list.append([case['source']])
            # self.proj_list.append([case['target_proj'], case['source_proj']])
            self.proj_list.append([case['target_proj']])

            self.spacing_list.append(case['spacing'])
            self.target_poses_list.append(case["target_poses"])
        
        print("the loading phase {} finished, total {} img and labels have been loaded".format(self.phase, len(img_label_dic)))


    def _resize_img(self, img, new_size, is_label=False):
        """
        :param img: image in numpy
        :param new_size: D*H*W
        :return:
        """
        dim = len(new_size)
        new_img = None
        if dim == 2:
            new_img = torch.nn.functional.interpolate(torch.from_numpy(img).unsqueeze(0), 
                                                  new_size, mode='bilinear', align_corners=True)[0]
        elif dim == 3:
            new_img = torch.nn.functional.interpolate(torch.from_numpy(img).unsqueeze(0).unsqueeze(0), 
                                                  new_size, mode='trilinear', align_corners=True)[0,0]
        
        return new_img.numpy()
    def _convert_extrinsic_to_pose(self, extrinsic, source_img):
        """
        :param extrinsic matrix
        :param source_img: source image
        :return: pose
        """
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        C = -np.dot(R.T, t.T)
        C[1] = -C[1] + source_img.spatial_shape[1] * source_img.spacing[1] / 2
        return C.reshape(1, 3)

    def _extrinsic_cam2world_to_SOUV(self, extrinsic, reorient, sdd, device=None):
        """
        R: (...,3,3)  cam->world rotation
        t: (...,3)    cam->world translation (world coords of camera origin / source)
        returns S,O,U,V: (...,3)
        """
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        if device is None:
            device = R.device if torch.is_tensor(R) else "cpu"

        R = torch.as_tensor(R, dtype=torch.float32, device=device)
        t = torch.as_tensor(t, dtype=torch.float32, device=device)

        # camera-frame canonical definitions
        Sc = torch.tensor([0., 0., 0.], device=device)
        Oc = torch.tensor([0., 0., float(sdd)], device=device)
        Uc = torch.tensor([1., 0., 0.], device=device)
        Vc = torch.tensor([0., 1., 0.], device=device)

        # broadcast-friendly matmul
        S = (R @ reorient @ Sc) + t
        O = (R @ reorient @ Oc) + t
        U = (R @ reorient @ Uc)
        V = (R @ reorient @ Vc)

        return S.unsqueeze(0), O.unsqueeze(0), U.unsqueeze(0), V.unsqueeze(0)

    def _normalize_intensity(self, img, linear_clip=False, clip_range=None):
        """
        a numpy image, normalize into intensity [-1,1]
        (img-img.min())/(img.max() - img.min())
        :param img: image
        :param linear_clip:  Linearly normalized image intensities so that the 95-th percentile gets mapped to 0.95; 0 stays 0
        :return:
        """

        if linear_clip:
            if clip_range is not None:
                img[img<clip_range[0]] = clip_range[0]
                img[img>clip_range[1]] = clip_range[1]
                normalized_img = (img-clip_range[0]) / (clip_range[1] - clip_range[0]) 
            else:
                img = img - img.min()
                normalized_img =img / np.percentile(img, 95) * 0.95
        else:
            # If we normalize in HU range of softtissue
            min_intensity = img.min()
            max_intensity = img.max()
            normalized_img = (img-img.min())/(max_intensity - min_intensity)
        normalized_img = normalized_img*2 - 1
        return normalized_img

    def __read_and_clean_itk_info(self,path):
        if path is not None:
            img = sitk.ReadImage(path)
            spacing_sitk = img.GetSpacing()
            img_sz_sitk = img.GetSize()
            return sitk.GetImageFromArray(sitk.GetArrayFromImage(img)), np.flipud(spacing_sitk), np.flipud(img_sz_sitk)
        else:
            return None, None, None

    def __split_dict(self, dict_to_split, split_num):
        index_list = list(range(len(dict_to_split)))
        index_split = np.array_split(np.array(index_list), split_num)
        split_dict = []
        for i in range(split_num):
            dj = dict_to_split[index_split[i][0]:index_split[i][-1]+1]
            split_dict.append(dj)
        return split_dict

    def __len__(self):
        return len(self.identifier_list )

    def __getitem__(self, idx):
        """
        :param idx: id of the items
        :return: the processed data, return as type of dic

        """
        idx = idx % len(self.identifier_list )

        filename = self.identifier_list [idx]
        zipnp_pair_list = self.pair_list[idx]
        zipnp_proj_list = self.proj_list[idx]
        pair_list = [blosc.unpack_array(item) for item in zipnp_pair_list]
        # pair_list = [item.detach().cpu().numpy().astype(np.float32) for item in zipnp_pair_list]
        proj_list = [blosc.unpack_array(item) for item in zipnp_proj_list]
        # proj_list = [item.detach().cpu().numpy().astype(np.float32) for item in zipnp_proj_list]

        # sample = {'source': np.expand_dims(pair_list[0], axis=0),
                #   'target': np.expand_dims(pair_list[1], axis=0)}
        sample = {'source': np.expand_dims(pair_list[0], axis=0)}

        if self.has_label:
            # sample["source_label"] = np.expand_dims(pair_list[2], axis=0)
            # sample["target_label"] = np.expand_dims(pair_list[3], axis=0)
            sample["source_label"] = np.expand_dims(pair_list[1], axis=0)
        sample["target_proj"] = np.asarray(proj_list[0]).astype(np.float32)
        # sample["source_proj"] = np.asarray(proj_list[1]).astype(np.float32)

        if self.transform:
            sample['source'] = self.transform(sample['source'])
            # sample['target'] = self.transform(sample['target'])
            if self.has_label:
                sample['source_label'] = self.transform(sample['source_label'])
                # sample['target_label'] = self.transform(sample['target_label'])
            sample['target_proj'] = self.transform(sample['target_proj'])
            # sample['source_proj'] = self.transform(sample['source_proj'])
        sample['target_poses'] = self.target_poses_list[idx]
        sample['spacing'] = self.spacing_list[idx].copy()
        return sample, filename


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        n_tensor = torch.from_numpy(sample)
        return n_tensor
