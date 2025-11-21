import os
import pathlib

import nibabel as nib
import nibabel.processing
import numpy as np
import SimpleITK as sitk


def resample_one(input_file_path, output_file_path):
    # 1. 讀取nii.gz檔為影像
    image = sitk.ReadImage(input_file_path)
    # 2. 將影像的體素塊尺寸Resample為1x1x1並創建新的影像
    new_spacing = [1.0, 1.0, 1.0]  # 目標體素塊尺寸
    original_spacing = image.GetSpacing()  # 原始體素塊尺寸
    original_size = image.GetSize()  # 原始尺寸
    transform = sitk.Transform()
    transform.SetIdentity()
    # 計算Resample的大小
    new_size = [int(sz * spc / new_spc + 0.5) for sz, spc, new_spc in zip(original_size, original_spacing, new_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)  # 使用線性插值器，可以根據需要更改插值方法
    resampler.SetTransform(transform)
    # 進行Resample
    new_image = resampler.Execute(image)
    new_image.SetSpacing(new_spacing)
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetDirection(image.GetDirection())
    # xyzt_units
    new_image.SetMetaData("xyzt_units", image.GetMetaData('xyzt_units'))
    # 保存新的影像
    sitk.WriteImage(new_image, output_file_path)
    return output_file_path


def get_original_z_index(img_array,z_i,img_1mm_ori,z_i1,z_pixdim_img):
    if z_i < 64:
        img_repeat = np.expand_dims(img_array, -1).repeat(z_i1, axis=-1)
        img_1mm_repeat = np.expand_dims(img_1mm_ori, 2).repeat(z_i, axis=2)
        diff = np.sum(np.abs(img_1mm_repeat - img_repeat), axis=(0, 1))
        argmin = np.argmin(diff, axis=1)
    else:
        argmin_list = []
        broadcast_shape = img_1mm_ori.shape
        for z_index in range(z_i):
            img_array_broadcast = np.broadcast_to(np.expand_dims(img_array[:, :, z_index], -1), broadcast_shape)
            diff = np.sum(np.abs(img_array_broadcast - img_1mm_ori), axis=(0, 1))
            argmin_list.append(np.argmin(diff))
        argmin = np.array(argmin_list)
    return check_original_z_index(argmin, img_array, img_1mm_ori, z_pixdim_img)


def check_original_z_index(argmin,img_array,img_1mm_ori,z_pixdim_img):
    z_index = np.arange(0, argmin.shape[0], 1)
    z_index += 1
    z_index *= int(z_pixdim_img)
    diff_index = z_index - argmin
    if np.any(diff_index < 0):
        error_index = z_index[diff_index < 0]
        for i in error_index:
            img_index = np.argmax(z_index == i)
            start_i = int(i - (z_pixdim_img))
            end_i = int(i)
            broadcast_shape = img_1mm_ori[:, :, start_i:end_i].shape
            img_array_broadcast = np.broadcast_to(np.expand_dims(img_array[:, :, img_index], -1), broadcast_shape)
            diff = np.sum(np.abs(img_array_broadcast - img_1mm_ori[:, :, start_i:end_i]), axis=(0, 1))
            argmin[img_index] = np.argmin(diff) + start_i
    return argmin



def resampleSynthSEG2original(raw_file: pathlib.Path,
                              resample_image_file: pathlib.Path,
                              resample_seg_file: pathlib.Path):
    img_nii, img_array, y_i, x_i, z_i, header_img, pixdim_img = get_volume_info(str(raw_file))
    img_1mm_nii, img_1mm_array, y_i1, x_i1, z_i1, header_img_1mm, pixdim_img_1mm = get_volume_info(str(resample_image_file))

    # 先把影像從230*230*140轉成 original*original*140
    img_1mm_ori_nii = nibabel.processing.conform(img_1mm_nii,
                                                 ((y_i, x_i, z_i1)),
                                                 (pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3]),
                                                 order=1)  # 影像用，256*256*140
    img_1mm_ori = np.array(img_1mm_ori_nii.dataobj)


    # 處理 小於0 的狀況
    z_pixdim_img = max(int(header_img['pixdim'][3]),1)
    argmin = get_original_z_index(img_array, z_i, img_1mm_ori, z_i1, z_pixdim_img)

    SynthSEG_1mm_nii = nib.load(str(resample_seg_file))  # 230*230*140
    SynthSEG_1mm_ori_nii = nibabel.processing.conform(SynthSEG_1mm_nii,
                                                      ((y_i, x_i, z_i1)),
                                                      (pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3])
                                                      , order=0)  # 256*256*140
    SynthSEG_1mm_ori = np.array(SynthSEG_1mm_ori_nii.dataobj)
    SynthSEG_1mm_ori = data_translate(SynthSEG_1mm_ori, SynthSEG_1mm_ori_nii)

    new_array = SynthSEG_1mm_ori[:, :, argmin]

    new_array_save = data_translate_back(new_array, img_nii)
    new_SynthSeg_nii = nii_img_replace(img_nii, new_array_save)
    original_seg_file = resample_seg_file.parent.joinpath(
        f"synthseg_{resample_seg_file.name.replace('resample', 'original')}")
    nib.save(new_SynthSeg_nii, original_seg_file)
    return original_seg_file



def resampleSynthSEG2original_z_index(raw_file: pathlib.Path,
                                      resample_image_file: pathlib.Path,
                                      resample_seg_file: pathlib.Path):
    img_nii, img_array, y_i, x_i, z_i, header_img, pixdim_img = get_volume_info(str(raw_file))
    img_1mm_nii, img_1mm_array, y_i1, x_i1, z_i1, header_img_1mm, pixdim_img_1mm = get_volume_info(
        str(resample_image_file))

    # 先把影像從230*230*140轉成 original*original*140
    img_1mm_ori_nii = nibabel.processing.conform(img_1mm_nii,
                                                 ((y_i, x_i, z_i1)),
                                                 (pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3]),
                                                 order=1)  # 影像用，256*256*140
    img_1mm_ori = np.array(img_1mm_ori_nii.dataobj)

    # 處理 小於0 的狀況
    z_pixdim_img = max(int(header_img['pixdim'][3]), 1)
    argmin = get_original_z_index(img_array, z_i, img_1mm_ori, z_i1, z_pixdim_img)

    SynthSEG_1mm_nii = nib.load(str(resample_seg_file))  # 230*230*140
    SynthSEG_1mm_ori_nii = nibabel.processing.conform(SynthSEG_1mm_nii,
                                                      ((y_i, x_i, z_i1)),
                                                      (pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3])
                                                      , order=0)  # 256*256*140
    SynthSEG_1mm_ori = np.array(SynthSEG_1mm_ori_nii.dataobj)
    SynthSEG_1mm_ori = data_translate(SynthSEG_1mm_ori, SynthSEG_1mm_ori_nii)

    new_array = SynthSEG_1mm_ori[:, :, argmin]

    new_array_save = data_translate_back(new_array, img_nii)
    new_SynthSeg_nii = nii_img_replace(img_nii, new_array_save)
    original_seg_file = resample_seg_file.parent.joinpath(
        f"synthseg_{resample_seg_file.name.replace('resample', 'original')}")
    nib.save(new_SynthSeg_nii, original_seg_file)
    return original_seg_file,argmin


def save_original_seg_by_argmin_z_index(raw_file: pathlib.Path,
                                        resample_seg_file: pathlib.Path,
                                        argmin):
    img_nii, img_array, y_i, x_i, z_i, header_img, pixdim_img = get_volume_info(str(raw_file))
    img_1mm_nii, img_1mm_array, y_i1, x_i1, z_i1, header_img_1mm, pixdim_img_1mm = get_volume_info(str(resample_seg_file))


    SynthSEG_1mm_nii = nib.load(str(resample_seg_file))  # 230*230*140
    SynthSEG_1mm_ori_nii = nibabel.processing.conform(SynthSEG_1mm_nii,
                                                      ((y_i, x_i, z_i1)),
                                                      (pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3])
                                                      , order=0)  # 256*256*140
    SynthSEG_1mm_ori = np.array(SynthSEG_1mm_ori_nii.dataobj)
    SynthSEG_1mm_ori = data_translate(SynthSEG_1mm_ori, SynthSEG_1mm_ori_nii)

    new_array = SynthSEG_1mm_ori[:, :, argmin]

    new_array_save = data_translate_back(new_array, img_nii)
    new_SynthSeg_nii = nii_img_replace(img_nii, new_array_save)
    original_seg_file = resample_seg_file.parent.joinpath(
        f"synthseg_{resample_seg_file.name.replace('resample', 'original')}")
    nib.save(new_SynthSeg_nii, original_seg_file)
    return original_seg_file


def save_nii_original_seg_by_argmin(raw_file: pathlib.Path,
                                    resample_seg_file: pathlib.Path,
                                    argmin):
    img_nii, img_array, y_i, x_i, z_i, header_img, pixdim_img = get_volume_info(str(raw_file))
    img_1mm_nii, img_1mm_array, y_i1, x_i1, z_i1, header_img_1mm, pixdim_img_1mm = get_volume_info(str(resample_seg_file))


    SynthSEG_1mm_nii = nib.load(str(resample_seg_file))  # 230*230*140
    SynthSEG_1mm_ori_nii = nibabel.processing.conform(SynthSEG_1mm_nii,
                                                      ((y_i, x_i, z_i1)),
                                                      (pixdim_img[1], pixdim_img[2], pixdim_img_1mm[3])
                                                      , order=0)  # 256*256*140
    SynthSEG_1mm_ori = np.array(SynthSEG_1mm_ori_nii.dataobj)
    SynthSEG_1mm_ori = data_translate(SynthSEG_1mm_ori, SynthSEG_1mm_ori_nii)

    new_array = SynthSEG_1mm_ori[:, :, argmin]

    new_array_save = data_translate_back(new_array, img_nii)
    new_SynthSeg_nii = nii_img_replace(img_nii, new_array_save)
    original_seg_file = resample_seg_file.parent.joinpath(
        f"synthseg_{resample_seg_file.name.replace('resample', 'original')}")
    nib.save(new_SynthSeg_nii, original_seg_file)
    return original_seg_file


def data_translate(img, nii):
    img = np.swapaxes(img, 0, 1)
    img = np.flip(img, 0)
    img = np.flip(img, -1)
    header = nii.header.copy()  # 抓出nii header 去算體積
    pixdim = header['pixdim']  # 可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)
        # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img

def data_translate_back(img, nii):
    header = nii.header.copy()  # 抓出nii header 去算體積
    pixdim = header['pixdim']  # 可以借此從nii的header抓出voxel size
    if pixdim[0] > 0:
        img = np.flip(img, 1)
    img = np.flip(img, -1)
    img = np.flip(img, 0)
    img = np.swapaxes(img, 1, 0)
    # img = np.expand_dims(np.expand_dims(img, axis=0), axis=4)
    return img


# nii改變影像後儲存，data => nib.load的輸出，new_img => 更改的影像
def nii_img_replace(data, new_img):
    affine = data.affine
    header = data.header.copy()
    new_nii = nib.nifti1.Nifti1Image(new_img, affine, header=header)
    return new_nii


def get_volume_info(file_path_str:str):
    img_nii = nib.load(file_path_str)  # 256*256*22
    img_array = np.array(img_nii.dataobj)
    y_i, x_i, z_i = img_array.shape
    header_img = img_nii.header.copy()  # 抓出nii header 去算體積
    pixdim_img = header_img['pixdim']  # 可以借此從nii的header抓出voxel size
    return img_nii,img_array,y_i, x_i, z_i,header_img, pixdim_img
