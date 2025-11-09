import os
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.regression import mean_absolute_error
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from PIL import Image
import torchvision.transforms.functional as F
from tqdm import tqdm
import pandas as pd
from skimage import io, color
from skimage.transform import resize
import numpy as np

def compare_images(str_img_orig, str_img_edit):
    # read images
    im_orig = resize(io.imread(str_img_orig), (256, 256))
    # im_edit = resize(io.imread(str_img_edit), (256, 256))

    im_edit = io.imread(str_img_edit)

    # convert to lab
    lab_orig = color.rgb2lab(im_orig)
    lab_edit = color.rgb2lab(im_edit)

    # calculate difference
    de_diff = color.deltaE_ciede2000(lab_orig, lab_edit)

    return np.mean(de_diff)


tar_folder = 'target'
datasets = os.listdir(tar_folder)
# datasets = ['Jung']
res_folder = 'ours'
methods = os.listdir(res_folder)
criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
for ds in datasets:
    lst = []
    for method in methods:
        metric = {'Method': method, 'PSNR': 0, 'SSIM': 0, 'RMSE': 0}
        imgs = os.listdir(os.path.join(res_folder, method, ds))
        stat_psnr = 0
        stat_ssim = 0
        stat_mae = 0
        stat_lpips = 0
        delta_e = 0
        size = len(imgs)
        for img in tqdm(imgs):
            if '.png' or '.jpg' in img:
                res = Image.open(os.path.join(res_folder, method, ds, img)).convert('RGB')
                tar = Image.open(os.path.join(tar_folder, ds, img)).convert('RGB')

                res = F.to_tensor(res)
                tar = F.to_tensor(tar)

                # res = F.resize(res, (256, 256))
                tar = F.resize(tar, (256, 256))

                res = res.cuda().unsqueeze(0)
                tar = tar.cuda().unsqueeze(0)

                stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
                stat_ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
                stat_mae += mean_absolute_error(res * 255, tar * 255).item()
                delta_e += compare_images(
                    os.path.join(tar_folder, ds, img),
                    os.path.join(res_folder, method, ds, img),
                )
                stat_lpips += criterion_lpips(res, tar).item()

        stat_psnr /= size
        stat_ssim /= size
        stat_mae /= size
        delta_e /= size
        stat_lpips /= size
        metric['PSNR'] = round(stat_psnr, 3)
        metric['SSIM'] = round(stat_ssim, 3)
        metric['MAE'] = round(stat_mae, 3)
        metric['Delta_E'] = round(delta_e, 3)
        metric['LPIPS'] = round(stat_lpips, 3)
        df = pd.DataFrame(metric, index=[0])
        lst.append(df)
        # print("method: {}, dataset: {}, PSNR: {}, SSIM: {}, RMSE: {}".format(method, ds, stat_psnr, stat_ssim, stat_rmse))
    lst = pd.concat(lst)
    lst.to_csv(ds + '.csv')


