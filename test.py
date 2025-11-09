import warnings

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.regression import mean_absolute_error
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from data import get_test_data
from models import *
from utils import *

warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)


def test():
    accelerator = Accelerator()

    # Data Loader
    val_dir = opt.TRAINING.VAL_DIR

    val_dataset = get_test_data(val_dir, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': opt.TRAINING.ORI})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)
    
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

    # Model & Metrics
    model = Model()

    os.makedirs('result', exist_ok=True)

    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, testloader = accelerator.prepare(model, testloader)

    model.eval()

    size = len(testloader)
    stat_psnr = 0
    stat_ssim = 0
    stat_lpips = 0
    stat_mae = 0
    for _, test_data in enumerate(tqdm(testloader)):
        # get the inputs; data is a list of [targets, inputs, filename]
        inp = test_data[0].contiguous()
        tar = test_data[1]

        with torch.no_grad():
            res = model(inp)[0].clamp(0, 1)

        save_image(res, os.path.join(os.getcwd(), "result", test_data[2][0]))

        stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1)
        stat_ssim += structural_similarity_index_measure(res, tar, data_range=1)
        stat_mae += mean_absolute_error(torch.mul(res, 255), torch.mul(tar, 255))
        stat_lpips += criterion_lpips(res, tar).item()

    stat_psnr /= size
    stat_ssim /= size
    stat_mae /= size
    stat_lpips /= size

    print("PSNR: {}, SSIM: {}, MAE: {}, LPIPS: {}".format(stat_psnr, stat_ssim, stat_mae, stat_lpips))


if __name__ == '__main__':
    test()
