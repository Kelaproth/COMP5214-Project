import skimage.metrics
import numpy as np
import lpips

def eval(model, lr_img, gt_img, basic_configs):
    """
    Evaluate PSNR, SSIM, MSE, Normalized MI as pixel-wise evaluation here.
    Evaluate LPIPS based on Alexnet here.
    """
    device = torch.device(basic_configs['device'])
    sr_img = model(lr_img)

    loss_fn = lpips.LPIPS(net='alex', spatial=spatial)
    loss_fn.to(device)
    dist = loss_fn.forward(sr_img, gt_img)
	dist_.append(dist.mean().item())
    
    psnr, ssim, mse, nmi = [], [], [], []
    yimg = sr_img.cpu().detach().numpy()
    gtimg = gt_img.cpu().detach().numpy()
    psnr.append(skimage.metrics.peak_signal_noise_ratio(yimg, gtimg, data_range=2))
    ssim.append(skimage.metrics.structural_similarity(yimg, gtimg))
    mse.append(skimage.metrics.mean_squared_error(yimg, gtimg))
    nmi.append(skimage.metrics.normalized_mutual_information(yimg, gtimg))

    
    print(" PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f LPIPS: %.4f"%(np.mean(psnr), np.mean(ssim), np.mean(mse), np.mean(nmi), np.mean(dist_)))