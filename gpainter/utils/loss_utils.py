import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from lpips import LPIPS
from ..modules import sparse as sp

def smooth_l1_loss(pred, target, beta=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


loss_fn_vgg = None
def lpips(img1, img2, value_range=(0, 1)):
    global loss_fn_vgg
    if loss_fn_vgg is None:
        loss_fn_vgg = LPIPS(net='vgg').cuda().eval()
    # normalize to [-1, 1]
    img1 = (img1 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    img2 = (img2 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    return loss_fn_vgg(img1, img2).mean()


def normal_angle(pred, gt):
    pred = pred * 2.0 - 1.0
    gt = gt * 2.0 - 1.0
    norms = pred.norm(dim=-1) * gt.norm(dim=-1)
    cos_sim = (pred * gt).sum(-1) / (norms + 1e-9)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    ang = torch.rad2deg(torch.acos(cos_sim[norms > 1e-9])).mean()
    if ang.isnan():
        return -1
    return ang


def curvature_loss(pred: sp.SparseTensor, target: sp.SparseTensor) -> torch.Tensor:
    """
    曲率感知损失 - 高曲率区域(边缘/角落)需要更精确重建

    原理: 从UDF二阶导数估计曲率
    - Hessian矩阵的特征值 → 主曲率
    - 高曲率区域权重更大
    """
    total_loss = 0.0

    for b in range(pred.shape[0]):
        layout = pred.layout[b]
        pred_feats = pred.feats[layout]
        target_feats = target.feats[layout]

        # Reshape到grid
        padded_size = int(round(pred_feats.shape[1] ** (1/3)))
        pred_grid = pred_feats.reshape(-1, padded_size, padded_size, padded_size)
        target_grid = target_feats.reshape(-1, padded_size, padded_size, padded_size)

        # 计算二阶导数(Laplacian) - 近似曲率
        laplacian_pred = compute_laplacian_3d(pred_grid)
        laplacian_target = compute_laplacian_3d(target_grid)

        # 高曲率区域 (|Laplacian| > threshold)
        curvature_weight = torch.abs(laplacian_target)
        curvature_weight = (curvature_weight - curvature_weight.min()) / \
                          (curvature_weight.max() - curvature_weight.min() + 1e-8)

        # 加权MSE
        diff = (pred_grid - target_grid) ** 2
        weighted_loss = (diff * (1 + 5 * curvature_weight)).mean()

        total_loss += weighted_loss

    return total_loss / pred.shape[0]


def compute_laplacian_3d(grid):
    """3D Laplacian算子: Δf = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²"""
    laplacian = torch.zeros_like(grid)

    # 中心差分计算二阶导
    laplacian[:, 1:-1, :, :] += grid[:, 2:, :, :] - 2*grid[:, 1:-1, :, :] + grid[:, :-2, :, :]
    laplacian[:, :, 1:-1, :] += grid[:, :, 2:, :] - 2*grid[:, :, 1:-1, :] + grid[:, :, :-2, :]
    laplacian[:, :, :, 1:-1] += grid[:, :, :, 2:] - 2*grid[:, :, :, 1:-1] + grid[:, :, :, :-2]

    return laplacian


def udf_gradient_consistency_loss(pred: sp.SparseTensor, target: sp.SparseTensor) -> torch.Tensor:
    """
    UDF梯度一致性 - ||∇UDF|| ≈ 1 约束

    理论基础: 正确的UDF应该在任何点梯度模长接近1
    这是Eikonal方程的约束: ||∇d|| = 1
    """
    total_loss = 0.0

    for b in range(pred.shape[0]):
        layout = pred.layout[b]
        pred_feats = pred.feats[layout]

        padded_size = int(round(pred_feats.shape[1] ** (1/3)))
        pred_grid = pred_feats.reshape(-1, padded_size, padded_size, padded_size)

        # 计算梯度
        grad_x = torch.diff(pred_grid, dim=1)
        grad_y = torch.diff(pred_grid, dim=2)
        grad_z = torch.diff(pred_grid, dim=3)

        # 梯度模长
        grad_norm_x = torch.abs(grad_x)
        grad_norm_y = torch.abs(grad_y)
        grad_norm_z = torch.abs(grad_z)

        # Eikonal约束: ||grad|| ≈ 1
        eikonal_loss = (
            torch.mean((grad_norm_x - 1.0) ** 2) +
            torch.mean((grad_norm_y - 1.0) ** 2) +
            torch.mean((grad_norm_z - 1.0) ** 2)
        ) / 3.0

        total_loss += eikonal_loss

    return total_loss / pred.shape[0]

def multiscale_consistency_loss(pred: sp.SparseTensor, target: sp.SparseTensor) -> torch.Tensor:
    """
    多尺度一致性 - 利用UDF的层级结构

    想法:
    - 下采样到不同分辨率
    - 确保粗糙尺度和精细尺度的UDF保持一致
    """
    total_loss = 0.0

    for b in range(pred.shape[0]):
        layout = pred.layout[b]
        pred_feats = pred.feats[layout]
        target_feats = target.feats[layout]

        padded_size = int(round(pred_feats.shape[1] ** (1/3)))
        pred_grid = pred_feats.reshape(-1, padded_size, padded_size, padded_size)
        target_grid = target_feats.reshape(-1, padded_size, padded_size, padded_size)

        # 多尺度下采样
        for scale in [2, 4]:
            pred_down = F.avg_pool3d(pred_grid.unsqueeze(1), kernel_size=scale, stride=scale).squeeze(1)
            target_down = F.avg_pool3d(target_grid.unsqueeze(1), kernel_size=scale, stride=scale).squeeze(1)

            scale_loss = F.mse_loss(pred_down, target_down)
            total_loss += scale_loss * (1.0 / scale)  # 粗尺度权重递减

    return total_loss / pred.shape[0]