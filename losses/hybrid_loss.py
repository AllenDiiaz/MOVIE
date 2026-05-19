import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from lpips import LPIPS

class ImageFlowNetLoss(nn.Module):
    """
    L = wm * MSE + ws * (1 - SSIM)/2 + wl * LPIPS
        + w_sm * || drift ||² + w_c * (1 - cos_sim(early, mid))

    其中 (1-SSIM)/2 可把 SSIM ∈ [-1,1] 映射到 [0,1]，和 MSE 量級一致；
    LPIPS 已經是距離形式 (越小越好)。
    """
    def __init__(
        self,
        mse_w: float   = 1.0,
        ssim_w: float  = 1.0,
        lpips_w: float = 1.0,
        smooth_w: float = 5e-4,
        cont_w: float   = 1.0,
        resize_lpips: bool = True        # LPIPS 需輸入 3-channel；此選項會自動複製通道
    ):
        super().__init__()
        self.mse_w   = mse_w
        self.ssim_w  = ssim_w
        self.lpips_w = lpips_w
        self.w_sm    = smooth_w
        self.w_c     = cont_w

        # torchmetrics 的 SSIM 要在 forward 用；LPIPS 建一個 once-for-all 的 model
        self.mse   = nn.MSELoss()
        self.lpips = LPIPS(net='vgg').eval()          # 記得在外部 model.eval()
        for p in self.lpips.parameters():             # LPIPS 不更新梯度
            p.requires_grad_(False)
        self.resize_lpips = resize_lpips

    @staticmethod
    def _to_rgb(x: torch.Tensor) -> torch.Tensor:
        """[B,1,H,W] → [B,3,H,W]  (通道複製；LPIPS 需要 3 通道)"""
        return x.repeat(1, 3, 1, 1)

    def forward(
        self,
        pred:  torch.Tensor,
        tgt:   torch.Tensor,
        drift: torch.Tensor | None = None,
        early: torch.Tensor | None = None,
        mid:   torch.Tensor | None = None,
    ) -> torch.Tensor:

        # ---------- 基本影像誤差 -------------------------------------------------
        mse  = self.mse(pred, tgt)
        ssim = (1.0 - ssim_fn(pred, tgt, data_range=1.0)) * 0.5     # → [0,1]
        if self.resize_lpips and pred.size(1) == 1:
            lpips_val = self.lpips(self._to_rgb(pred), self._to_rgb(tgt)).mean()
        else:
            lpips_val = self.lpips(pred, tgt).mean()

        loss = (
            self.mse_w   * mse  +
            self.ssim_w  * ssim +
            self.lpips_w * lpips_val
        )

        # ---------- 其他正則項 ---------------------------------------------------
        if drift is not None:
            loss = loss + self.w_sm * drift.pow(2).mean()

        if (early is not None) and (mid is not None):
            e = F.normalize(early, p=2, dim=1)
            m = F.normalize(mid,   p=2, dim=1)
            loss = loss + self.w_c * (1.0 - (e * m).sum(1).mean())

        return loss
    
class TripletContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()  # ✅ 更簡潔的寫法
        self.margin = margin

    def forward(self, output, early_frame, late_frame):
        # 計算 L2 距離
        positive_dist = F.mse_loss(output, late_frame)  # Output 應該接近 Late Frame
        negative_dist = F.mse_loss(output, early_frame)  # Output 應該遠離 Early Frame

        # Triplet Loss 計算
        loss = F.relu(positive_dist - negative_dist + self.margin)
        return loss
    
class CosineContrastiveLoss(nn.Module):
    def __init__(self, lambda_early=0.5, lambda_late=0.5):
        super().__init__()
        self.lambda_early = lambda_early
        self.lambda_late = lambda_late

    def forward(self, output, early_frame, late_frame):
        # **確保輸出的通道數匹配 early_frame**
        if output.shape[1] == 1:
            output = output.repeat(1, 3, 1, 1)

        # **確保 output 和 late_frame 解析度一致**
        if output.shape[2:] != late_frame.shape[2:]:
            output = F.interpolate(output, size=late_frame.shape[2:], mode="bilinear", align_corners=False)

        if early_frame.shape[2:] != late_frame.shape[2:]:
            early_frame = F.interpolate(early_frame, size=late_frame.shape[2:], mode="bilinear", align_corners=False)

        cos_sim_late = cosine_loss(output, late_frame)
        cos_sim_early = cosine_loss(output, early_frame)

        loss = self.lambda_late * (1 - cos_sim_late) + self.lambda_early * cos_sim_early
        return loss