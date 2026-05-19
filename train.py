# -*- coding: utf-8 -*- Training & Validate (GuideDropOut)
"""
train.py  –  LKMUNet-SDE-FiLM    (micro-batch, ckpt, LPIPS val-only)
"""

from __future__ import annotations    
from pathlib import Path
from typing import Tuple, Dict, List
import gc, math, os, json
import numpy as np              
import pandas as pd              
import matplotlib.pyplot as plt  
import seaborn as sns            

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from torchmetrics.functional import peak_signal_noise_ratio as calc_psnr
from torchmetrics.functional import structural_similarity_index_measure as calc_ssim
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from lpips import LPIPS
from colorama import Fore, Style
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random    
import warnings
from datetime import datetime

from models.sde_film import build_lkmunet_sde_film
from models.ode_film import build_lkmunet_ode_film
from models.lkmunet import apply_gradient_checkpoint, CheckpointWrapper
from datasets.early2late_dataset import Early2LateWithLatentDataset

sns.set_style("whitegrid")

# ────────────────────────── 全域開關 ──────────────────────────
PLOT_METRIC  = True
VISUALIZE    = True
SAVE_SYMLINK = True
ACC_STEPS    = 4   
LR           = 1e-4
WEIGHT_DECAY = 3e-2
DROP_FULL_P  = 0.3
DROP_PARTIAL_P = 0.3
CLIP_GRAD_NORM = 1.0
# ─────────────────────────────────────────────────────────────

# ═══════════════════════ Utilities ════════════════════════
def ensure_5d(t: torch.Tensor) -> torch.Tensor:
    t = t.squeeze()
    if t.ndim == 5: return t
    if t.ndim == 4: return t.unsqueeze(2)
    if t.ndim == 3: return t.unsqueeze(1).unsqueeze(2)
    raise ValueError(f"latent shape {t.shape} unsupported")

def squeeze_early(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(2) if (x.ndim == 5 and x.size(2) == 1) else x

# ────────────────────────── Baseline CSV ───────────────────
def load_baseline_metrics(fold: int, csv_path: str) -> Dict[str, float]:
    """Return {'psnr':…, 'ssim':…, 'lpips':…} for given fold."""
    if not csv_path:
        return {"psnr": -np.inf, "ssim": -np.inf, "lpips": np.inf}
    if not Path(csv_path).exists():
        return {'psnr': -np.inf, 'ssim': -np.inf, 'lpips': np.inf}
    df = pd.read_csv(csv_path)
    row = df[df["Fold"] == fold]
    if row.empty:
        return {'psnr': -np.inf, 'ssim': -np.inf, 'lpips': np.inf}
    return {'psnr': row["mean_psnr"].max(),
            'ssim': row["mean_ssim"].max(),
            'lpips': row["mean_lpips"].min()}

# ────────────────────────── helper：metric plot ────────────────
def update_and_plot_metrics(
        epoch: int,
        train_losses: List[float], valid_losses: List[float],
        psnr_scores: List[float],  ssim_scores: List[float],
        lpips_scores: List[float], mse_scores: List[float],
        save_dir: Path):

    loss_dir   = save_dir / "Loss";   loss_dir.mkdir(parents=True, exist_ok=True)
    metric_dir = save_dir / "Metric"; metric_dir.mkdir(parents=True, exist_ok=True)

    np.save(loss_dir / "train_losses.npy",  np.array(train_losses,  np.float32))
    np.save(loss_dir / "valid_losses.npy",  np.array(valid_losses,  np.float32))
    np.save(metric_dir / "psnr_scores.npy", np.array(psnr_scores, np.float32))
    np.save(metric_dir / "ssim_scores.npy", np.array(ssim_scores, np.float32))
    np.save(metric_dir / "lpips_scores.npy",np.array(lpips_scores, np.float32))
    np.save(metric_dir / "mse_scores.npy",  np.array(mse_scores,  np.float32))

    plt.figure(figsize=(6,4))
    sns.lineplot(x=range(1,len(train_losses)+1), y=train_losses, label="Train")
    sns.lineplot(x=range(1,len(valid_losses)+1), y=valid_losses, label="Valid")
    best = int(np.argmin(valid_losses)); best_val = valid_losses[best]
    plt.scatter(best+1, best_val, color='red')
    plt.annotate(f"Best E{best+1}: {best_val:.4f}", (best+1, best_val),
                 xytext=(0,-12), textcoords="offset points", ha="center", fontsize=8)
    plt.title("Loss Curve"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.tight_layout(); plt.savefig(loss_dir/"loss_curve.png", dpi=300); plt.close()

    def _curve(arr, name, larger_better=True):
        plt.figure(figsize=(6,4))
        sns.lineplot(x=range(1,len(arr)+1), y=arr)
        idx = int(np.argmax(arr) if larger_better else np.argmin(arr))
        plt.scatter(idx+1, arr[idx], color='red')
        plt.annotate(f"Best E{idx+1}: {arr[idx]:.4f}", (idx+1, arr[idx]),
                     xytext=(0,-12), textcoords="offset points",
                     ha="center", fontsize=8)
        plt.title(f"{name} Curve"); plt.xlabel("Epoch"); plt.ylabel(name)
        plt.tight_layout()
        plt.savefig(metric_dir/f"{name.lower()}_curve.png", dpi=300)
        plt.close()
    _curve(psnr_scores, "PSNR", True)
    _curve(ssim_scores, "SSIM", True)
    _curve(lpips_scores, "LPIPS", False)
    _curve(mse_scores, "MSE", False)

def to_5d(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        return None
    if x.ndim == 5:
        if x.shape[2] in (1, 3) and x.shape[1] != 1:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x
    if x.ndim == 6:
        B, C1, M, C2, H, W = x.shape
        if C1 == 1 and C2 == 1:
            return x.view(B, M, 1, H, W)
        if C2 == 1:
            return x.permute(0, 2, 1, 4, 5).contiguous()
        raise ValueError(f"6‑D latent 不支援形狀 {x.shape}")
    if x.ndim == 4:
        return x.unsqueeze(1)
    if x.ndim == 3:
        return x.unsqueeze(1).unsqueeze(2)
    raise ValueError(f"unsupported latent shape {x.shape}")


def guide_dropout(mid_5d: torch.Tensor | None,
                  p_full: float = DROP_FULL_P,
                  p_part: float = DROP_PARTIAL_P) -> torch.Tensor | None:
    if mid_5d is None:
        return None
    if torch.rand(1) < p_full:
        return None
    if mid_5d.size(1) > 1:
        keep = torch.rand(mid_5d.size(1), device=mid_5d.device) > p_part
        if keep.sum() == 0:
            keep[torch.randint(0, keep.numel(), (1,))] = True
        mid_5d = mid_5d[:, keep]
    return mid_5d


# ──────────────────── Validation Visualization ──────────────────
def visualize_validation_epoch(
    inputs: Dict[str, np.ndarray],
    guides: Dict[str, np.ndarray],
    synths: Dict[str, np.ndarray],
    gts:    Dict[str, np.ndarray],
    epoch:  int,
    save_dir: Path,
    json_path: str,                     # ← 改成參數，不再 hardcode
    dpi: int = 300
):
    if not inputs:
        return

    vis_dir = save_dir / "Visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not Path(json_path).exists():
        return

    with open(json_path, "r") as f:
        data = json.load(f)
    meta = {d["Subject"]: d for d in data} if isinstance(data, list) else data

    prefer     = ["Subject010","Subject042","Subject001","Subject036","Subject076"]
    diag_order = ["AD","CAA_ICH","CAA_CI","HTN"]
    chosen: Dict[str,str] = {}
    for diag in diag_order:
        cand = [s for s in prefer if s in inputs and meta.get(s, {}).get("Diagnosis") == diag]
        if not cand:
            cand = [s for s in inputs if meta.get(s, {}).get("Diagnosis") == diag]
        if cand:
            chosen[diag] = cand[0]
    if not chosen:
        return

    rows = len(chosen)
    fig, ax = plt.subplots(rows, 6, figsize=(18, 3*rows), dpi=dpi)
    if rows == 1:
        ax = [ax]
    titles = ["Input", "Guide-1", "Guide-2", "Guide-3", "Synthesis", "GT"]

    for r, diag in enumerate(diag_order):
        if diag not in chosen:
            continue
        sid = chosen[diag]
        row = ax[r]
        imgs = [inputs[sid], *guides[sid], synths[sid], gts[sid]]

        for c, img in enumerate(imgs):
            im = np.rot90(img)
            vmin, vmax = im.min(), im.max()
            row[c].imshow(im, cmap="hot", vmin=vmin, vmax=vmax)
            row[c].axis("off")
            row[c].set_title(f"{titles[c]}\nmin={vmin:.3f}, max={vmax:.3f}", fontsize=9)

        info = meta.get(sid, {})
        row[0].set_ylabel(
            f"{sid}\n{diag}\nAβ:{info.get('Abeta', '?')}",
            fontsize=10, rotation=0, labelpad=50, va="center"
        )

    plt.suptitle(f"Validation Visualization – Epoch {epoch}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(vis_dir / f"epoch_{epoch:03d}_validation.png")
    plt.close()


# ───────────────────────────────── loss ───────────────────────
class ImageFlowNetLoss(nn.Module):
    """
    L = wm * MSE + ws * (1 - SSIM)/2 + wl * LPIPS
        + w_sm * || drift ||² + w_c * (1 - cos_sim(early, mid))
    """
    def __init__(
        self,
        mse_w: float    = 1.0,
        ssim_w: float   = 1.0,
        lpips_w: float  = 1.0,
        smooth_w: float = 5e-4,
        cont_w: float   = 1.0,
        resize_lpips: bool = True
    ):
        super().__init__()
        self.mse_w   = mse_w
        self.ssim_w  = ssim_w
        self.lpips_w = lpips_w
        self.w_sm    = smooth_w
        self.w_c     = cont_w
        self.mse     = nn.MSELoss()
        self.lpips   = LPIPS(net='vgg').eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        self.resize_lpips = resize_lpips

    @staticmethod
    def _to_rgb(x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, 3, 1, 1)

    def forward(
        self,
        pred:  torch.Tensor,
        tgt:   torch.Tensor,
        drift: torch.Tensor | None = None,
        early: torch.Tensor | None = None,
        mid:   torch.Tensor | None = None,
    ) -> torch.Tensor:
        mse  = self.mse(pred, tgt)
        ssim = (1.0 - ssim_fn(pred, tgt, data_range=1.0)) * 0.5
        if self.resize_lpips and pred.size(1) == 1:
            lpips_val = self.lpips(self._to_rgb(pred), self._to_rgb(tgt)).mean()
        else:
            lpips_val = self.lpips(pred, tgt).mean()

        loss = (
            self.mse_w   * mse  +
            self.ssim_w  * ssim +
            self.lpips_w * lpips_val
        )

        if drift is not None:
            loss = loss + self.w_sm * drift.pow(2).mean()

        if (early is not None) and (mid is not None):
            e = F.normalize(early, p=2, dim=1)
            m = F.normalize(mid,   p=2, dim=1)
            loss = loss + self.w_c * (1.0 - (e * m).sum(1).mean())

        return loss

# ───────────────────────────────── early-stopper ─────────────────────────
class EarlyStopper:
    def __init__(self, patience: int, delta: float, baseline: Dict[str, float]):
        self.patience, self.delta = patience, delta
        self.best_score = -np.inf
        self.best_lpips = np.inf
        self.wait = 0
        self.baseline = baseline
        self.first = True

    def check_improve(self, psnr: float, ssim: float, lpips: float) -> bool:
        if self.first:
            self.first = False
            return True
        better_than_base = (psnr > self.baseline['psnr'] and
                            ssim > self.baseline['ssim'] and
                            lpips < self.baseline['lpips'])
        if not better_than_base:
            return False
        score = psnr + 25*ssim
        improved = (score > self.best_score + self.delta) or (lpips < self.best_lpips - self.delta)
        if improved:
            self.best_score, self.best_lpips, self.wait = score, lpips, 0
        else:
            self.wait += 1
        return improved

    @property
    def stop(self) -> bool:
        return self.wait >= self.patience


# ═══════════════════════ Train & Validate ══════════════════════
def train_and_validate(model          : nn.Module,
                       train_loader   : DataLoader,
                       valid_loader   : DataLoader,
                       optimizer,
                       warmup_sched,
                       epoch_sched,
                       num_epochs     : int,
                       save_path      : str | Path,
                       fold           : int,
                       acc_steps      : int = ACC_STEPS,
                       baseline_csv   : str = "",
                       vis_json_path  : str = ""):   # ← 改成參數

    device       = next(model.parameters()).device
    criterion_tr = ImageFlowNetLoss().to(device)
    lpips_val    = LPIPS(net='vgg').to(device).eval()

    save_path = Path(save_path)
    baseline  = load_baseline_metrics(fold, baseline_csv)
    stopper   = EarlyStopper(patience=25, delta=0.05, baseline=baseline)

    ckpt_dir = save_path / f"fold_{fold}/ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    h_tr, h_val = [], []
    h_psnr, h_ssim, h_lpips, h_mse = [], [], [], []

    for epoch in range(1, num_epochs+1):

        p_full = min(1.0, 0.3 + epoch / 40)
        train_mse_sum = 0.0
        nsamp_train   = 0

        # -------------------- TRAIN ----------------------------------------
        model.train(); running = 0; step = 0
        optimizer.zero_grad(set_to_none=True)

        for batch in tqdm(train_loader, desc=f'E{epoch:02d}[train]'):
            x   = batch['input'].float().to(device)
            tgt = batch['ground_truth'].float().to(device)
            mid = to_5d(batch["latent_target"].float().to(device))
            x, tgt = squeeze_early(x), squeeze_early(tgt)

            mid = guide_dropout(mid, p_full=p_full)

            pred, drift, early_lat, mid_lat = model(x, t=torch.tensor(1.0, device=device), mid_pet=mid)
            loss = criterion_tr(pred, tgt, drift, early_lat, mid_lat) / acc_steps
            loss.backward()

            mse_b = F.mse_loss(pred, tgt).item() * x.size(0)
            train_mse_sum += mse_b
            nsamp_train   += x.size(0)

            step += 1
            if step % acc_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                warmup_sched.step()

            running += loss.item() * acc_steps

        train_loss = running / len(train_loader)
        train_mse  = train_mse_sum / nsamp_train

        # ============= Validation ========================
        model.eval()
        psnr = ssim = lp = mse_sum = val_loss_sum = 0; nsamp = 0
        vis_inputs, vis_guides, vis_syns, vis_gts = {}, {}, {}, {}

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f'E{epoch:02d}[val]'):
                sid_list = [s for s in batch.get('subject')]

                x   = batch['input'].float().to(device)
                tgt = batch['ground_truth'].float().to(device)
                mid = to_5d(batch["latent_target"].float().to(device))
                x, tgt = squeeze_early(x), squeeze_early(tgt)

                pred, drift, early_lat, mid_lat = model(
                    x, t=torch.tensor(1.0, device=device), mid_pet=None)
                bs = x.size(0); nsamp += bs

                val_loss_sum += criterion_tr(pred, tgt, drift, early_lat, mid_lat).item() * bs
                psnr += calc_psnr(pred, tgt, data_range=1.).item() * bs
                ssim += calc_ssim(pred.clamp(0,1), tgt.clamp(0,1), data_range=1.).item() * bs
                lp   += lpips_val(pred.expand(-1,3,-1,-1),
                                  tgt.expand(-1,3,-1,-1)).mean().item() * bs
                mse_sum += F.mse_loss(pred, tgt).item() * bs

                if VISUALIZE:
                    x_np    = x.cpu().numpy()
                    pred_np = pred.cpu().numpy()
                    tgt_np  = tgt.cpu().numpy()
                    g_np    = mid.cpu().numpy()

                    for i, sid in enumerate(sid_list):
                        if sid not in vis_inputs:
                            vis_inputs[sid] = x_np[i, 0]
                            g3 = g_np[i, :, 0]
                            if g3.shape[0] < 3:
                                g3 = np.pad(g3, ((0,3-g3.shape[0]),(0,0),(0,0)), mode='edge')
                            vis_guides[sid] = g3[:3]
                            vis_syns[sid]   = pred_np[i, 0]
                            vis_gts[sid]    = tgt_np[i, 0]

        psnr /= nsamp; ssim /= nsamp; lp /= nsamp
        val_loss  = val_loss_sum / nsamp
        train_mse = train_mse_sum / nsamp_train
        val_mse   = mse_sum / nsamp

        print(f"E{epoch:02d} ▸ "
              f"train‑L {train_loss:.4f} | "
              f"valid‑L {val_loss:.4f} | "
              f"train‑MSE {train_mse:.5f} | "
              f"val‑MSE {val_mse:.5f} | "
              f"PSNR {psnr:.2f}  SSIM {ssim:.4f}  LPIPS {lp:.4f}")

        epoch_sched.step(ssim)

        h_tr.append(train_loss)
        h_val.append(val_loss)
        h_psnr.append(psnr)
        h_ssim.append(ssim)
        h_lpips.append(lp)
        h_mse.append(val_mse)

        if PLOT_METRIC:
            update_and_plot_metrics(epoch, h_tr, h_val, h_psnr, h_ssim, h_lpips, h_mse, save_path)
        if VISUALIZE:
            visualize_validation_epoch(vis_inputs, vis_guides, vis_syns, vis_gts,
                                       epoch, save_path, json_path=vis_json_path)

        improved = stopper.check_improve(psnr, ssim, lp)

        if improved:
            ckpt_name = (
                f"fold{fold}_epoch{epoch:03d}"
                f"_psnr{psnr:.2f}_ssim{ssim:.4f}_lpips{lp:.4f}.pth"
            )
            ckpt_path = ckpt_dir / ckpt_name
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": optimizer.state_dict()}, ckpt_path)

            if SAVE_SYMLINK:
                best_link = ckpt_dir / "best.pth"
                try:
                    if best_link.exists() or best_link.is_symlink():
                        best_link.unlink()
                    os.symlink(src=str(ckpt_path), dst=str(best_link))
                except OSError as e:
                    print(Fore.YELLOW + f"[symlink] {e}" + Style.RESET_ALL)

            print(Fore.GREEN + "  ↑ new best – model saved" + Style.RESET_ALL)

        if epoch == num_epochs:
            last_name = (
                f"fold{fold}_epoch{epoch:03d}"
                f"_psnr{psnr:.2f}_ssim{ssim:.4f}_lpips{lp:.4f}_last.pth"
            )
            last_path = ckpt_dir / last_name
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": optimizer.state_dict()}, last_path)

            last_link = ckpt_dir / "last.pth"
            try:
                if last_link.exists() or last_link.is_symlink():
                    last_link.unlink()
                os.symlink(src=str(last_path), dst=str(last_link))
            except OSError as e:
                print(Fore.YELLOW + f"[symlink] {e}" + Style.RESET_ALL)

            print(Fore.CYAN + f"  → final‑epoch model saved to {last_path.name}" + Style.RESET_ALL)


# ═══════════════════════ Main ══════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, choices=["sde","ode"], default="sde")
    parser.add_argument("--data-root",  type=str, required=True,
                        help="預處理後資料集的路徑")
    parser.add_argument("--save-dir",   type=str, default="./runs",
                        help="訓練結果儲存的上層資料夾")
    parser.add_argument("--baseline-csv", type=str, default="",
                        help="Baseline 指標 CSV（可不提供）")
    parser.add_argument("--vis-json",   type=str, default="",
                        help="Validation 視覺化用的 subject info JSON（可不提供）")
    parser.add_argument("--seeds",      type=int, nargs="+", default=[42])
    parser.add_argument("--folds",      type=int, nargs="+", default=[0,1,2,3,4])
    parser.add_argument("--epochs",     type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--strategy",   type=str, default="stack",
                        choices=["auto","concat","stack"])
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open("configs/stratified_5fold_all.json", "r") as f:
        split_all = json.load(f)

    cfg = dict(
        input_channels=1,
        n_stages=4,
        features_per_stage=[32,64,128,256],
        conv_op=nn.Conv2d,
        kernel_sizes=[(3,3)]*4,
        strides=[(1,1),(2,2),(2,2),(2,2)],
        n_conv_per_stage=[2]*4,
        num_classes=1,
        n_conv_per_stage_decoder=[2,2,2],
        deep_supervision=False,
        conv_bias=False,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=dict(eps=1e-5, affine=True),
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=dict(inplace=False),
    )

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        print(Fore.GREEN + f"[Seed] {seed} fixed" + Style.RESET_ALL)

    def make_loader(split: str, fold: int):
        ds = Early2LateWithLatentDataset(
            pet_dir=args.data_root,
            dataset_type=split,
            fold=fold,
            resize_image_size=(128, 128),
            split=split_all)
        return DataLoader(
            ds, batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=4, pin_memory=True
        )

    def run_one_fold(fold: int, base_path: str):
        print(Fore.CYAN + f"\n=== Train Fold {fold} ===" + Style.RESET_ALL)

        if args.model == "sde":
            model = build_lkmunet_sde_film(cfg, strategy=args.strategy).to(DEVICE)
        else:
            model = build_lkmunet_ode_film(cfg, strategy=args.strategy).to(DEVICE)
        apply_gradient_checkpoint(model)

        train_loader = make_loader("train", fold)
        valid_loader = make_loader("val",   fold)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

        warmup_fn    = lambda it: min(1., it / 2_000)
        warmup_sched = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)

        plateau_sched = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10,
            threshold=1e-4, cooldown=5, verbose=True,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_dir  = Path(base_path) / f"LKMUSDE_F{fold}_{timestamp}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_and_validate(
            model, train_loader, valid_loader,
            optimizer, warmup_sched,
            epoch_sched=plateau_sched,
            num_epochs=args.epochs,
            save_path=fold_dir,
            fold=fold,
            acc_steps=ACC_STEPS,
            baseline_csv=args.baseline_csv,
            vis_json_path=args.vis_json,
        )

        torch.cuda.empty_cache(); gc.collect()

    # ── 主程式 ──
    for sd in args.seeds:
        set_seed(sd)
        base_path = str(Path(args.save_dir) / f"runs_seed_LKMUNet_{args.model.upper()}{sd:05d}")
        for f in args.folds:
            run_one_fold(f, base_path)

    print(Fore.GREEN + "All seeds & folds finished!" + Style.RESET_ALL)