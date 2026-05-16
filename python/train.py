import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.network import build_model
from model.losses import CombinedLoss
from model.warp import SimpleFlowEstimator
from model.raft import RaftFlow
from data.dataset import build_dataloader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def parse_args():
    parser = argparse.ArgumentParser(description="Train Temporal SR Network")
    parser.add_argument("--dataset", type=str, default="vimeo90k", choices=["vimeo90k", "reds", "combined"])
    parser.add_argument("--flow_estimator", type=str, default="raft", choices=["simple", "raft"])
    parser.add_argument("--data_root", type=str, default=None,
                        help="Required for --dataset vimeo90k or reds. Ignored for combined.")
    parser.add_argument("--vimeo_root", type=str, default=None,
                        help="Path to Vimeo-90K root (for --dataset combined)")
    parser.add_argument("--reds_root", type=str, default=None,
                        help="Path to REDS root (for --dataset combined)")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--patch_size", type=int, default=128, help="LR patch size")
    parser.add_argument("--mid_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--pixel_weight", type=float, default=1.0)
    parser.add_argument("--perceptual_weight", type=float, default=0.01)
    parser.add_argument("--scheduled_sampling_prob", type=float, default=0.3,
                        help="Probability of injecting noise into hr_prev to mitigate exposure bias")
    parser.add_argument("--scheduled_sampling_std", type=float, default=0.02,
                        help="Std of noise injected into hr_prev (in [0,1] pixel space)")
    parser.add_argument("--use_precomputed_flow", action="store_true",
                        help="Load precomputed RAFT flows from disk instead of computing on-the-fly. "
                             "Run precompute.py first.")
    parser.add_argument("--degradation_mode", type=str, default="none",
                        choices=["none", "mild", "heavy"],
                        help="LR degradation pipeline (Real-ESRGAN style). Only applied at training time.")
    parser.add_argument("--amp", action="store_true", default=True, help="Mixed precision")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cuda', 'cpu', or 'privateuseone' (DirectML)")
    return parser.parse_args()

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def estimate_flow_batch(
    flow_estimator: nn.Module,
    lr_curr: torch.Tensor,
    lr_prev: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        flow = flow_estimator(lr_curr, lr_prev)
    return flow

def train_one_epoch(
    model: nn.Module,
    flow_estimator: nn.Module,
    dataloader,
    criterion: CombinedLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool,
    writer: SummaryWriter,
    global_step: int,
    sched_sampling_prob: float = 0.0,
    sched_sampling_std: float = 0.02,
) -> tuple[float, int]:
    model.train()
    flow_estimator.eval()  

    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        lr_curr = batch["lr_curr"].to(device)
        lr_prev = batch["lr_prev"].to(device)
        lr_next = batch["lr_next"].to(device)
        hr_curr = batch["hr_curr"].to(device)
        hr_prev = batch["hr_prev"].to(device)

        if "flow_lr_prev" in batch:
            flow_lr_prev = batch["flow_lr_prev"].to(device)
            flow_lr_next = batch["flow_lr_next"].to(device)
        else:
            flow_lr_prev = estimate_flow_batch(flow_estimator, lr_curr, lr_prev)
            flow_lr_next = estimate_flow_batch(flow_estimator, lr_curr, lr_next)

        if sched_sampling_prob > 0 and torch.rand(1).item() < sched_sampling_prob:
            noise = torch.randn_like(hr_prev) * sched_sampling_std
            prev_sr = (hr_prev + noise).clamp(0, 1)
        else:
            prev_sr = hr_prev

        optimizer.zero_grad()

        amp_device_type = "cuda" if device.type == "cuda" else "cpu"
        with autocast(device_type=amp_device_type, enabled=use_amp):
            sr_output = model(lr_curr, prev_sr, lr_next, flow_lr_prev, flow_lr_next)
            loss, loss_dict = criterion(sr_output, hr_curr)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss_dict["total_loss"]
        num_batches += 1
        global_step += 1

        if global_step % 50 == 0:
            for key, val in loss_dict.items():
                writer.add_scalar(f"train/{key}", val, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        pbar.set_postfix(loss=f"{loss_dict['total_loss']:.4f}")

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss, global_step

@torch.no_grad()
def validate(
    model: nn.Module,
    flow_estimator: nn.Module,
    dataloader,
    criterion: CombinedLoss,
    device: torch.device,
) -> tuple[float, float, float]:

    model.eval()
    flow_estimator.eval()

    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    num_batches = 0
    n_samples = 0

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        lr_curr = batch["lr_curr"].to(device)
        lr_prev = batch["lr_prev"].to(device)
        lr_next = batch["lr_next"].to(device)
        hr_curr = batch["hr_curr"].to(device)
        hr_prev = batch["hr_prev"].to(device)

        if "flow_lr_prev" in batch:
            flow_lr_prev = batch["flow_lr_prev"].to(device)
            flow_lr_next = batch["flow_lr_next"].to(device)
        else:
            flow_lr_prev = estimate_flow_batch(flow_estimator, lr_curr, lr_prev)
            flow_lr_next = estimate_flow_batch(flow_estimator, lr_curr, lr_next)
        prev_sr = hr_prev

        sr_output = model(lr_curr, prev_sr, lr_next, flow_lr_prev, flow_lr_next)
        loss, _ = criterion(sr_output, hr_curr)

        running_loss += loss.item()

        sr_np = sr_output.clamp(0, 1).cpu().numpy()
        hr_np = hr_curr.cpu().numpy()

        for i in range(sr_np.shape[0]):
            sr_img = sr_np[i].transpose(1, 2, 0) 
            hr_img = hr_np[i].transpose(1, 2, 0)
            running_psnr += peak_signal_noise_ratio(hr_img, sr_img, data_range=1.0)
            running_ssim += structural_similarity(
                hr_img, sr_img, data_range=1.0, channel_axis=2
            )

        num_batches += 1
        n_samples += sr_np.shape[0]

    return (
        running_loss / max(num_batches, 1),
        running_psnr / max(n_samples, 1),
        running_ssim / max(n_samples, 1),
    )

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_psnr: float,
    path: Path,
):
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_psnr": best_psnr,
        },
        path,
    )

def main():
    args = parse_args()

    if args.dataset == "combined":
        if not args.vimeo_root or not args.reds_root:
            raise SystemExit("--dataset combined requires both --vimeo_root and --reds_root")
    else:
        if not args.data_root:
            raise SystemExit(f"--dataset {args.dataset} requires --data_root")

    torch.backends.cudnn.benchmark = True
    device = get_device(args.device)
    assert device.type == "cuda", f"no nvidia gpu running or its not setup properly"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    model = build_model(
        mid_channels=args.mid_channels,
        num_res_blocks=args.num_res_blocks,
        scale=args.scale,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    if args.use_precomputed_flow:
        print("precomputed raft")
        flow_estimator = nn.Identity() 
    elif args.flow_estimator == "raft":
        flow_estimator = RaftFlow().to(device)
    else:
        flow_estimator = SimpleFlowEstimator(num_levels=4).to(device)

    flow_estimator.eval()
    for p in flow_estimator.parameters():
        p.requires_grad = False

    criterion = CombinedLoss(
        pixel_weight=args.pixel_weight,
        perceptual_weight=args.perceptual_weight,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    train_loader = build_dataloader(
        args.dataset, args.data_root, args.scale, "train",
        args.batch_size, args.patch_size, args.num_workers,
        use_precomputed_flow=args.use_precomputed_flow,
        vimeo_root=args.vimeo_root, reds_root=args.reds_root,
        degradation_mode=args.degradation_mode,
    )
    val_loader = build_dataloader(
        args.dataset, args.data_root, args.scale, "test",
        batch_size=4, patch_size=args.patch_size, num_workers=args.num_workers,
        use_precomputed_flow=args.use_precomputed_flow,
        vimeo_root=args.vimeo_root, reds_root=args.reds_root,
        degradation_mode="none",
    )

    start_epoch = 0
    global_step = 0
    best_psnr = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_psnr = ckpt["best_psnr"]
        print(f"Resumed from epoch {start_epoch}, step {global_step}, best PSNR: {best_psnr:.2f}")

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Dataset: {args.dataset}, Scale: {args.scale}x, Batch: {args.batch_size}")
    print(f"LR patch: {args.patch_size}, HR patch: {args.patch_size * args.scale}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, global_step = train_one_epoch(
            model, flow_estimator, train_loader, criterion, optimizer, scaler,
            device, epoch, args.amp, writer, global_step,
            sched_sampling_prob=args.scheduled_sampling_prob,
            sched_sampling_std=args.scheduled_sampling_std,
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_loss, val_psnr, val_ssim = validate(
                model, flow_estimator, val_loader, criterion, device,
            )
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/psnr", val_psnr, epoch)
            writer.add_scalar("val/ssim", val_ssim, epoch)

            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"PSNR: {val_psnr:.2f} dB | "
                f"SSIM: {val_ssim:.4f} | "
                f"Time: {time.time() - t0:.1f}s"
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step, best_psnr,
                    ckpt_dir / "best_model.pth",
                )
                print(f"new highest psnr: {best_psnr:.2f} db")
        else:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Time: {time.time() - t0:.1f}s"
            )

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, best_psnr,
                ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth",
            )

    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, global_step, best_psnr,
        ckpt_dir / "final_model.pth",
    )
    print(f"highest psnr: {best_psnr:.2f} db")
    writer.close()

if __name__ == "__main__":
    main()