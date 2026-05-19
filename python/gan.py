import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from model.network import build_model
from model.losses import CombinedLoss, GANLoss
from model.discriminator import build_discriminator
from model.warp import SimpleFlowEstimator
from model.raft import RaftFlow
from data.dataset import build_dataloader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def parse_args():
    parser = argparse.ArgumentParser(description="GAN fine-tuning (Stage 2) for Temporal SR")
    parser.add_argument("--dataset", type=str, default="game",
                        choices=["vimeo90k", "reds", "combined", "game"])
    parser.add_argument("--flow_estimator", type=str, default="raft", choices=["simple", "raft"])
    parser.add_argument("--data_root", type=str, default=None,
                        help="Required for vimeo90k / reds / game. Ignored for combined.")
    parser.add_argument("--vimeo_root", type=str, default=None)
    parser.add_argument("--reds_root", type=str, default=None)
    parser.add_argument("--scale", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--patch_size", type=int, default=128, help="LR patch size")
    parser.add_argument("--mid_channels", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=15)
    parser.add_argument("--num_feat", type=int, default=64,
                        help="Discriminator base channel width")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--gan_checkpoint", type=str, required=True,
                        help="Stage 1 checkpoint (pixel/perceptual-trained) to fine-tune from")
    parser.add_argument("--gen_lr", type=float, default=1e-4)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    parser.add_argument("--gan_weight", type=float, default=0.1,
                        help="Weight on the adversarial term. Lower to ~0.05 if training is unstable.")
    parser.add_argument("--pixel_weight", type=float, default=1.0)
    parser.add_argument("--perceptual_weight", type=float, default=1.0)
    parser.add_argument("--degradation_mode", type=str, default="mild",
                        choices=["none", "mild", "heavy"])
    parser.add_argument("--amp", action="store_true", default=True, help="Mixed precision")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_gan")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume a GAN run from a gan_*.pth checkpoint")
    parser.add_argument("--log_dir", type=str, default="runs_gan")
    parser.add_argument("--val_sample_dir", type=str, default="val_samples",
                        help="Where to dump bicubic|SR|HR comparison images each validation epoch")
    parser.add_argument("--num_val_samples", type=int, default=4,
                        help="How many fixed validation samples to dump per validation epoch")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def estimate_flow_batch(
    flow_estimator: nn.Module,
    lr_curr: torch.Tensor,
    lr_other: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        flow = flow_estimator(lr_curr, lr_other)
    return flow

def train_one_epoch_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    flow_estimator: nn.Module,
    dataloader,
    content_criterion: CombinedLoss,
    gan_criterion: GANLoss,
    g_optimizer: optim.Optimizer,
    d_optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool,
    writer: SummaryWriter,
    global_step: int,
    gan_weight: float,
) -> tuple[dict[str, float], int]:
    generator.train()
    discriminator.train()
    flow_estimator.eval()

    running = {"g_total": 0.0, "g_content": 0.0, "g_adv": 0.0,
               "d_real": 0.0, "d_fake": 0.0}
    num_batches = 0
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"

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

        prev_sr = hr_prev

        with autocast(device_type=amp_device_type, enabled=use_amp):
            sr_output = generator(lr_curr, prev_sr, lr_next, flow_lr_prev, flow_lr_next)

        for p in discriminator.parameters():
            p.requires_grad = True
        d_optimizer.zero_grad()
        with autocast(device_type=amp_device_type, enabled=use_amp):
            real_pred = discriminator(hr_curr)
            fake_pred = discriminator(sr_output.detach())
            d_loss_real = gan_criterion(real_pred, True)
            d_loss_fake = gan_criterion(fake_pred, False)
            d_loss = d_loss_real + d_loss_fake
        scaler.scale(d_loss).backward()
        scaler.step(d_optimizer)

        for p in discriminator.parameters():
            p.requires_grad = False
        g_optimizer.zero_grad()
        with autocast(device_type=amp_device_type, enabled=use_amp):
            content_loss, content_dict = content_criterion(sr_output, hr_curr)
            fake_pred_g = discriminator(sr_output)
            adv_loss = gan_criterion(fake_pred_g, True)
            g_loss = content_loss + gan_weight * adv_loss
        scaler.scale(g_loss).backward()
        scaler.unscale_(g_optimizer)
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        scaler.step(g_optimizer)
        scaler.update()

        running["g_total"] += g_loss.item()
        running["g_content"] += content_loss.item()
        running["g_adv"] += adv_loss.item()
        running["d_real"] += d_loss_real.item()
        running["d_fake"] += d_loss_fake.item()
        num_batches += 1
        global_step += 1

        if global_step % 50 == 0:
            writer.add_scalar("train/g_total", g_loss.item(), global_step)
            writer.add_scalar("train/g_content", content_loss.item(), global_step)
            writer.add_scalar("train/g_adv", adv_loss.item(), global_step)
            writer.add_scalar("train/d_real", d_loss_real.item(), global_step)
            writer.add_scalar("train/d_fake", d_loss_fake.item(), global_step)
            for key, val in content_dict.items():
                writer.add_scalar(f"train/{key}", val, global_step)
            writer.add_scalar("train/gen_lr", g_optimizer.param_groups[0]["lr"], global_step)

        pbar.set_postfix(
            g=f"{g_loss.item():.3f}",
            adv=f"{adv_loss.item():.3f}",
            d=f"{(d_loss_real.item() + d_loss_fake.item()):.3f}",
        )

    n = max(num_batches, 1)
    averaged = {k: v / n for k, v in running.items()}
    return averaged, global_step

@torch.no_grad()
def validate_gan(
    generator: nn.Module,
    flow_estimator: nn.Module,
    dataloader,
    device: torch.device,
    save_dir: str | None = None,
    epoch: int = 0,
    num_save: int = 4,
) -> tuple[float, float]:
    generator.eval()
    flow_estimator.eval()

    running_psnr = 0.0
    running_ssim = 0.0
    n_samples = 0

    saved = 0
    epoch_dir = None
    if save_dir is not None and num_save > 0:
        epoch_dir = Path(save_dir) / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

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

        sr_output = generator(lr_curr, prev_sr, lr_next, flow_lr_prev, flow_lr_next)
        sr_clamped = sr_output.clamp(0, 1)

        if epoch_dir is not None and saved < num_save:
            scale = hr_curr.shape[-1] // lr_curr.shape[-1]
            bicubic = F.interpolate(
                lr_curr, scale_factor=scale, mode="bicubic", align_corners=False
            ).clamp(0, 1)
            for i in range(sr_clamped.shape[0]):
                if saved >= num_save:
                    break
                triptych = torch.cat(
                    [bicubic[i], sr_clamped[i], hr_curr[i]], dim=-1
                )
                save_image(triptych, epoch_dir / f"sample_{saved:02d}.png")
                saved += 1

        sr_np = sr_clamped.cpu().numpy()
        hr_np = hr_curr.cpu().numpy()
        for i in range(sr_np.shape[0]):
            sr_img = sr_np[i].transpose(1, 2, 0)
            hr_img = hr_np[i].transpose(1, 2, 0)
            running_psnr += peak_signal_noise_ratio(hr_img, sr_img, data_range=1.0)
            running_ssim += structural_similarity(
                hr_img, sr_img, data_range=1.0, channel_axis=2
            )
        n_samples += sr_np.shape[0]

    return running_psnr / max(n_samples, 1), running_ssim / max(n_samples, 1)

def save_gan_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: optim.Optimizer,
    d_optimizer: optim.Optimizer,
    g_scheduler,
    d_scheduler,
    epoch: int,
    global_step: int,
    path: Path,
):
    # Generator saved under model_state_dict so video.py, benchmark.py can load GAN without changes
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "g_scheduler_state_dict": g_scheduler.state_dict(),
            "d_scheduler_state_dict": d_scheduler.state_dict(),
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
    assert device.type == "cuda", "no nvidia gpu running or its not setup properly"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    generator = build_model(
        mid_channels=args.mid_channels,
        num_res_blocks=args.num_res_blocks,
        scale=args.scale,
    ).to(device)
    stage1 = torch.load(args.gan_checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" not in stage1:
        raise SystemExit(
            f"{args.gan_checkpoint} has no model_state_dict key — "
            f"this should be a model checkpoint from train.py"
        )
    generator.load_state_dict(stage1["model_state_dict"])
    print(f"Generator: loaded Stage 1 weights from {args.gan_checkpoint}")
    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")

    discriminator = build_discriminator(num_feat=args.num_feat).to(device)
    print(f"disc params {sum(p.numel() for p in discriminator.parameters()):,}")

    if args.flow_estimator == "raft":
        flow_estimator = RaftFlow().to(device)
    else:
        flow_estimator = SimpleFlowEstimator(num_levels=4).to(device)
    flow_estimator.eval()
    for p in flow_estimator.parameters():
        p.requires_grad = False

    content_criterion = CombinedLoss(
        pixel_weight=args.pixel_weight,
        perceptual_weight=args.perceptual_weight,
    ).to(device)
    gan_criterion = GANLoss().to(device)

    g_optimizer = optim.AdamW(generator.parameters(), lr=args.gen_lr, betas=(0.9, 0.99))
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=args.disc_lr, betas=(0.9, 0.99))
    g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=args.epochs, eta_min=1e-7)
    d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    train_loader = build_dataloader(
        args.dataset, args.data_root, args.scale, "train",
        args.batch_size, args.patch_size, args.num_workers,
        use_precomputed_flow=False,
        vimeo_root=args.vimeo_root, reds_root=args.reds_root,
        degradation_mode=args.degradation_mode,
    )
    val_loader = build_dataloader(
        args.dataset, args.data_root, args.scale, "test",
        batch_size=4, patch_size=args.patch_size, num_workers=args.num_workers,
        use_precomputed_flow=False,
        vimeo_root=args.vimeo_root, reds_root=args.reds_root,
        degradation_mode="none",
    )

    start_epoch = 0
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        generator.load_state_dict(ckpt["model_state_dict"])
        discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        g_optimizer.load_state_dict(ckpt["g_optimizer_state_dict"])
        d_optimizer.load_state_dict(ckpt["d_optimizer_state_dict"])
        g_scheduler.load_state_dict(ckpt["g_scheduler_state_dict"])
        d_scheduler.load_state_dict(ckpt["d_scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        print(f"Resumed GAN run from epoch {start_epoch}, step {global_step}")

    print(f"\nStage 2 (GAN) fine-tuning for {args.epochs} epochs")
    print(f"Dataset: {args.dataset}, Scale: {args.scale}x, Batch: {args.batch_size}")
    print(f"gan_weight: {args.gan_weight}, pixel: {args.pixel_weight}, "
          f"perceptual: {args.perceptual_weight}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        stats, global_step = train_one_epoch_gan(
            generator, discriminator, flow_estimator, train_loader,
            content_criterion, gan_criterion, g_optimizer, d_optimizer,
            scaler, device, epoch, args.amp, writer, global_step, args.gan_weight,
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_psnr, val_ssim = validate_gan(
                generator, flow_estimator, val_loader, device,
                save_dir=args.val_sample_dir, epoch=epoch,
                num_save=args.num_val_samples,
            )
            writer.add_scalar("val/psnr", val_psnr, epoch)
            writer.add_scalar("val/ssim", val_ssim, epoch)
            print(
                f"Epoch {epoch:3d} | "
                f"G {stats['g_total']:.4f} (content {stats['g_content']:.4f}, "
                f"adv {stats['g_adv']:.4f}) | "
                f"D real {stats['d_real']:.4f} fake {stats['d_fake']:.4f} | "
                f"PSNR {val_psnr:.2f} dB | SSIM {val_ssim:.4f} | "
                f"Time {time.time() - t0:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch:3d} | "
                f"G {stats['g_total']:.4f} (content {stats['g_content']:.4f}, "
                f"adv {stats['g_adv']:.4f}) | "
                f"D real {stats['d_real']:.4f} fake {stats['d_fake']:.4f} | "
                f"Time {time.time() - t0:.1f}s"
            )

        g_scheduler.step()
        d_scheduler.step()

        save_gan_checkpoint(
            generator, discriminator, g_optimizer, d_optimizer,
            g_scheduler, d_scheduler, epoch, global_step,
            ckpt_dir / "gan_latest.pth",
        )
        if (epoch + 1) % 5 == 0:
            save_gan_checkpoint(
                generator, discriminator, g_optimizer, d_optimizer,
                g_scheduler, d_scheduler, epoch, global_step,
                ckpt_dir / f"gan_epoch_{epoch:03d}.pth",
            )

    save_gan_checkpoint(
        generator, discriminator, g_optimizer, d_optimizer,
        g_scheduler, d_scheduler, args.epochs - 1, global_step,
        ckpt_dir / "gan_final.pth",
    )
    print("psnr ought to be lower, eyeball and pick best model.")
    writer.close()

if __name__ == "__main__":
    main()