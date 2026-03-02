import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import wandb

def train_model(model, train_loader, val_loader, config, device):
    torch.autograd.set_detect_anomaly(True)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate
    )

    os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")

    model.to(device)

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        train_vq = 0

        for batch in tqdm(train_loader):
            mel = batch["mel"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()
            recon, vq = model(mel, label)

            recon_loss = F.mse_loss(recon, mel)
            loss = recon_loss + 0.1 * vq  # scale down VQ contribution 
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            optimizer.step()

            train_loss += recon_loss.item()
            train_vq += vq.item()

        train_loss /= len(train_loader)
        train_vq /= len(train_loader)

        model.eval()
        val_loss = 0
        val_vq = 0

        with torch.no_grad():
            for batch in val_loader:
                mel = batch["mel"].to(device)
                label = batch["label"].to(device)
                recon, vq = model(mel, label)
                val_loss += F.mse_loss(recon, mel).item()
                val_vq += vq.item()

        val_loss /= len(val_loader)
        val_vq /= len(val_loader)

        if config.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/recon_loss": train_loss,
                "train/vq_loss": train_vq,
                "val/recon_loss": val_loss,
                "val/vq_loss": val_vq,
                "lr": optimizer.param_groups[0]["lr"]
            })

        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        torch.save(ckpt, f"checkpoints/epoch_{epoch+1:03d}.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, "checkpoints/best_model.pt")
            if config.use_wandb:
                wandb.save("checkpoints/best_model.pt")
