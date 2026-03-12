import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader
from torchcfm import ConditionalFlowMatcher

from data import ActionStepDataset
from models import ActionVectorField


DEFAULTS = dict(
    model="action_mlp",
    epochs=100,
    batch_size=256,
    lr=3e-4,
    sigma=0.0,
    save_dir="checkpoints_action",
    log_every=100,
    hidden_dims=[512, 512, 512],
    time_embed_dim=64,
    activation="silu",
    data_root=None,
    env_robot="EnvEmptyNoWait2D-RobotCompositeTwoPlanarDisk",
    device="cuda" if torch.cuda.is_available() else "cpu",
)


def load_config() -> SimpleNamespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=None, help="YAML config file")
    for key, default in DEFAULTS.items():
        flag = f"--{key}"
        if isinstance(default, list):
            parser.add_argument(flag, nargs="+", type=type(default[0]), default=None)
        elif isinstance(default, bool):
            parser.add_argument(flag, type=lambda v: v.lower() in ("true", "1", "yes"), default=None)
        elif default is None:
            parser.add_argument(flag, type=str, default=None)
        else:
            parser.add_argument(flag, type=type(default), default=None)
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))
    for key in DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return SimpleNamespace(**cfg)


def main():
    cfg = load_config()
    root = Path(__file__).resolve().parent
    data_root = Path(cfg.data_root) if cfg.data_root else (root / "data_trajectories")
    device = torch.device(cfg.device)

    ds = ActionStepDataset(data_root, env_robot=cfg.env_robot)

    print(f"Dataset: {len(ds)} step-samples  |  "
          f"state_dim={ds.state_dim}  action_dim={ds.action_dim}  "
          f"n_disks={ds.n_disks}  condition_dim={ds.condition_dim}")

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = ActionVectorField(
        action_dim=ds.action_dim,
        condition_dim=ds.condition_dim,
        time_embed_dim=cfg.time_embed_dim,
        hidden_dims=tuple(cfg.hidden_dims),
        activation=cfg.activation,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ActionVectorField  |  {n_params:,} parameters")

    cfm = ConditionalFlowMatcher(sigma=cfg.sigma)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    save_dir = root / cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_state, batch_goal, batch_action, batch_gf in loader:
            batch_state = batch_state.to(device)
            batch_goal = batch_goal.to(device)
            batch_action = batch_action.to(device)
            batch_gf = batch_gf.to(device)

            x1 = batch_action
            x0 = torch.randn_like(x1)
            t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)

            c = torch.cat([batch_state, batch_goal, batch_gf], dim=-1)
            v = model(xt, t, c)

            loss = (v - ut).pow(2).mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            if global_step % cfg.log_every == 0:
                print(f"  step {global_step:>6d}  loss {loss.item():.6f}")

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        print(f"epoch {epoch + 1}/{cfg.epochs}  avg_loss {avg:.6f}  lr {scheduler.get_last_lr()[0]:.2e}")

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "action_dim": ds.action_dim,
        "condition_dim": ds.condition_dim,
        "state_dim": ds.state_dim,
        "pos_dim": ds.pos_dim,
        "n_disks": ds.n_disks,
        "hidden_dims": list(cfg.hidden_dims),
        "time_embed_dim": cfg.time_embed_dim,
        "activation": cfg.activation,
        "model_type": "action_mlp",
    }
    ckpt_path = save_dir / "action_checkpoint.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
