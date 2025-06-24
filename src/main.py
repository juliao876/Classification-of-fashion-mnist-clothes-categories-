import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from models.cnn          import EnhancedCNN
from data_loader         import get_dataloaders, get_csv_test_loader
from train               import train_epoch, eval_epoch
from evaluate            import evaluate, plot_confusion
from early_stopping      import EarlyStopping

def ensure_dirs():
    os.makedirs("checkpoints", exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        choices=["train","eval","test"], default="train")
    p.add_argument("--epochs",      type=int,   default=30,
                   help="If --resume, how many NEW epochs to run; else total epochs")
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--resume",      type=str,   default=None)
    p.add_argument("--use_mixup",   action="store_true")
    p.add_argument("--mixup_alpha", type=float, default=0.4)
    p.add_argument("--use_cutmix",  action="store_true")
    p.add_argument("--cutmix_alpha",type=float, default=1.0)
    p.add_argument("--confusion",   action="store_true")
    p.add_argument("--save_name",   type=str,   default="checkpoints/best.pth")
    p.add_argument("--test_csv",    type=str,   default=None,
                   help="Path to CSV for test mode")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # weighted + smoothing
    weights = torch.ones(10, device=device)
    for c in (0,2,3,4,6):
        weights[c] = 1.5
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    # data loaders
    if args.mode in ("train","eval"):
        train_loader, val_loader, test_loader = get_dataloaders(
            args.batch_size, data_dir="data/", val_frac=0.1
        )
    if args.mode == "test":
        if not args.test_csv:
            raise ValueError("--test_csv is required in test mode")
        test_loader = get_csv_test_loader(args.test_csv, batch_size=args.batch_size)

    model = EnhancedCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # OneCycleLR for batch-level scheduling if training
    scheduler = None
    if args.mode == "train":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs if not args.resume else args.epochs
        )

    # resume logic
    start_epoch, best_acc = 1, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("best_acc", 0.0)

    # TRAIN
    if args.mode == "train":
        es = EarlyStopping(patience=args.patience)

        if args.resume:
            end_epoch = start_epoch + args.epochs - 1
        else:
            end_epoch = args.epochs

        for ep in range(start_epoch, end_epoch + 1):
            tr_loss = train_epoch(
                model, train_loader, criterion, optimizer, device,
                scheduler=scheduler,
                use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
                use_cutmix=args.use_cutmix, cutmix_alpha=args.cutmix_alpha
            )
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
            print(f"[Epoch {ep}] TrL:{tr_loss:.4f} ValL:{val_loss:.4f} ValA:{val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "best_acc": best_acc
                }, args.save_name)

            if es(val_loss):
                print("→ early stopping")
                break

    # EVAL or TEST
    if args.mode in ("eval","test"):
        if args.mode == "eval":
            ckpt = torch.load(args.save_name, map_location=device)
            model.load_state_dict(ckpt["model_state"])
        acc, cm = evaluate(model, test_loader, device)
        print(f"Accuracy: {acc*100:.2f}%")
        if args.confusion:
            names = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
                     "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
            plot_confusion(cm, names)

if __name__ == "__main__":
    main()
