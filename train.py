import os
from pathlib import Path
import argparse
import logging
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import timeit

from utils.utils import setup_seed, init_weight
from utils.loss import ProbOhemCrossEntropy2d
from utils.lr_scheduler import WarmupPolyLR
from utils.metric import ConfusionMatrix
from model.builder import build_model
from dataset.builder import build_dataset_train


def validate(val_loader, model, criterion, writer, epoch):
    """
    Validation loop to evaluate the model on the validation dataset.
    """
    model.eval()
    total_loss = 0.0
    metric = ConfusionMatrix(num_classes=val_loader.dataset.num_classes)
    with torch.no_grad():
        for i, (images, targets, _, _) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            
            targets = targets.cpu()
            pred = outputs.argmax(dim=1).cpu()
            metric.add_batch(targets, pred)

    average_loss = total_loss / len(val_loader)
    logging.info(f"Validation Epoch [{epoch}], Average Loss: {average_loss:.4f}")
    writer.add_scalar('val/loss', average_loss, epoch)
    
    mean_iou, per_class_iou, _ = metric.jaccard()
    logging.info(f"Validation Epoch [{epoch}], Mean IoU: {mean_iou:.4f}")
    writer.add_scalar('val/mean_iou', mean_iou, epoch)
    for i, iou in enumerate(per_class_iou):
        writer.add_scalar(f'val/class_{i}_iou', iou, epoch)
    
    return average_loss



def train(train_loader, model, criterion, optimizer, scheduler, writer, epoch):
    """
    Main training loop.
    """
    model.train()
    total_loss = 0.0
    metric = ConfusionMatrix(num_classes=train_loader.dataset.num_classes)
    for i, (images, targets, _, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        
        targets = targets.cpu()
        pred = outputs.argmax(dim=1).cpu()
        metric.add_batch(targets, pred)

    average_loss = total_loss / len(train_loader)
    logging.info(f"Epoch [{epoch}], Average Loss: {average_loss:.4f}")
    writer.add_scalar('train/loss', average_loss, epoch)
    
    mean_iou, per_class_iou, _ = metric.jaccard()
    logging.info(f"Epoch [{epoch}], Mean IoU: {mean_iou:.4f}")
    writer.add_scalar('train/mean_iou', mean_iou, epoch)
    for i, iou in enumerate(per_class_iou):
        writer.add_scalar(f'train/class_{i}_iou', iou, epoch)
    
    return average_loss
    


def fit(args):
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    logging.info(f"=====> Input size: {input_size}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_count = torch.cuda.device_count()
    logging.info(f"=====> Found {device_count} GPU(s) available.")
    if device_count > 0:
        device_list = ','.join(str(i) for i in range(device_count))
        os.environ['CUDA_VISIBLE_DEVICES'] = device_list
        logging.info(f"=====> Using GPU IDs: {device_list}")
    else:
        logging.info("=====> No CUDA devices available, using CPU.")
    
    # Set up Seed
    setup_seed(args.seed)
    logging.info(f"=====> Seed: {args.seed}")
    
    # Set up dataset
    data_stat, train_loader, val_loader = build_dataset_train(
        args.dataset, input_size, args.batch_size, args.num_workers, 
        args.scale, args.mirror
    )
    weight = torch.from_numpy(data_stat['class_weights'])
    logging.info(f"=====> Dataset: {args.dataset}")
    logging.info(f"Class weights: {weight}")
    logging.info(f"Mean: {data_stat['mean']}, Std: {data_stat['std']}")
    
    # Set up model
    cudnn.enabled = True
    logging.info("=====> Building Network")
    model = build_model(args.model, data_stat['num_classes'])
    model = model.to(device)
    init_weight(model, nn.init.kaiming_normal_, nn.BatchNorm2d, 
                bn_eps=1e-3, bn_momentum=0.1, mode='fan_in')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"=====> Total parameters: {total_params / 1e6:.2f}M")
    if device_count > 1:
        model = nn.DataParallel(model)
    else:
        model = model.cuda()
    
    # Set up loss function
    div = device_count if device_count > 0 else 1
    min_kept = int(args.batch_size // div * h * w * 16)
    criterion = ProbOhemCrossEntropy2d(weight=weight, ignore_index=data_stat['ignore_index'], 
                                       thresh=0.7, min_kept=min_kept)
    criterion = criterion.to(device)
    
    # Set up optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 args.lr, weight_decay=1e-5)
    scheduler = WarmupPolyLR(optimizer, T_max=args.max_iter, warmup_factor=1.0 / 3,
                             warmup_iters=100, power=0.9)
    
    # Set up TensorBoard
    log_dir = Path(args.log_dir) / args.dataset / args.model
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Set up transfer learning if specified
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.is_file():
            logging.info(f"=====> Load checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.start_epoch = checkpoint['epoch']
            logging.info(f"=====> Resuming from epoch {args.start_epoch}")
        else:
            logging.error(f"=====> Checkpoint file not found: {checkpoint_path}")
    elif args.pretrained:
        pretrained_path = Path(args.pretrained)
        if pretrained_path.is_file():
            logging.info(f"=====> Load pretrained model from {pretrained_path}")
            pretrained_state = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(pretrained_state['model'])
        else:
            logging.error(f"=====> Pretrained model file not found: {pretrained_path}")
    
    # Training loop
    try:
        train_loss_list = []
        for epoch in range(args.max_epochs):
            train_loss = train(train_loader, model, criterion, optimizer, scheduler, writer, epoch)
            train_loss_list.append(train_loss)
            
            if (epoch + 1) % args.val_interval == 0:
                val_loss = validate(args, val_loader, model, criterion, writer, epoch)
                logging.info(f"Epoch [{epoch + 1}/{args.max_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save final model
            last_model_path = log_dir / f"model_{epoch - 1}.pth"
            try:
                last_model_path.unlink()
            except FileNotFoundError:
                pass
            
            final_model_path = log_dir / f"model_{epoch}.pth"
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': args.max_epochs,
                'train_loss': train_loss_list
            }, final_model_path)
            logging.info(f"=====> Final model saved to {final_model_path}")
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
    finally:
        # Close TensorBoard writer
        logging.info("=====> Closing TensorBoard writer")
        writer.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument('--dataset', type=str, default='cityscapes', help='Dataset name')
    parser.add_argument('--model', type=str, default='FBSNet', help='Model name')
    parser.add_argument('--input_size', type=str, default='512,1024', help='Input size (height,width)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=4.5e-2, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval in epochs')
    parser.add_argument('--scale', type=bool, default=True, help='Enable random scaling')
    parser.add_argument('--mirror', type=bool, default=True, help='Enable random mirroring')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='Directory to save logs and models')
    parser.add_argument('--resume', type=bool, default=False, help='Path to resume from a checkpoint')
    parser.add_argument('--pretrained', type=bool, default=False, help='Path to a pretrained model')
    
    args = parser.parse_args()
    args.max_iter = args.max_epochs * args.batch_size
    
    logging.basicConfig(level=logging.INFO)
    
    start = timeit.default_timer()
    fit(args)
    end = timeit.default_timer()
    hour = (end - start) // 3600
    minute = ((end - start) % 3600) // 60
    second = (end - start) % 60
    logging.info(f"Training completed in {int(hour)}h {int(minute)}m {int(second)}s")