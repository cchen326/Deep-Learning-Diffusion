import torch
import os

def load_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None, checkpoint_path='checkpoints/checkpoint.pth'):

    print("loading checkpoint")
    # NOTE: PyTorch 2.6 defaults to `weights_only=True`, which breaks checkpoints that
    # store configuration values (e.g. ruamel.yaml scalar floats). For our own training
    # artifacts it is safe to opt back into the original behaviour.
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    print("loading unet")
    unet.load_state_dict(checkpoint['unet_state_dict'])
    print("loading scheduler")
    scheduler_state = checkpoint.get('scheduler_state_dict', {})
    if isinstance(scheduler_state, dict):
        scheduler_state = dict(scheduler_state)  # shallow copy
        if 'timesteps' in scheduler_state:
            saved = scheduler_state['timesteps']
            current = getattr(scheduler, 'timesteps', None)
            if current is not None and getattr(saved, 'shape', None) != getattr(current, 'shape', None):
                scheduler_state.pop('timesteps')
    scheduler.load_state_dict(scheduler_state, strict=False)

    if vae is not None and 'vae_state_dict' in checkpoint:
        print("loading vae")
        vae.load_state_dict(checkpoint['vae_state_dict'])

    if class_embedder is not None and 'class_embedder_state_dict' in checkpoint:
        print("loading class_embedder")
        class_embedder.load_state_dict(checkpoint['class_embedder_state_dict'])

    # Load optimizer state for resuming training
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        print("loading optimizer")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Get the epoch number for resuming
    start_epoch = checkpoint.get('epoch', 0)

    # Free checkpoint dict from memory
    del checkpoint
    torch.cuda.empty_cache()
    print(f"checkpoint freed from memory (was at epoch {start_epoch})")

    return start_epoch
    
    
        

def save_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None, epoch=None, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define checkpoint file name
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')

    checkpoint = {
        'unet_state_dict': unet.state_dict(),
        'scheduler_state_dict': {
            k: v
            for k, v in scheduler.state_dict().items()
            if k != 'timesteps'
        },
    }
    
    if vae is not None:
        checkpoint['vae_state_dict'] = vae.state_dict()
    
    if class_embedder is not None:
        checkpoint['class_embedder_state_dict'] = class_embedder.state_dict()
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    
    # Manage checkpoint history
    manage_checkpoints(save_dir, keep_last_n=10)


def manage_checkpoints(save_dir, keep_last_n=10):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    if len(checkpoints) > keep_last_n + 1:  # keep_last_n + 1 to account for the latest checkpoint
        for checkpoint_file in checkpoints[:-keep_last_n-1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")
