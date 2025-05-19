import logging
import wandb
import torch

def setup_logger(log_file):
    """
    Set up the logger to save logs to a file and print to the console.
    
    Args:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        # format="%(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Log to file
            logging.StreamHandler()  # Print to console
        ]
    )

def log_arguments(args):
    """
    Log all argument values.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    logging.info("Parsed arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
        
        
def wandb_log_mem():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # to MB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)  # to MB

    # log to WandB
    wandb.log({
        "GPU/Allocated_Memory_MB": allocated_memory,
        "GPU/Reserved_Memory_MB": reserved_memory,
    })