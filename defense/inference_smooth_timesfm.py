import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path + '/models/moirai_torch')
sys.path.append('./')

from AttackTSFM.utils.utils import set_seed
from AttackTSFM.data.data_module import DataModule
from AttackTSFM.data.data_manager import DataManager
from AttackTSFM.utils.log import setup_logger
from AttackTSFM.utils.evaluator import Evaluator
from AttackTSFM.attack import PGD, SimBA

from models import TimesFm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def causal_moving_average(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Apply a causal moving average along the last dimension.
    x: [B, L] or [B, K, L]
    k: kernel size (>=1); k=1 is an identity mapping.
    """
    if k <= 1:
        return x
    if x.ndim == 2:
        # [B, L] -> [B, 1, L]
        x2 = x.unsqueeze(1)
        # Left pad (k-1) to keep causality; replicate at the boundary
        x2 = F.pad(x2, (k - 1, 0), mode='replicate')
        w = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / k
        y = F.conv1d(x2, w, stride=1, padding=0)  # [B, 1, L]
        return y.squeeze(1)
    elif x.ndim == 3:
        # [B, K, L] -> treat K as channel dimension
        x2 = x
        x2 = F.pad(x2, (k - 1, 0), mode='replicate')  # [B, K, L+(k-1)]
        # Grouped conv: smooth each variable channel independently
        B, C, _ = x2.shape
        w = torch.ones(C, 1, k, device=x.device, dtype=x.dtype) / k
        y = F.conv1d(x2, w, stride=1, padding=0, groups=C)  # [B, C, L]
        return y
    else:
        raise ValueError(f"Unsupported input ndim={x.ndim}, expect 2 or 3.")


class SmoothWrapper(torch.nn.Module):
    """
    Apply causal moving average to the past (history window) BEFORE forward,
    then call the underlying model. Only modifies the past; time features are
    kept unchanged. Also implements .loss to be compatible with PGD.
    """
    def __init__(self, base_model: TimesFm, kernel_size: int):
        super().__init__()
        self.base = base_model
        self.kernel_size = int(kernel_size)
        # (Optional) expose common attributes to avoid attribute errors when accessed externally
        for attr in ["context_len", "horizon_len", "input_patch_len",
                     "output_patch_len", "num_outputs", "tolerance",
                     "pad_val"]:
            if hasattr(base_model, attr):
                setattr(self, attr, getattr(base_model, attr))

    def forward(self, past_tensor: torch.Tensor, **kwargs):
        x = past_tensor
        if x.ndim == 3:
            x = rearrange(x, 'b l k -> b k l')
            x = causal_moving_average(x, self.kernel_size)
            x = rearrange(x, 'b k l -> b l k')
        else:
            x = causal_moving_average(x, self.kernel_size)
        # Call the underlying TimesFm
        return self.base(x, **kwargs)

    def loss(self, past_target: torch.Tensor, future_target: torch.Tensor):
        """
        Keep the same signature/behavior as TimesFm.loss:
        - Smooth the input past_target via forward()
        - future_target can be [B, H] or [B, L, H] (here multivariate is [B, L, K]),
          and we keep the same handling as in TimesFm.loss
        """
        # Smooth through forward and obtain mean forecasts
        mean_forecasts, _ = self.forward(past_target)
        forecasts = mean_forecasts
        # Same as TimesFm.loss: for multivariate, convert [B, K, L] -> [B, L, K]
        if future_target.ndim == 3:
            forecasts = rearrange(forecasts, 'b k l -> b l k')
        return torch.mean((forecasts - future_target) ** 2)


def parse_args():
    p = argparse.ArgumentParser("Inference-time smoothing for TimesFm (clean/PGD/SimBA eval)")

    # Data & model
    p.add_argument("--dataset", type=str, default="etth1")
    p.add_argument("--dataset_path", type=str, default="./datasets")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--target_dim", type=int, default=None)
    p.add_argument("--context_length", type=int, default=None)
    p.add_argument("--prediction_length", type=int, default=None)
    p.add_argument("--model_size", type=str, default="200m")  # 200m / 500m

    # Smoothing configs
    p.add_argument("--kernels", type=str, default="1,3,5,7",
                   help="comma-separated kernel sizes; 1 means no smoothing (baseline)")

    # Eval / logging
    p.add_argument("--save_dir", type=str, default="./smooth_eval_logs")
    p.add_argument("--log", type=str, default="inference_smooth_timesfm.log")
    p.add_argument("--seed", type=int, default=42)
    
    p.add_argument("--gift_eval", type=lambda x: x.lower() in ['true','1','yes'])

    # Adversarial evaluation (same convention as main.py)
    p.add_argument("--robust_eval", type=lambda x: x.lower() in ['true','1','yes'],
                   default=True, help="run clean/PGD/SimBA robustness eval")
    p.add_argument("--attack_gt", type=lambda x: x.lower() in ['true','1','yes'],
                   default=True, help="untargeted w.r.t. ground-truth if True, else to clean pred")
    p.add_argument("--attack_ratio", type=float, default=1.0,
                   help="fraction of context steps to perturb")
    p.add_argument("--epsilon", type=float, default=0.5,
                   help="PGD/SimBA perturbation bound (L_inf)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="PGD step size")
    p.add_argument("--num_iterations", type=int, default=50,
                   help="PGD/SimBA iterations (or max flips)")
    p.add_argument("--simba_mode", type=str, default="random",
                   help="SimBA mode: ['random','coordinate']")
    p.add_argument("--simba_basis", type=str, default="point",
                   help="SimBA basis: ['point','dct','wavelet']")
    p.add_argument("--norm", type=str, default="linf",
                   help="PGD/FGSM norm: ['linf','l2']")

    # Switch between test/valid split (default: test)
    p.add_argument("--use_test_split", type=lambda x: x.lower() in ['true','1','yes'],
                   default=True, help="use test_dataloader (True) or valid_dataloader (False)")

    return p.parse_args()


def evaluate_robustness_with_smoothing(model_wrapped: torch.nn.Module,
                                       dm: DataModule,
                                       evaluator: Evaluator,
                                       args):
    """
    Evaluate the given "smoothing + model" combo on clean / PGD / SimBA.
    Note: enable gradients for PGD; clean and post-attack forwards can be no_grad.
    """
    model_wrapped.eval()
    freq = dm.data_manager.freq
    ctx_len = dm.data_manager.context_length
    point_num = int(args.attack_ratio * ctx_len)

    buf = {
        'clean': {'inp': [], 'gt': [], 'q': []},
        'pgd':   {'inp': [], 'gt': [], 'q': []},
        'simba': {'inp': [], 'gt': [], 'q': []},
    }

    loader = dm.test_dataloader() if args.use_test_split else dm.valid_dataloader()

    for batch in loader:
        past = batch['past_target_cdf'].to(device)   # [B, L] or [B, L, K]
        fut  = batch['future_target_cdf'].to(device) # [B, H] or [B, H, K]
        if past.ndim == 3:                           # flatten channel dimension into batch
            past = rearrange(past, 'b l k -> (b k) l')
            fut  = rearrange(fut,  'b l k -> (b k) l')

        # ---- clean ----
        with torch.no_grad():
            mean_clean, q_clean = model_wrapped(past)
        buf['clean']['inp'].append(past.detach().cpu())
        buf['clean']['gt'].append(fut.detach().cpu())
        buf['clean']['q'].append(q_clean.detach().cpu())

        # Attack target
        attack_target = fut.detach() if args.attack_gt else mean_clean.detach()

        if point_num > 0 and args.robust_eval:
            # ---- PGD ---- (requires gradients)
            with torch.enable_grad():
                model_wrapped.zero_grad(set_to_none=True)
                pgd = PGD(model_wrapped, alpha=args.alpha, targeted=False, norm=args.norm)
                pert_past_pgd = pgd.attack_batch(
                    past, attack_target,
                    n_dims=point_num, max_iters=args.num_iterations,
                    epsilon=args.epsilon,
                    past_datastamps=batch.get('past_time_feat', None),
                    future_datastamps=batch.get('future_time_feat', None)
                )
            with torch.no_grad():
                _, q_pgd = model_wrapped(pert_past_pgd)
            buf['pgd']['inp'].append(past.detach().cpu())
            buf['pgd']['gt'].append(fut.detach().cpu())
            buf['pgd']['q'].append(q_pgd.detach().cpu())

            # ---- SimBA ---- (black-box)
            simba = SimBA(model_wrapped, targeted=False,
                          mode=args.simba_mode, basis=args.simba_basis)
            max_iters = int(min(point_num, args.num_iterations))
            pert_past_simba = simba.attack_batch(
                past, attack_target,
                n_dims=point_num, max_iters=max_iters,
                epsilon=args.epsilon,
                past_datastamps=batch.get('past_time_feat', None),
                future_datastamps=batch.get('future_time_feat', None)
            )
            with torch.no_grad():
                _, q_simba = model_wrapped(pert_past_simba)
            buf['simba']['inp'].append(past.detach().cpu())
            buf['simba']['gt'].append(fut.detach().cpu())
            buf['simba']['q'].append(q_simba.detach().cpu())
        else:
            # If not attacking, reuse clean results for pgd/simba buckets
            for key in ['pgd', 'simba']:
                buf[key]['inp'].append(past.detach().cpu())
                buf[key]['gt'].append(fut.detach().cpu())
                buf[key]['q'].append(q_clean.detach().cpu())

    # Aggregate metrics
    out = {}
    for k in ['clean', 'pgd', 'simba']:
        metrics = evaluator(
            torch.cat(buf[k]['gt'],  0),
            torch.cat(buf[k]['q'],   0),
            torch.cat(buf[k]['inp'], 0),
            freq=freq
        )
        out[k] = metrics
    return out


def main(args):
    set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    setup_logger(os.path.join(args.save_dir, args.log))
    logging.info(vars(args))

    # Data
    dmgr = DataManager(
        dataset=args.dataset,
        path=args.dataset_path,
        history_length=args.context_length,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        scaler='standard',
        timeenc=2,
        target_dim=args.target_dim,
        gift_eval=args.gift_eval,
    )
    data = DataModule(dmgr, batch_size=args.batch_size,
                      test_batch_size=args.batch_size,
                      num_workers=args.num_workers)

    # Model
    base_model = TimesFm(
        context_len=dmgr.context_length,
        input_patch_len=32,
        output_patch_len=128,
        model_size=args.model_size,
        horizon_len=dmgr.prediction_length,
        ckpt_path=abs_path,
        device=device,
    ).to(device)
    base_model.eval()

    evaluator = Evaluator()

    # Parse kernel list
    kernels = [int(s.strip()) for s in args.kernels.split(',') if s.strip()]
    # De-duplicate & sort (ensure 1 appears first as the baseline)
    kernels = sorted(set(kernels))
    if 1 not in kernels:
        kernels = [1] + kernels

    # Evaluate each kernel
    def _fmt(m):  # compact printer
        return f"ND={m['ND']:.4f}  NRMSE={m['NRMSE']:.4f}  CRPS={m['CRPS']:.4f}"

    baseline_metrics = None
    for ksz in kernels:
        model_wrapped = SmoothWrapper(base_model, kernel_size=ksz).to(device)
        results = evaluate_robustness_with_smoothing(model_wrapped, data, evaluator, args)

        if ksz == 1:
            baseline_metrics = results
            logging.info(f"[SMOOTH k={ksz}] (Baseline, no smoothing)")
        else:
            logging.info(f"[SMOOTH k={ksz}]")

        # Print clean/PGD/SimBA
        logging.info("  clean: " + _fmt(results['clean']))
        logging.info("  PGD  : " + _fmt(results['pgd']))
        logging.info("  SimBA: " + _fmt(results['simba']))

        # If a baseline exists, also print deltas
        if baseline_metrics is not None and ksz != 1:
            logging.info("  vs baseline (k=1):")
            logging.info("    Δclean: " +
                         f"ΔND={results['clean']['ND']-baseline_metrics['clean']['ND']:+.4f}  "
                         f"ΔNRMSE={results['clean']['NRMSE']-baseline_metrics['clean']['NRMSE']:+.4f}  "
                         f"ΔCRPS={results['clean']['CRPS']-baseline_metrics['clean']['CRPS']:+.4f}")
            logging.info("    ΔPGD  : " +
                         f"ΔND={results['pgd']['ND']-baseline_metrics['pgd']['ND']:+.4f}  "
                         f"ΔNRMSE={results['pgd']['NRMSE']-baseline_metrics['pgd']['NRMSE']:+.4f}  "
                         f"ΔCRPS={results['pgd']['CRPS']-baseline_metrics['pgd']['CRPS']:+.4f}")
            logging.info("    ΔSimBA: " +
                         f"ΔND={results['simba']['ND']-baseline_metrics['simba']['ND']:+.4f}  "
                         f"ΔNRMSE={results['simba']['NRMSE']-baseline_metrics['simba']['NRMSE']:+.4f}  "
                         f"ΔCRPS={results['simba']['CRPS']-baseline_metrics['simba']['CRPS']:+.4f}")

    logging.info("Done smoothing evaluation.")


if __name__ == "__main__":
    args = parse_args()
    main(args)