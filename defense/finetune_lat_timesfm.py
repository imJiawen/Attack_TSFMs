import os
import sys
import time
import argparse
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from einops import rearrange, repeat

# --- keep search paths consistent with main.py ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path + '/models/moirai_torch')
sys.path.append('./')

from AttackTSFM.utils.utils import set_seed
from AttackTSFM.data.data_module import DataModule
from AttackTSFM.data.data_manager import DataManager
from AttackTSFM.utils.log import setup_logger
from AttackTSFM.utils.evaluator import Evaluator

from tqdm import trange
from models import TimesFm
from AttackTSFM.attack import PGD, SimBA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    p = argparse.ArgumentParser("LAT finetune for TimesFm (Algorithm 2)")

    # Data & model
    p.add_argument("--dataset", type=str, default="etth1")
    p.add_argument("--dataset_path", type=str, default="./datasets")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--target_dim", type=int, default=None)
    p.add_argument("--context_length", type=int, default=None)
    p.add_argument("--prediction_length", type=int, default=None)
    p.add_argument("--model_size", type=str, default="200m")  # 200m / 500m
    p.add_argument("--max_iter", type=int, default=100)

    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # LAT (threat model: δ^h in latent space)
    p.add_argument("--inner_steps", type=int, default=3, help="T_delta: inner ascent steps")
    p.add_argument("--eta_delta", type=float, default=0.5, help="η_δ: ascent step size")
    p.add_argument("--eps", type=float, default=1.0, help="ε: Lp-ball radius (latent)")
    p.add_argument("--p_norm", type=str, default="linf", choices=["linf", "l2"])
    p.add_argument("--std_normalize", type=lambda x: x.lower() in ['true','1','yes'],
                   default=True, help="normalize δ^h by latent std before L∞ bound")
    p.add_argument("--attack_last_patch_only", type=lambda x: x.lower() in ['true','1','yes'],
                   default=True, help="attack only the last latent patch")
    
    p.add_argument("--gift_eval", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="use gift_eval datasets.")

    # Eval / saving
    p.add_argument("--save_dir", type=str, default="./lat_timesfm_ckpt")
    p.add_argument("--log", type=str, default="finetune_lat_timesfm.log")
    p.add_argument("--eval_every", type=int, default=1)
    
    # Robustness evaluation
    p.add_argument("--robust_eval", type=lambda x: x.lower() in ['true','1','yes'],
               default=True, help="run clean/PGD/SimBA robustness eval")
    # Attack configs (same style as main.py)
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

    return p.parse_args()

def project_lp(delta, eps, norm="linf"):
    if norm == "linf":
        return delta.clamp_(-eps, eps)
    elif norm == "l2":
        flat = delta.view(delta.size(0), -1)
        nrm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        scale = (eps / nrm).clamp(max=1.0)
        flat = flat * scale
        return flat.view_as(delta)
    else:
        raise ValueError(f"Unknown norm {norm}")

@torch.no_grad()
def get_latent_once(model: TimesFm, inputs: torch.Tensor):
    """
    Return stats and h (= model_output).
    h shape: B x N x D (N = number of patches).
    """
    input_ts, input_padding = model._preprocess(inputs, horizon_len=model.output_patch_len)
    
    final_out = input_ts
    max_len = input_ts.shape[1]
    current_padding = input_padding[:, 0:final_out.shape[1]]
    input_ts = final_out[:, -max_len:]
    input_padding = current_padding[:, -max_len:]

    model_input, patched_padding, stats, _ = model._preprocess_input(input_ts, input_padding)

    bs = inputs.shape[0]
    f_emb = model.freq_emb(torch.tensor([0] * bs).unsqueeze(-1).long()).to(inputs.device)
    model_input = model_input + f_emb

    h = model.stacked_transformer_layer(model_input.float(), paddings=patched_padding.float())
    return stats, h


def decode_from_latent(model: TimesFm, h, stats):
    """
    Use horizon_ff_layer + reverse_transform to get the current-step output
    (the last decoded patch).
    Returns:
      mean_step: B x H
      full_step: B x H x Q
    """
    step_all = model._postprocess_output(h, stats)       # B x N x H x Q
    full_step = step_all[:, -1, :model.output_patch_len, :]
    mean_step = full_step[..., 0]
    return mean_step, full_step


def inner_lat_ascent(model: TimesFm,
                     x_win: torch.Tensor,
                     y_first: torch.Tensor,
                     steps: int,
                     eta_delta: float,
                     eps: float,
                     p_norm: str,
                     use_std_norm: bool = True,
                     attack_last_patch_only: bool = True):
    """
    Return the generated δ^h (detached; no gradient flows outward).
    """
    stats, h = get_latent_once(model, x_win)     # h: B x N x D
    if attack_last_patch_only:
        h_tgt = h[:, -1:, :]                     # B x 1 x D
    else:
        h_tgt = h                                # B x N x D  (attack all patches)

    # Initialize δ^h ~ N(0, I), then project to constraint set
    delta = torch.randn_like(h_tgt)

    # Per-dim std for normalization to stabilize the L∞ budget
    if use_std_norm:
        with torch.no_grad():
            std = h_tgt.flatten(0, 1).std(dim=0, keepdim=True).clamp_min(1e-5)  # 1 x D
            std = std.view(1, 1, -1)  # Bx1xD broadcast
    else:
        std = torch.ones_like(h_tgt)

    if p_norm == "linf":
        delta = (delta / std).clamp_(-eps, eps) * std
    elif p_norm == "l2":
        # Project in the normalized space, then map back
        v = delta / std
        v = project_lp(v, eps, norm="l2")
        delta = v * std

    delta.requires_grad_(True)

    # Global min/max of the clean activations (over batch & attacked slice)
    with torch.no_grad():
        h_min = h_tgt.amin(dim=(0, 1), keepdim=True)
        h_max = h_tgt.amax(dim=(0, 1), keepdim=True)

    for _ in range(steps):
        # Build perturbed h
        if attack_last_patch_only:
            h_adv = h.clone()
            h_adv[:, -1:, :] = (h_tgt + delta)
        else:
            h_adv = h_tgt + delta

        # Adversarial objective: σ=+1 (untargeted) — maximize error to ground-truth
        adv_mean, _ = decode_from_latent(model, h_adv if attack_last_patch_only else h_adv, stats)
        
        pred_len = y_first.shape[1]
        # Replaceable with MAE/CRPS/Pinball if desired
        loss_adv = ((adv_mean[:, -pred_len:] - y_first) ** 2).mean()

        # Gradient ascent
        g = torch.autograd.grad(loss_adv, delta, retain_graph=False, create_graph=False)[0]
        if p_norm == "linf":
            delta = (delta + eta_delta * torch.sign(g)).detach()
            delta = (delta / std).clamp_(-eps, eps) * std
        else:  # l2
            delta = (delta + eta_delta * g).detach()
            v = (delta / std)
            v = project_lp(v, eps, norm="l2")
            delta = v * std

        # (Optional) clip activations to [min(h), max(h)]
        adv_lat = (h_tgt + delta).clamp(h_min, h_max)
        delta = (adv_lat - h_tgt).detach()
        delta.requires_grad_(True)

    # Final adversarial single-step prediction (for reference)
    with torch.no_grad():
        if attack_last_patch_only:
            h_final = h.clone()
            h_final[:, -1:, :] = h_tgt + delta
        else:
            h_final = h_tgt + delta
        adv_mean, _ = decode_from_latent(model, h_final if attack_last_patch_only else h_final, stats)

    # return adv_mean.detach(), delta.detach()
    return delta.detach()


def evaluate_robustness(model: TimesFm,
                        dm: DataModule,
                        evaluator: Evaluator,
                        args):
    """
    Evaluate three settings: clean / PGD / SimBA.
    Return a dict:
      {'clean': {...}, 'pgd': {...}, 'simba': {...}}
    Note: enable gradients during PGD; clean and post-attack forwards can use no_grad.
    """
    model.eval()
    freq = dm.data_manager.freq
    ctx_len = dm.data_manager.context_length
    point_num = int(args.attack_ratio * ctx_len)

    buf = {
        'clean': {'inp': [], 'gt': [], 'q': []},
        'pgd':   {'inp': [], 'gt': [], 'q': []},
        'simba': {'inp': [], 'gt': [], 'q': []},
    }

    for batch in dm.test_dataloader():
        past = batch['past_target_cdf'].to(device)
        fut  = batch['future_target_cdf'].to(device)
        if past.ndim == 3:                         # (B, L, K) -> (B*K, L)
            past = rearrange(past, 'b l k -> (b k) l')
            fut  = rearrange(fut,  'b l k -> (b k) l')

        # ---------- clean forward (no gradients) ----------
        with torch.no_grad():
            mean_clean, q_clean = model(past)

        buf['clean']['inp'].append(past.detach().cpu())
        buf['clean']['gt'].append(fut.detach().cpu())
        buf['clean']['q'].append(q_clean.detach().cpu())

        # Attack target (untargeted w.r.t. GT; otherwise w.r.t. clean prediction)
        attack_target = fut.detach() if args.attack_gt else mean_clean.detach()

        if point_num > 0:
            # ---------- PGD adversarial example generation (requires gradients) ----------
            from AttackTSFM.attack import PGD, SimBA  # could also be imported at file top
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                pgd = PGD(model, alpha=args.alpha, targeted=False, norm=args.norm)
                pert_past_pgd = pgd.attack_batch(
                    past, attack_target,
                    n_dims=point_num, max_iters=args.num_iterations,
                    epsilon=args.epsilon,
                    past_datastamps=batch.get('past_time_feat', None),
                    future_datastamps=batch.get('future_time_feat', None)
                )
            # forward after attack (no gradients)
            with torch.no_grad():
                _, q_pgd = model(pert_past_pgd)
            buf['pgd']['inp'].append(past.detach().cpu())
            buf['pgd']['gt'].append(fut.detach().cpu())
            buf['pgd']['q'].append(q_pgd.detach().cpu())

            # ---------- SimBA (black-box; no gradients, and do not wrap in enable_grad) ----------
            simba = SimBA(model, targeted=False,
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
                _, q_simba = model(pert_past_simba)
            buf['simba']['inp'].append(past.detach().cpu())
            buf['simba']['gt'].append(fut.detach().cpu())
            buf['simba']['q'].append(q_simba.detach().cpu())
        else:
            # If no attack, reuse the clean results for pgd/simba buckets
            for key in ['pgd', 'simba']:
                buf[key]['inp'].append(past.detach().cpu())
                buf[key]['gt'].append(fut.detach().cpu())
                buf[key]['q'].append(q_clean.detach().cpu())

    # ---------- compute metrics ----------
    out = {}
    
    for k in ['clean', 'pgd', 'simba']:
        res = evaluator(
            torch.cat(buf[k]['gt'], dim=0),
            torch.cat(buf[k]['q'], dim=0),
            torch.cat(buf[k]['inp'], dim=0),
            freq=freq
        )

        out[k] = res
    return out


def adv_forward_with_delta(model: TimesFm,
                           inputs: torch.Tensor,
                           delta_star: torch.Tensor,
                           attack_last_patch_only: bool = True):
    """
    For the outer step: recompute h (with gradients), add δ* (detached) to the
    target latent, and obtain the adversarial single-step prediction mean (B, H).
    """

    input_ts, input_padding = model._preprocess(inputs, horizon_len=model.output_patch_len)
    
    final_out = input_ts
    max_len = input_ts.shape[1]
    current_padding = input_padding[:, 0:final_out.shape[1]]
    input_ts = final_out[:, -max_len:]
    input_padding = current_padding[:, -max_len:]
    
    model_input, patched_padding, stats, _ = model._preprocess_input(input_ts, input_padding)

    bs = inputs.shape[0]
    f_emb = model.freq_emb(torch.tensor([0] * bs).unsqueeze(-1).long()).to(inputs.device)
    model_input = model_input + f_emb

    h = model.stacked_transformer_layer(model_input.float(), paddings=patched_padding.float())
    
    if attack_last_patch_only:
        h_adv = h.clone()
        h_adv[:, -1:, :] = h_adv[:, -1:, :] + delta_star.detach()
    else:
        h_adv = h + delta_star.detach()

    adv_mean, _ = decode_from_latent(model, h_adv, stats)
    return adv_mean

def train(args):
    set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    setup_logger(os.path.join(args.save_dir, args.log))
    logging.info(vars(args))

    # Data
    dmgr = DataManager(
        dataset = args.dataset,
        path = args.dataset_path,
        history_length=args.context_length,
        context_length = args.context_length,
        prediction_length = args.prediction_length,
        scaler='standard',
        timeenc=2,
        target_dim=args.target_dim,
        gift_eval=args.gift_eval,)
    data = DataModule(dmgr, batch_size=args.batch_size,
                      test_batch_size=args.batch_size,
                      num_workers=args.num_workers)

    # Model
    model = TimesFm(
        context_len=dmgr.context_length,
        input_patch_len=32,
        output_patch_len=128,
        model_size=args.model_size,
        horizon_len=dmgr.prediction_length,
        ckpt_path=abs_path,
        device=device,
    ).to(device)
    
    evaluator = Evaluator()
    if args.robust_eval:
        pre_metrics  = evaluate_robustness(model, data, evaluator, args)
                
    model.train()

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in trange(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, steps = 0.0, 0
        
        count = 0

        for batch in data.train_dataloader():
            past = batch['past_target_cdf'].to(device)
            fut = batch['future_target_cdf'].to(device)
            
            count += 1
            if count > args.max_iter:
                break

            # Flatten multivariate dimension into batch
            if past.ndim == 3:
                past = rearrange(past, 'b l k -> (b k) l')
                fut = rearrange(fut, 'b l k -> (b k) l')
        
            # Align only the first output patch label (TimesFm decodes one patch per step)
            H1 = model.output_patch_len
            y_first = fut[:, :H1].detach()

            # Inner ascent: compute δ^{h*} (Algorithm 2: lines 4–11)
            # 1) Inner step: only solve δ*
            delta_star = inner_lat_ascent(
                model, past, y_first,
                steps=args.inner_steps,
                eta_delta=args.eta_delta,
                eps=args.eps,
                p_norm=args.p_norm,
                use_std_norm=args.std_normalize,
                attack_last_patch_only=args.attack_last_patch_only
            )

            # 2) Outer step: differentiable forward; use δ* to generate adversarial prediction
            adv_mean_first = adv_forward_with_delta(
                model, past, delta_star,
                attack_last_patch_only=args.attack_last_patch_only
            )

            pred_len = y_first.shape[1]
            loss = ((adv_mean_first[:, -pred_len:] - y_first) ** 2).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            run_loss += loss.item()
            steps += 1

        logging.info(f"[Epoch {epoch}] train_loss={run_loss/max(1,steps):.6f} "
                     f"time={(time.time()-t0):.1f}s")

        if epoch % args.eval_every == 0:
            if args.robust_eval:
                post_metrics = evaluate_robustness(model, data, evaluator, args)

                def _fmt(m):  # compact printer
                    return f"ND={m['ND']:.4f}  NRMSE={m['NRMSE']:.4f}  CRPS={m['CRPS']:.4f}"

                logging.info("Epoch: " + str(epoch)) 
                logging.info("[ROBUST EVAL] BEFORE LAT: clean  " + _fmt(pre_metrics['clean']) + "AFTER  LAT: clean " + _fmt(post_metrics['clean']))
                logging.info("[ROBUST EVAL] BEFORE LAT: PGD    " + _fmt(pre_metrics['pgd']) + "AFTER  LAT: PGD   " + _fmt(post_metrics['pgd']))
                logging.info("[ROBUST EVAL] BEFORE LAT: SimBA  " + _fmt(pre_metrics['simba']) + "AFTER  LAT: SimBA " + _fmt(post_metrics['simba']))

    final_ckpt = os.path.join(args.save_dir, "timesfm_lat_final.pt")
    torch.save(model.state_dict(), final_ckpt)
    logging.info(f"Saved final model: {final_ckpt}")


if __name__ == "__main__":
    args = parse_args()
    train(args)