import torch
from .utils import project_perturbation

class ZOO:
    def __init__(self, model, targeted=False, alpha=0.01, mode='random', q=10, norm='linf', ):
        self.model = model
        self.alpha = alpha  # Small perturbation for finite difference gradient
        self.targeted = targeted
        self.mode = mode  # 'vector' or 'coordinate'
        self.q = q
        self.norm = norm
        
    def get_probs(self, x, y, **kwargs):
        forecasts, _ = self.model(x, **kwargs)
        log_probs = (forecasts - y) ** 2
        return log_probs.mean(dim=-1)

    def _scale_eps(self, epsilon, past_tensor):
        std = past_tensor.std(dim=-1, keepdim=True)
        return std * epsilon

    def expand_vector(self, x, length):
        batch_size, n_dim = x.shape
        z = torch.zeros(batch_size, length).to(x.device)
        z[:, -n_dim:] = x
        return z

    def attack_batch(self, past_tensor, target_tensor, n_dims, max_iters, eta=1, epsilon=0.5, **kwargs):
        device = past_tensor.device
        batch_size, length = past_tensor.shape

        # Initialize perturbation
        x = torch.zeros(batch_size, n_dims, device=device)
        
        if self.norm == 'l0':
            eps_scaled = epsilon
        else:
            eps_scaled = self._scale_eps(epsilon, past_tensor)
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_iters):
                if self.mode == 'random':
                    grad = torch.zeros_like(x)
                    base = self.get_probs(past_tensor + self.expand_vector(x, length), target_tensor, **kwargs)
                    
                    for _ in range(self.q):
                        u = torch.randn_like(x)
                        perturbed = x + self.alpha * u
                        adv = past_tensor + self.expand_vector(perturbed, length)
                        perturbed_loss = self.get_probs(adv, target_tensor, **kwargs)

                        # ∇f ≈ (f(θ + αu) - f(θ)) / α * u
                        est = ((perturbed_loss - base).unsqueeze(1) / self.alpha) * u
                        grad += est
                        
                    grad = grad / self.q
                    if self.targeted:
                        grad = -grad

                    x = x + eta * grad
                    x = project_perturbation(self.norm, x, epsilon=eps_scaled)

                elif self.mode == 'coordinate':
                    grad = torch.zeros_like(x)
                    base = self.get_probs(past_tensor + self.expand_vector(x, length), target_tensor, **kwargs)  # [B]

                    for i in range(n_dims):
                        ei = torch.zeros_like(x)
                        ei[:, i] = self.alpha

                        x_perturbed = x + ei
                        adv = past_tensor + self.expand_vector(x_perturbed, length)
                        loss_perturbed = self.get_probs(adv, target_tensor, **kwargs)  # [B]

                        g_i = ((loss_perturbed - base).unsqueeze(1) / self.alpha) * ei
                        grad += g_i

                    if self.targeted:
                        grad = -grad

                    x = x + eta * grad
                    x = project_perturbation(self.norm, x, epsilon=eps_scaled)
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

        # Final adversarial example
        adv = past_tensor + self.expand_vector(x, length)
        return adv


class ZOOAdam:
    def __init__(self, model, targeted=False, alpha=0.01, mode='random', q=10, norm='linf',
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model = model
        self.alpha = alpha
        self.targeted = targeted
        self.mode = mode  # 'random' or 'coordinate'
        self.q = q
        self.norm = norm

        # Adam hyperparameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon_adam = epsilon  # avoid conflict with epsilon for l_inf

    def get_probs(self, x, y, **kwargs):
        forecasts, _ = self.model(x, **kwargs)
        return ((forecasts - y) ** 2).mean(dim=-1)

    def _scale_eps(self, epsilon, past_tensor):
        std = past_tensor.std(dim=-1, keepdim=True)
        return std * epsilon

    def expand_vector(self, x, length):
        batch_size, n_dim = x.shape
        z = torch.zeros(batch_size, length).to(x.device)
        z[:, -n_dim:] = x
        return z

    def attack_batch(self, past_tensor, target_tensor, n_dims, max_iters, eta=1.0, epsilon=0.5, **kwargs):
        device = past_tensor.device
        batch_size, length = past_tensor.shape
        x = torch.zeros(batch_size, n_dims, device=device)
        m = torch.zeros_like(x)
        v = torch.zeros_like(x)

        if self.norm == 'l0':
            eps_scaled = epsilon
        else:
            eps_scaled = self._scale_eps(epsilon, past_tensor)

        self.model.eval()
        with torch.no_grad():
            for t in range(1, max_iters + 1):
                if self.mode == 'random':
                    grad = torch.zeros_like(x)
                    base = self.get_probs(past_tensor + self.expand_vector(x, length), target_tensor, **kwargs)

                    for _ in range(self.q):
                        u = torch.randn_like(x)
                        perturbed = x + self.alpha * u
                        adv = past_tensor + self.expand_vector(perturbed, length)
                        perturbed_loss = self.get_probs(adv, target_tensor, **kwargs)
                        est = ((perturbed_loss - base).unsqueeze(1) / self.alpha) * u
                        grad += est

                    grad = grad / self.q
                    if self.targeted:
                        grad = -grad

                elif self.mode == 'coordinate':
                    grad = torch.zeros_like(x)
                    base = self.get_probs(past_tensor + self.expand_vector(x, length), target_tensor, **kwargs)

                    for i in range(n_dims):
                        ei = torch.zeros_like(x)
                        ei[:, i] = self.alpha
                        x_perturbed = x + ei
                        adv = past_tensor + self.expand_vector(x_perturbed, length)
                        loss_perturbed = self.get_probs(adv, target_tensor, **kwargs)
                        g_i = ((loss_perturbed - base).unsqueeze(1) / self.alpha) * ei
                        grad += g_i

                    if self.targeted:
                        grad = -grad

                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

                # --- Adam update ---
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

                m_hat = m / (1 - self.beta1 ** t)
                v_hat = v / (1 - self.beta2 ** t)

                x = x + eta * m_hat / (v_hat.sqrt() + self.epsilon_adam)

                # Project to constraint set
                x = project_perturbation(self.norm, x, epsilon=eps_scaled)

        adv = past_tensor + self.expand_vector(x, length)
        return adv