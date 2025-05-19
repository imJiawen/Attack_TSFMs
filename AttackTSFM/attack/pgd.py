import torch
from .utils import project_perturbation

class PGD:
	def __init__(self, model, alpha=0.01, targeted=False, random_threshold=1e-6, norm='linf'):
		self.model = model
		self.alpha = alpha
		self.targeted = targeted
		self.random_threshold = random_threshold
		self.norm = norm
  
	def _scale(self, epsilon, past_tensor):
		std = past_tensor.std(dim=-1, keepdim=True)
		scaled_epsilon = std * epsilon
		return scaled_epsilon
		
	def attack_batch(self, past_tensor, target_tensor, n_dims=10, max_iters=10, epsilon=0.1, *args, **kwargs):
		"""
		Perform a PGD attack on the last `point_num` elements of the last dimension of `past_tensor`,
		where `epsilon` is scaled by the standard deviation of the sequence.

		Args:
			model: The model being attacked.
			past_tensor: The original input tensor, shape [b, l] or [b, k, l].
			actuals_tensor: The ground truth tensor.
			target_tensor: Optional target tensor for targeted attack.
			epsilon: The maximum perturbation as a multiplier of the standard deviation.
			alpha: The step size for each iteration.
			num_iterations: The number of attack iterations.
			point_num: The size of the last segment in the last dimension to be attacked.

		Returns:
			perturbed_input: The perturbed tensor after the PGD attack.
		"""
		# Make the input tensor require gradients
		device = past_tensor.device
		perturbed_input = past_tensor.clone().detach().requires_grad_(True).to(device)

		# Calculate the standard deviation of the sequence across the last dimension
		scaled_epsilon = self._scale(epsilon, past_tensor)
		scaled_alpha = self._scale(self.alpha, past_tensor)

		# Get the shape of the tensor
		shape = perturbed_input.shape

		# Determine the attack range for the last dimension
		attack_start_idx = shape[-1] - n_dims

		for _ in range(max_iters):
			# Forward pass through the model with the current perturbed input
			if self.targeted:
				loss = -self.model.loss(perturbed_input, target_tensor)
			else:
				loss = self.model.loss(perturbed_input, target_tensor)

			# is loss is too small, use random perturbation instead for this iteration
			if torch.abs(loss).item() < self.random_threshold:
				grad = (torch.rand_like(perturbed_input) - 0.5) * 2
			else:
				# PGD uodate
				self.model.zero_grad()
				loss.backward()
				grad = perturbed_input.grad.data

			# Slice to the attack region
			pert_slice = perturbed_input[..., attack_start_idx:]
			past_slice = past_tensor[..., attack_start_idx:]
			grad_slice = grad[..., attack_start_idx:]
   
			# PGD update
			adv_part = pert_slice + scaled_alpha * torch.sign(grad_slice)
			delta = adv_part - past_slice

			# Apply norm constraint
			projected = project_perturbation(self.norm, delta.view(delta.shape[0], -1), scaled_epsilon.view(scaled_epsilon.shape[0], -1))
			perturbed_input.data[..., attack_start_idx:] = past_slice + projected.view_as(delta)

			perturbed_input.requires_grad_(True)

		return perturbed_input
