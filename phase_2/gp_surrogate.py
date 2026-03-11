"""
Gaussian Process Surrogate & ARD Categorical Kernel
===================================================
Models the discrete combinatorial text space to continuous MSE loss
using an Automatic Relevance Determination (ARD) categorical kernel.
"""

import math
import torch
import gpytorch
import numpy as np
from torch import Tensor
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.priors import GammaPrior, UniformPrior
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.transforms import normalize_indices

# ─── 1. Core Math: Memory-Efficient Categorical Distance ──────────────────────

def detach_variable(inputs):
    if isinstance(inputs, tuple):
        return tuple([detach_variable(x) for x in inputs])
    else:
        x = inputs.detach()
        x.requires_grad = inputs.requires_grad
        return x

class BinaryGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lengthscale, x1, x2):
        nbatch = 256
        detached_ls = detach_variable(lengthscale)
        m = len(x1.shape)
        if m == 2:
            N, M, L = x1.shape[0], x2.shape[0], x1.shape[-1]
        elif m ==  3:
            N, M, L = x1.shape[0], x2.shape[1], x1.shape[-1]
        
        dists = []
        if m == 2:
            for i in range(int(np.ceil(N / nbatch))):
                dists_pt = []
                for j in range(int(np.ceil(M / nbatch))):
                    binary_pt_pt = x1[i*nbatch:(i+1)*nbatch].unsqueeze(-2) != x2[j*nbatch:(j+1)*nbatch].unsqueeze(-3)
                    tmp_pt = binary_pt_pt / detached_ls
                    dists_pt.append(tmp_pt.mean(-1))
                    del binary_pt_pt, tmp_pt
                dists.append(torch.cat(dists_pt, dim=1))
            dists_final = torch.cat(dists, dim=0)

        elif m == 3:
            for i in range(int(np.ceil(N / nbatch))):
                dists_pt = []
                for j in range(int(np.ceil(M / nbatch))):
                    binary_pt_pt = x1[i*nbatch:(i+1)*nbatch].unsqueeze(-2) != x2[i*nbatch:(i+1)*nbatch, j*nbatch:(j+1)*nbatch].unsqueeze(-3)
                    tmp_pt = binary_pt_pt / detached_ls
                    dists_pt.append(tmp_pt.mean(-1))
                    del binary_pt_pt, tmp_pt
                dists.append(torch.cat(dists_pt, dim=2))
            dists_final = torch.cat(dists, dim=0)
            
        del dists, dists_pt
        ctx.save_for_backward(x1, x2, detached_ls)
        return dists_final

    @staticmethod
    def backward(ctx, grad_output):
        nbatch = 256
        x1, x2, detached_ls = ctx.saved_tensors
        m = len(x1.shape)
        if m == 2:
            sumdim = [0,1]
            N, M, L = x1.shape[0], x2.shape[0], x1.shape[-1]
        elif m ==  3:
            sumdim = [0,1,2]
            N, M, L = x1.shape[0], x2.shape[1], x1.shape[-1]

        grad = 0
        if m == 2:
            for i in range(int(np.ceil(N / nbatch))):
                for j in range(int(np.ceil(M / nbatch))):
                    binary_pt_pt = x1[i*nbatch:(i+1)*nbatch].unsqueeze(-2) != x2[j*nbatch:(j+1)*nbatch].unsqueeze(-3)
                    tmp = -grad_output[i*nbatch:(i+1)*nbatch, j*nbatch:(j+1)*nbatch].unsqueeze(-1) * binary_pt_pt
                    grad += torch.sum(tmp / torch.square(detached_ls), dim=sumdim) / L
                    del binary_pt_pt, tmp
        elif m == 3:
            for i in range(int(np.ceil(N / nbatch))):
                for j in range(int(np.ceil(M / nbatch))):
                    binary_pt_pt = x1[i*nbatch:(i+1)*nbatch].unsqueeze(-2) != x2[i*nbatch:(i+1)*nbatch, j*nbatch:(j+1)*nbatch].unsqueeze(-3)
                    tmp = -grad_output[i*nbatch:(i+1)*nbatch, :, j*nbatch:(j+1)*nbatch].unsqueeze(-1) * binary_pt_pt
                    grad += torch.sum(tmp / torch.square(detached_ls), dim=sumdim) / L
                    del binary_pt_pt, tmp
        return grad.view(1,1,-1), None, None


# ─── 2. ARD Categorical Kernel ────────────────────────────────────────────────

class CategoricalKernel2(Kernel):
    has_lengthscale = True

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, key: int = 0, **kwargs) -> Tensor:
        if key == 0: 
            dists = BinaryGradientFunction.apply(self.lengthscale.unsqueeze(-2), x1, x2)
        elif key == 1:
            delta = x1.unsqueeze(-2) != x2.unsqueeze(-3)
            dists = delta / self.lengthscale.unsqueeze(-2)
            if last_dim_is_batch:
                dists = dists.transpose(-3, -1)
            else:
                dists = dists.mean(-1)
        res = torch.exp(-dists)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res


# ─── 3. BoTorch GP Wrapper ────────────────────────────────────────────────────

class NewMixedSingleTaskGP(SingleTaskGP):
    def __init__(self, train_X: Tensor, train_Y: Tensor, cat_dims: list, likelihood=None):
        if len(cat_dims) == 0:
            raise ValueError("Must specify categorical dimensions")
            
        self._ignore_X_dims_scaling_check = cat_dims
        
        # Safely determine batch shapes
        try:
            input_batch_shape, aug_batch_shape = self.get_batch_dimensions(train_X=train_X, train_Y=train_Y)
        except AttributeError:
            input_batch_shape = train_X.shape[:-2]
            aug_batch_shape = input_batch_shape

        # if likelihood is None:
        #     min_noise = 1e-5 if train_X.dtype == torch.float else 1e-6
        #     likelihood = GaussianLikelihood(
        #         batch_shape=aug_batch_shape,
        #         noise_constraint=GreaterThan(min_noise, transform=None, initial_value=1e-3),
        #         noise_prior=GammaPrior(0.9, 10.0),
        #     )

        if likelihood is None:
            # Hardcoding a higher minimum noise floor (1e-3) so it expects variance
            min_noise = 1e-3 
            likelihood = GaussianLikelihood(
                batch_shape=aug_batch_shape,
                noise_constraint=GreaterThan(min_noise, transform=None, initial_value=1e-3),
                noise_prior=GammaPrior(0.9, 10.0),
            )

        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)

        lengthscale_prior = GammaPrior(3.0, 6.0)
        outputscale_prior = UniformPrior(0, 1, validate_args=False)
        outputscale_constraint = Interval(0, 1, initial_value=0.1)
        
        # Ensure priors are on the same device as the training data
        outputscale_prior.low = outputscale_prior.low.to(train_X.device)
        outputscale_prior.high = outputscale_prior.high.to(train_X.device)
        
        covar_module = ScaleKernel(
            CategoricalKernel2(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(cat_dims),
                lengthscale_constraint=GreaterThan(1e-06),
                lengthscale_prior=lengthscale_prior
            ),
            batch_shape=aug_batch_shape,
            outputscale_constraint=outputscale_constraint,
            outputscale_prior=outputscale_prior,
        ) 
        
        super().__init__(
            train_X=train_X, train_Y=train_Y, likelihood=likelihood, covar_module=covar_module
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# ─── 4. The Clean API (Orchestrator Module) ───────────────────────────────────

class GPSurrogate:
    """
    Clean wrapper for fitting the GP and calculating acquisition scores.
    Replaces `MyGPModel` and `fit_model_partial`.
    """
    def __init__(self, sequence_length: int, device: str):
        self.device = torch.device(device)
        self.sequence_length = sequence_length
        self.model = None
        self.partition_size = 512 # Prevents OOM during prediction

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor, fit_iter: int = 20):
        """
        Fits the GP to the history of evaluated sequences using Exact Marginal Log-Likelihood.
        """
        train_X = train_X.to(dtype=torch.float32, device=self.device)
        train_Y = train_Y.to(dtype=torch.float32, device=self.device)
        
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)
            
        cat_dims = list(range(self.sequence_length))
        
        # Re-initialize the model to absorb new data shapes
        self.model = NewMixedSingleTaskGP(train_X=train_X, train_Y=train_Y, cat_dims=cat_dims).to(self.device)
        # self.model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        self.model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-3))
        self.model.mean_module.initialize(constant=-1.0)
        
        self.model.train()
        self.model.likelihood.train()

        # Optimize hyperparameters (Beta lengthscales)
        optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=0.1)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model).to(self.device)
        
        for _ in range(fit_iter):
            optimizer.zero_grad()
            output = self.model(train_X)
            loss = -mll(output, self.model.train_targets)
            loss.backward()
            optimizer.step()
            
    def predict(self, test_X: torch.Tensor):
        """Outputs predictive mean and variance in memory-safe batches."""
        self.model.eval()
        self.model.likelihood.eval()

        test_X = test_X.to(dtype=torch.float32, device=self.device)
        N = test_X.shape[0]
        N_pt = int(np.ceil(N / self.partition_size))
        
        means, variances = [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(N_pt):
                eval_X_pt = test_X[self.partition_size * i : self.partition_size * (i+1)]
                pred_pt = self.model(eval_X_pt)
                pred_pt = self.model.likelihood(pred_pt)
                
                means.append(pred_pt.mean.detach())
                variances.append(pred_pt.variance.clamp_min(1e-9).detach())
                
        return torch.cat(means, dim=0), torch.cat(variances, dim=0)

    def acquisition(self, test_X: torch.Tensor, best_f: float):
        """
        Computes the Expected Improvement (EI) for untested sequences.
        Memory-batched internally to prevent crash on huge candidate spaces.
        """
        mean, var = self.predict(test_X)
        std = torch.sqrt(var)
        
        # Expected Improvement Math
        Z = (mean - best_f) / std
        normal = torch.distributions.Normal(0, 1)
        pdf = torch.exp(normal.log_prob(Z))
        cdf = normal.cdf(Z)
        
        ei = std * (Z * cdf + pdf)
        # If std is 0, EI is 0
        ei[std == 0.0] = 0.0 
        
        return ei

# ─── Sanity Check ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1. Setup mock data (5 evaluated sequences of length 16)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    seq_length = 16
    mock_X = torch.randint(0, 5, (5, seq_length))
    mock_Y = torch.randn(5) # Random MSE losses

    # 2. Init and Fit Surrogate
    surrogate = GPSurrogate(sequence_length=seq_length, device=device)
    print("Fitting GP to mock data...")
    surrogate.fit(mock_X, mock_Y, fit_iter=10)
    
    # 3. Predict and score candidates
    print("Scoring 1,000 random test candidates...")
    test_candidates = torch.randint(0, 5, (1000, seq_length))
    
    # Best observed loss so far
    best_loss = mock_Y.max().item() 
    
    ei_scores = surrogate.acquisition(test_candidates, best_f=best_loss)
    print(f"Top 5 Candidate EI Scores: {ei_scores.topk(5).values.cpu().numpy()}")