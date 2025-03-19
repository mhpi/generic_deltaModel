from typing import Any, Dict, Optional

import torch


class RmseCombLoss(torch.nn.Module):
    """Combination root mean squared error (RMSE) loss function.

    This loss combines the RMSE of the target variable with the RMSE of
    the log-transformed target variable.

    The RMSE is calculated as:
        p: predicted value,
        t: target value,
        RMSE = sqrt(mean((p - t)^2))
    
    The log-sqrt RMSE is calculated as:
        p: predicted value,
        t: target value,
        RMSE = sqrt(mean((log(sqrt(p)) - log(sqrt(t)))^2))

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        The device to run loss function on.
    **kwargs
        Additional arguments.

        - alpha: Weighting factor for the log-sqrt RMSE. Default is 0.25.

        - beta: Stability term to prevent division by zero. Default is 1e-6.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs: float,
    ) -> None:
        super().__init__()
        self.name = 'Combination RMSE Loss'
        self.config = config
        self.device = device

        self.alpha = kwargs.get('alpha', config.get('alpha', 0.25))
        self.beta = kwargs.get('beta', config.get('beta', 1e-6))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute loss.
        
        Parameters
        ----------
        y_pred
            Tensor of predicted target data.
        y_obs
            Tensor of target observation data.
        **kwargs
            Additional arguments for interface compatibility, not used.

        Returns
        -------
        torch.Tensor
            The combined loss.
        """
        prediction = y_pred.squeeze()
        target = y_obs[:, :, 0]

        if len(target) > 0:
            # Mask where observations are valid (not NaN).            
            mask = ~torch.isnan(target)
            p_sub = prediction[mask]
            t_sub = target[mask]
            
            # RMSE
            p_sub1 = torch.log10(torch.sqrt(prediction + self.beta) + 0.1)
            t_sub1 = torch.log10(torch.sqrt(target + self.beta) + 0.1)
            loss1 = torch.sqrt(((p_sub - t_sub) ** 2).mean())  # RMSE item

            # Log-Sqrt RMSE
            mask2 = ~torch.isnan(t_sub1)
            p_sub2 = p_sub1[mask2]
            t_sub2 = t_sub1[mask2]
            loss2 = torch.sqrt(((p_sub2 - t_sub2) ** 2).mean())

            # Combined losses
            loss = (1.0 - self.alpha) * loss1 + self.alpha * loss2
        else:
            loss = torch.tensor(0.0, device=self.device)
        return loss
