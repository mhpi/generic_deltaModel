from typing import Dict


def print_metrics(
    metrics: Dict[str, Dict[str, float]],
    metric_names: str,
    mode: str = 'median',
    precision: int = 3
) -> None:
    """
    Prints either the median or the mean ± standard deviation for the specified metrics,
    with a specified number of significant digits after the decimal point.

    Parameters
    ----------
    metrics : dict
        Dictionary containing the metrics data.
    metric_names : list
        List of metric names to process.
    mode : str, optional
        Either "median" or "mean_std". Defaults to "median".
    precision : int, optional
        Number of significant digits after the decimal point. Defaults to 3.
    """
    if mode not in ["median", "mean_std"]:
        print("Error: Mode must be 'median' or 'mean_std'.")
        return

    if mode == "median":
        mode_name = "Median"
    else:
        mode_name = "Mean ± Std"

    print(f"{mode_name} of Metrics (Prec: {precision} digits):")
    print("-" * 40)
    for name in metric_names:
        if name in metrics:
            if mode == "median" and "median" in metrics[name]:
                value = metrics[name]["median"]
                print(f"{name.capitalize()}: {value:.{precision}f}")
            elif mode == "mean_std" and "mean" in metrics[name] and "std" in metrics[name]:
                mean = metrics[name]["mean"]
                std = metrics[name]["std"]
                print(f"{name.capitalize()}: {mean:.{precision}f} ± {std:.{precision}f}")
            else:
                print(f"{name.capitalize()}: Metric data incomplete")
        else:
            print(f"{name.capitalize()}: Metric not found")
    print("-" * 40)


def format_resample_interval(resample: str) -> str:
    """Formats the resampling interval into a human-readable string.
    
    Parameters
    ----------
    resample: str
        The resampling interval (e.g., 'D', 'W', '3D', 'M', 'Y').

    Returns
    -------
    str
        A formatted string describing the resampling interval.
    """
    # Check if the interval contains a number (e.g., "3D")
    if any(char.isdigit() for char in resample):
        # Extract the numeric part and the unit part
        num = ''.join(filter(str.isdigit, resample))
        unit = ''.join(filter(str.isalpha, resample))
        
        # Map units to human-readable names
        if num == '1':
            unit_map = {
                'D': 'daily',
                'W': 'weekly',
                'M': 'monthly',
                'Y': 'yearly',
            }
        else:
            unit_map = {
                'D': 'days',
                'W': 'weeks',
                'M': 'months',
                'Y': 'years',
            }
        return f"{num} {unit_map.get(unit, unit)}"
    else:
        # Single-character intervals (e.g., "D", "W")
        unit_map = {
            'D': 'daily',
            'W': 'weekly',
            'M': 'monthly',
            'Y': 'yearly',
        }
        return unit_map.get(resample, resample)
