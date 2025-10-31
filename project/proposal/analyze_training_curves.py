"""
Extract TensorBoard training curves and fit models to estimate convergence.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def log_model(x, a, b):
    """Simplified logarithmic: a * log(x + 1) + b"""
    return a * np.log(x + 1) + b

def exp_decay_model(x, a, b, c):
    """Exponential decay: a * exp(-bx) + c"""
    return a * np.exp(-b * x) + c

def power_law_model(x, a, b):
    """Power law: a * x^b"""
    return a * x ** b

def extract_tensorboard_data(logdir):
    """Extract scalar data from TensorBoard logs."""
    ea = EventAccumulator(logdir)
    ea.Reload()
    
    scalars = ea.Tags()['scalars']
    data = {}
    
    for scalar_name in scalars:
        scalar_events = ea.Scalars(scalar_name)

        if not scalar_events:
            continue

        # 1) Keep only the latest session by detecting a large wall_time gap
        #    between consecutive events and selecting the final contiguous block.
        #    This avoids mixing previous long runs with the current ongoing run.
        GAP_SECONDS = 3 * 60  # consider gaps >10 minutes as run/session boundaries
        # Sort by wall_time to detect gaps
        scalar_events_sorted = sorted(scalar_events, key=lambda e: e.wall_time)
        wall_times = np.array([e.wall_time for e in scalar_events_sorted])
        gaps = np.diff(wall_times)
        last_big_gap_idx = np.where(gaps > GAP_SECONDS)[0]
        if len(last_big_gap_idx) > 0:
            start_idx = last_big_gap_idx[-1] + 1
        else:
            start_idx = 0
        latest_session_events = scalar_events_sorted[start_idx:]

        # 2) Deduplicate by step within the latest session, keeping the latest wall_time value
        step_to_event = {}
        for ev in latest_session_events:
            if ev.step not in step_to_event or ev.wall_time > step_to_event[ev.step].wall_time:
                step_to_event[ev.step] = ev

        # Build arrays sorted by step
        unique_steps = sorted(step_to_event.keys())
        steps = np.array(unique_steps)
        values = np.array([step_to_event[s].value for s in unique_steps])

        data[scalar_name] = {'steps': steps, 'values': values}
    
    return data

def fit_and_predict(data, metric_name, epochs_to_predict=10_000, use_recent_only=True, recent_window=0.8):
    """Fit curve and predict future values.
    
    Args:
        use_recent_only: If True, only use the most recent data points for fitting
        recent_window: Fraction of data to use when use_recent_only=True (default: 0.75 = last 75%)
    """
    steps = data['steps']
    values = data['values']
    
    # Sort by steps to ensure proper ordering
    sort_idx = np.argsort(steps)
    steps = steps[sort_idx]
    values = values[sort_idx]
    
    # Store original full dataset for plotting
    original_steps = steps.copy()
    original_values = values.copy()
    
    # Optionally use only recent data points for better fit to recent values
    if use_recent_only and len(values) > 10:
        n_keep = max(10, int(len(values) * recent_window))
        steps = steps[-n_keep:]
        values = values[-n_keep:]
    
    # Try different models and pick the best
    models_to_try = []
    
    # Try exponential decay for decreasing metrics
    if values[-1] < values[0]:
        models_to_try.append((exp_decay_model, [abs(values[0] - values[-1]), 0.01, values[-1]]))
        models_to_try.append((log_model, [values[0] - values[-1], values[-1]]))
    
    # Try logarithmic for increasing metrics
    else:
        models_to_try.append((log_model, [values[-1] - values[0], values[0]]))
        models_to_try.append((power_law_model, [values[-1] - values[0], 0.5]))
    
    best_error = float('inf')
    best_model = None
    best_params = None
    best_model_name = None
    
    for model_func, p0 in models_to_try:
        try:
            # Use actual step numbers, not normalized
            popt, _ = curve_fit(model_func, steps, values, p0=p0, maxfev=10_000)
            fit_values = model_func(steps, *popt)
            error = np.sqrt(np.mean((values - fit_values)**2))
            
            # Check if fit is reasonable (no extreme values)
            if np.any(fit_values < 0) or np.any(np.isnan(fit_values)):
                continue
                
            if error < best_error:
                best_error = error
                best_model = model_func
                best_params = popt
                best_model_name = model_func.__name__
        except (RuntimeError, ValueError, TypeError):
            continue
    
    if best_model is None:
        return None
    
    # Predict future values
    future_steps = np.arange(steps[-1] + 1, epochs_to_predict + 1)
    predicted_values = best_model(future_steps, *best_params)
    
    # Ensure predictions are reasonable
    predicted_values = np.clip(predicted_values, 0, np.max(values) * 10)
    
    # Calculate final predicted value
    final_value = best_model(epochs_to_predict, *best_params)
    final_value = np.clip(final_value, 0, np.max(values) * 10)
    
    # Estimate convergence
    converged_epoch = epochs_to_predict
    if len(predicted_values) > 10:
        for i in range(10, len(predicted_values) - 5):
            recent = predicted_values[i-5:i+5]
            if np.std(recent) < 0.001:
                converged_epoch = future_steps[i]
                break
    
    return {
        'params': best_params,
        'model': best_model,
        'model_name': best_model_name,
        'predictions': predicted_values,
        'future_steps': future_steps,
        'converged_epoch': converged_epoch,
        'final_predicted_value': final_value,
        'fit_quality': best_error,
        'current_value': original_values[-1],
        'steps': original_steps,
        'values': original_values
    }

def main():
    print("Analyzing TensorBoard training curves...")
    print("=" * 60)
    
    logdir = '.cache/logs'
    data = extract_tensorboard_data(logdir)
    
    if not data:
        print("No TensorBoard data found!")
        return
    
    print("\nAvailable metrics:")
    for metric in data.keys():
        print(f"  - {metric}")
    
    # Fit curves for loss and PSNR metrics
    metrics_to_analyze = [
        ('Loss/Train', 'Train Loss'),
        ('Loss/Validation', 'Validation Loss'),
        ('PSNR/Train', 'Train PSNR'),
        ('PSNR/Validation', 'Validation PSNR')
    ]
    
    fits = {}
    
    for metric_key, metric_name in metrics_to_analyze:
        if metric_key in data:
            print(f"\nAnalyzing {metric_name}...")
            result = fit_and_predict(data[metric_key], metric_key)
            
            if result:
                fits[metric_key] = result
                current = result['current_value']
                final = result['final_predicted_value']
                
                # Calculate relative error for context
                if 'PSNR' in metric_key:
                    rel_error = (result['fit_quality'] / current) * 100
                    print(f"  Current value: {current:.2f} dB")
                    print(f"  Predicted final: {final:.2f} dB")
                    print(f"  Estimated convergence at epoch: {result['converged_epoch']}")
                    print(f"  Fit quality (RMSE): {result['fit_quality']:.2f} dB ({rel_error:.1f}% of current)")
                    print(f"  Model used: {result['model_name']}")
                else:
                    rel_error = (result['fit_quality'] / current) * 100
                    print(f"  Current value: {current:.6f}")
                    print(f"  Predicted final: {final:.6f}")
                    print(f"  Estimated convergence at epoch: {result['converged_epoch']}")
                    print(f"  Fit quality (RMSE): {result['fit_quality']:.6f} ({rel_error:.1f}% of current)")
                    print(f"  Model used: {result['model_name']}")
    
    # Create visualization
    print("\nGenerating visualization...")
    num_metrics = len([k for k in fits.keys()])
    
    if num_metrics == 0:
        print("No metrics fitted.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss (Train + Validation)
    ax = axes[0]
    if 'Loss/Train' in fits and 'Loss/Validation' in fits:
        # Train Loss
        train_fit = fits['Loss/Train']
        ax.plot(train_fit['steps'], train_fit['values'], 'b-o', label='Train Loss', markersize=4, linewidth=2, alpha=0.8)
        ax.plot(train_fit['future_steps'], train_fit['predictions'], 'b--', label='Train Predicted', linewidth=2, alpha=0.5)
        
        # Validation Loss
        val_fit = fits['Loss/Validation']
        ax.plot(val_fit['steps'], val_fit['values'], 'r-o', label='Validation Loss', markersize=4, linewidth=2, alpha=0.8)
        ax.plot(val_fit['future_steps'], val_fit['predictions'], 'r--', label='Val Predicted', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: PSNR (Train + Validation)
    ax = axes[1]
    if 'PSNR/Train' in fits and 'PSNR/Validation' in fits:
        # Train PSNR
        train_fit = fits['PSNR/Train']
        ax.plot(train_fit['steps'], train_fit['values'], 'b-o', label='Train PSNR', markersize=4, linewidth=2, alpha=0.8)
        ax.plot(train_fit['future_steps'], train_fit['predictions'], 'b--', label='Train Predicted', linewidth=2, alpha=0.5)
        
        # Validation PSNR
        val_fit = fits['PSNR/Validation']
        ax.plot(val_fit['steps'], val_fit['values'], 'r-o', label='Validation PSNR', markersize=4, linewidth=2, alpha=0.8)
        ax.plot(val_fit['future_steps'], val_fit['predictions'], 'r--', label='Val Predicted', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title('Training and Validation PSNR', fontsize=14)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = '.cache/frames/curve_fitting_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - CURVE FITTING ANALYSIS")
    print("=" * 60)
    
    if 'Loss/Validation' in fits:
        print("\nValidation Loss:")
        result = fits['Loss/Validation']
        print(f"  Current: {result['current_value']:.6f}")
        print(f"  Predicted final: {result['final_predicted_value']:.6f}")
        print(f"  Estimated convergence: epoch {result['converged_epoch']}")
    
    if 'PSNR/Validation' in fits:
        print("\nValidation PSNR:")
        result = fits['PSNR/Validation']
        print(f"  Current: {result['current_value']:.2f} dB")
        print(f"  Predicted final: {result['final_predicted_value']:.2f} dB")
        print(f"  Estimated convergence: epoch {result['converged_epoch']}")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
