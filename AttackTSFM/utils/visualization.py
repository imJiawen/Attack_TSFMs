import matplotlib.pyplot as plt

def plot_results_with_quantile_adv(
    input_data=None, 
    perturbed_input_data=None, 
    perturbation=None, 
    target_data=None, 
    transformed_target_data=None, 
    forecast_df=None, 
    forecast_df_perturbed=None, 
    title=None,
    figsize=(12, 4),
    grid=True,
    axvline=True,
    legend=True,
    fontsize=12,
    linewidth=1.5,
    no_ticks=False,
    xlabel=None,
    ylabel=None,
    save_path=None,
    attack_color=None,
    point_only=False,
):
    """
    Plot forecast results with quantiles and adversarial perturbations.
    
    Args:
        input_data (array-like): Past input data.
        perturbed_input_data (array-like): Perturbed past input data.
        target_data (array-like): Ground truth target data.
        forecast_df (array-like): Forecast data with quantiles (q=0.1, 0.5, 0.9).
        forecast_df_perturbed (array-like): Perturbed forecast data with quantiles.
        title (str): Title of the plot.
    """
    # Ground truth and input ranges
    past_input = next(
    (v for v in [input_data, perturbed_input_data, perturbation] if v is not None),
        None  
    )
    if past_input is not None:
        input_len = len(past_input)
    else:
        input_len = 0
    
    future_target = next(
    (v for v in [target_data, transformed_target_data, forecast_df, forecast_df_perturbed] if v is not None),
        None  
    )
    if future_target is not None:
        target_len = len(future_target)
    else:
        target_len = 0
    
    x_past = range(input_len)
    x_future = range(input_len, input_len + target_len)

    plt.figure(figsize=figsize)

    # Plot forecast with quantiles
    if forecast_df is not None:
        if forecast_df.shape[-1] == 3:
            low, median, high = 0,1,2
        elif forecast_df.shape[-1] == 10:
            low, median, high = 1,5,9
        else:
            low, high = 0, -1
            median = int(forecast_df.shape[-1] // 2)
        
        plt.plot(x_future, forecast_df[:, median], label="Raw Predition (q=0.5)", color="green", linestyle="--", linewidth=linewidth)
        
        if not point_only:
            plt.fill_between(
                x_future, forecast_df[:, low], forecast_df[:, high], color="green", alpha=0.2, label="Raw Uncertainty (q=0.1 - q=0.9)"
            )
    
    # Plot perturbed forecast with quantiles
    if forecast_df_perturbed is not None:
        if forecast_df_perturbed.shape[-1] == 3:
            low, median, high = 0,1,2
        elif forecast_df_perturbed.shape[-1] == 10:
            low, median, high = 1,5,9
        else:
            low, high = 0, -1
            median = int(forecast_df_perturbed.shape[-1] // 2)
            
        plt.plot(x_future, forecast_df_perturbed[:, median], label="Attacked Prediction (q=0.5)", color="crimson", linestyle="dashdot", linewidth=linewidth)
        
        if not point_only:
            plt.fill_between(
                x_future, forecast_df_perturbed[:, low], forecast_df_perturbed[:, high], color="crimson", alpha=0.2, label="Attacked Uncertainty (q=0.1 - q=0.9)"
            )

    # Plot ground truth data
    if target_data is not None:
        plt.plot(x_future, target_data, label="Ground Truth (Target)", color="black", linewidth=linewidth)
        
    if perturbation is not None:
        plt.plot(x_past, perturbation, label="Perturbation", color="gray", linewidth=linewidth)
        
    # Plot past input data
    if input_data is not None:
        plt.plot(x_past, input_data, label="Raw Input", color="black", linewidth=linewidth)
    
    # Plot perturbed past input data
    if perturbed_input_data is not None:
        # "dodgerblue"
        if attack_color:
            plt.plot(x_past, perturbed_input_data, label="Perturbated Input", color=attack_color, linewidth=linewidth)
        else:
            plt.plot(x_past, perturbed_input_data, label="Perturbated Input", color="dodgerblue", alpha=0.7, linewidth=linewidth) 
        
    if transformed_target_data is not None:
        plt.plot(x_future, transformed_target_data, label="Transformed Target", color="blue", alpha=0.6, linewidth=linewidth)

    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize)
        
    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize)
    
    if legend:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=fontsize)
    if grid:
        plt.grid(True, alpha=0.3)
    
    if no_ticks:
        plt.xticks([])
        plt.yticks([])
    else:
        # plt.xticks(rotation=45, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

    # Add title if provided
    if title:
        plt.title(title, fontsize=fontsize)

    # Adjust layout and display the plot
    if axvline and target_len > 0:
        plt.axvline(x=input_len, color='gray', linestyle=':', linewidth=2, label='Forecast Start')
    
    plt.style.use('default')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

