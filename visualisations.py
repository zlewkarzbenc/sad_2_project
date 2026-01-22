import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, kruskal


def plot_attractor_histogram(metadata_path, output_path):
    """Plots a histogram of the attractor ratios from the metadata CSV file."""
    
    metadata = pd.read_csv(metadata_path)
    attractors = metadata['attractors']
    plt.figure(figsize=(5, 4))
    plt.hist(attractors, bins=20)
    plt.xlabel('Ratio (%)')
    plt.ylabel('Count')
    plt.title('Atractor ratio in trajectories')
    plt.tight_layout()
    plt.show()
    
    filename = "attractor_histogram.png"
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, filename))


def plot_metric_comparison(df_metrics, metrics, output_path):
    """Plots comparison of specified metrics between BDE and MDL score functions."""

    BDE_metrics = df_metrics['BDE'][metrics]
    MDL_metrics = df_metrics['MDL'][metrics]

    # Add a column to identify the dataframe
    BDE_metrics["Score function"] = "BDE"
    MDL_metrics["Score function"] = "MDL"
    
    # Combine
    df_combined = pd.concat([BDE_metrics, MDL_metrics])

    # Melt for seaborn
    df_melt = df_combined.reset_index().melt(
        id_vars=["Dataset", "Score function"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Value"
    )

    # Plot
    plt.figure(figsize=(10, 8))
    sns.boxplot(
        data=df_melt,
        x="Metric",
        y="Value",
        hue="Score function",
        palette={"BDE": "skyblue", "MDL": "limegreen"}
    )

    plt.ylabel("Metric value")


    y_max = df_melt["Value"].max()  # for annotation

    for i, metric in enumerate(metrics):
        vals1 = BDE_metrics[metric].values
        vals2 = MDL_metrics[metric].values

        stat, pval = wilcoxon(vals1, vals2, alternative='two-sided')
        
        plt.text(
            i, y_max + 0.01, f"p={pval:.3f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
        
    plt.title("Comparison of metrics between DataFrames")
    plt.legend(title='Score function', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()

    filename = "metric_comparison.png"
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, filename))


def plot_parameter_effects(df_metrics_BDE, df_metrics_MDL, cols, output_path):

    """Plots the effect of different parameters on Precision metric for BDE and MDL score functions."""

    # Prepare data
    df_metrics_BDE.columns = df_metrics_BDE.columns.droplevel(level=0)
    df_metrics_BDE['Score function'] = 'BDE'
    df_metrics_MDL.columns = df_metrics_MDL.columns.droplevel(level=0)
    df_metrics_MDL['Score function'] = 'MDL'
    df_metrics = pd.concat([df_metrics_BDE, df_metrics_MDL], axis=0)

    # Extract parameters from index
    df_metrics = pd.DataFrame(df_metrics['Precision'])
    idx = df_metrics.index.to_series().astype(str)

    df_metrics['mode'] = idx.str.extract(r"_m_([^_]+)_")
    df_metrics['number_of_traj'] = idx.str.extract(r"_t_(\d+)")
    df_metrics['number_of_steps'] = idx.str.extract(r"_s_(\d+)")
    df_metrics['frequency'] = idx.str.extract(r"_f_(\d+)")

    cols.append('mode')

    # Plotting
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 5), sharey=True)

    for ax, col in zip(axes, cols):
        data_sort = df_metrics.sort_values(by=col)
        
        sns.violinplot(
            data=data_sort,
            x=col,
            y='Precision',
            inner='quartile',  # show quartiles inside violin
            hue=col,
            palette='Set2',
            ax=ax
        )


        # Prepare data for test
        unique_values = data_sort[col].unique()
        groups = [
        data_sort.loc[data_sort[col] == v, 'Precision'].dropna()
        for v in unique_values
        ]

        # Run test
        if len(groups) > 2:
            stat, pval = kruskal(*groups)
            ax.set_title(f'Precision by {col}, Kruskal p={pval:.3f}')
            
        else:
            stat, pval = wilcoxon(groups[0], groups[1])
            ax.set_title(f'Precision by {col}, Wilcoxon p={pval:.3f}')
            ax.legend().remove()
            
        ax.set_xlabel(col)
        ax.set_ylabel('Precision')
    
    plt.tight_layout()
    plt.show()

    filename = "parameter_effects.png"
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, filename))
