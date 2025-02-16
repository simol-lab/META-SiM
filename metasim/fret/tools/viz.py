"""Library for visualizations used in META-SiM."""
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as plt_colors
import metasim.fret.core.atlas as atlas
from sklearn import mixture
from sciplotlib import style as spstyle


def plot_umap(umap_coord, label, color, color_name, color_map='plasma',label_order=None):
    """Plots the UMAP figures for traces grouped by label values.

    This function generates the UMAP figures used in the META-SiM paper.

    Args:
        umap_coord: The 2D coordinates of the umap, as a numpy tensor.
        label: The list of label values, as a numpy array.
        color: The color label for all traces, as a numpy array.
        color_name: The name of the color label, such as entropy.
        color_map: The name of colormap. If not set, use plasma as default.
        label_order: A list of labels to plot. If set, order the figures with these label values.

    Returns:
        A figure of UMAP plots.
    """
    if label_order is None:
        label_order = sorted(list(set(label.tolist())))
    counter = 0
    ax_min = np.min(umap_coord)
    ax_max = np.max(umap_coord)
    with plt.style.context(spstyle.get_style('nature-reviews')):
        fig, axes = plt.subplots(ncols=len(label_order), nrows=1, figsize=(5 * len(label_order), 5));
    for l in label_order:
        idx = (label == l)
        z = umap_coord[idx, ...]
        with plt.style.context(spstyle.get_style('nature-reviews')):
            if not isinstance(axes, np.ndarray):
                axes = [axes]  # handles single figure case.
            ax = axes[counter]
            counter += 1
            s = ax.scatter(z[:, 0], z[:, 1], c=color[idx], s=30, alpha=0.6, linewidths=0, cmap=color_map, vmin=np.quantile(color, 0.01), vmax=np.quantile(color, 0.99))
            ax.set_xlim([ax_min, ax_max])
            ax.set_ylim([ax_min, ax_max])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            ax.set_title(f'{l}')
            ax.text(x=ax.get_xlim()[0]+ 0.1, y=ax.get_ylim()[0] + 0.1, s=f'n={z.shape[0]}', fontdict={'size': 12})

    color_bar = fig.colorbar(s, ax=axes, label=color_name)
    color_bar.set_alpha(1)
    return fig, axes


def get_umap_reducer(embedding, n_neighbors=30, n_epochs=500):
    """Fits a 2-D UMAP reducer given embeddings.

    In this function, we fix several parameters in the UMAP model
    in order to obtain stable and robust UMAP output.
    The fixed parameters are:
        1. random_state
        2. negative_sample_rate
        3. min_dist

    Args:
        embedding: The embedding tensor.
        n_neighbors: The number of neighbors used for UMAP optimization.
        n_epochs: The epochs number of UMAP optimization.

    Returns:
        A umap reducer. The reducer can be used by:
        umap_coord = reducer.transform(embedding)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import umap
        reducer = umap.UMAP(
            n_epochs=n_epochs, n_neighbors=n_neighbors, min_dist=0.0,
            random_state=np.random.RandomState(60), n_components=2,
            negative_sample_rate=15)
        reducer.fit(embedding)
    return reducer


def plot_atlas_reference():
    """Plots the Atlas boundaries and legends in a reference figure.

    This figure can be used as a reference when reading the Atlas UMAP.

    Returns:
        A figure of the Atlas reference.
    """
    return _plot_atlas_reference()

def _plot_atlas_reference(
        fig=None, ax=None, clean_only=False, use_legend=True,
):
    """Internal function for ploting Atlas boundaries and legends in a reference figure.

    This figure can be used as a reference when reading the Atlas UMAP.

    Returns:
        A figure of the Atlas reference.
    """
    saved_atlas_density_models = atlas.get_atlas_density_model()
    with plt.style.context(spstyle.get_style('nature-reviews')):
        if fig is None:
            fig = plt.figure(figsize=(9, 6))
        if ax is None:
            ax = fig.add_axes([0.2, 0.17, 0.68, 0.7])
        contour_counter = 0
        for l in saved_atlas_density_models:
            xmin = -10
            xmax = 22
            ymin = -11
            ymax = 18
            color = plt.cm.tab20.colors + plt.cm.tab20b.colors[::2]
            x = np.linspace(xmin, xmax, num=200)
            y = np.linspace(ymin, ymax, num=200)
            X, Y = np.meshgrid(x, y)
            clf = saved_atlas_density_models[l]
            XX = np.array([X.ravel(), Y.ravel()]).T
            Z = clf.score_samples(XX)
            Z = np.exp(Z.reshape(X.shape))
            cmap = plt_colors.LinearSegmentedColormap.from_list("", [color[contour_counter], color[contour_counter], color[contour_counter]])
            if (not clean_only) or ('-c-' in l):
                CS = ax.contour(
                    X, Y, Z, cmap=cmap, levels=[1e-2]
                )
                if use_legend:
                    if hasattr(CS, 'collections'):
                            CS.collections[0].set_label(l)
                    elif hasattr(CS, 'get_paths'):
                            CS.get_paths()[0].set_label(l)

            contour_counter += 1
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_aspect('equal')
        ax.grid(True)
        if use_legend:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=4)
            plt.title('smFRET Atlas')


def plot_atlas(embedding, label, color, color_name, label_order=None):
    """Plots the Atlas UMAP figures for traces grouped by label values.

    This function generates the Atlas figures used in the META-SiM paper.

    Args:
        embedding: The embedding of traces, generated by model.encode().
        label: The list of label values, as a numpy array.
        color: The color label for all traces, as a numpy array.
        color_name: The name of the color label, such as entropy.
        label_order: A list of labels to plot. If set, order the figures with these label values.

    Returns:
        A figure of Atlas UMAP plots.
    """
    umap_coord = atlas.get_atlas_2d(embedding)
    fig, axes = plot_umap(umap_coord, label=label, color=color,
                          color_name=color_name, label_order=label_order)
    saved_atlas_density_models = atlas.get_atlas_density_model()
    for ax in axes:
        _plot_atlas_reference(
            fig=fig, ax=ax, clean_only=True, use_legend=False,
        )
    return fig, axes


def plot_fret_histograms(fret_efficiency, label, label_order=None):
    """Plots the FRET histogram figures for traces grouped by label values.

    This function generates the Atlas figures used in the META-SiM paper.

    Args:
        fret_efficiency: The list of FRET efficiency, generated by get_fret_efficiency().
        label: The list of label values, as a numpy array.
        label_order: A list of labels to plot. If set, order the figures with these label values.

    Returns:
        A figure of FRET efficiency plots.
    """
    if label_order is None:
        label_order = sorted(list(set(label.tolist())))

    with plt.style.context(spstyle.get_style('nature-reviews')):
        fig, axes = plt.subplots(ncols=len(label_order), nrows=1, figsize=(5 * len(label_order), 5));
        if not isinstance(axes, np.ndarray):
            axes = [axes]

    count = 0
    for l in label_order:
        idx = (label == l)
        values = []
        for i, found in enumerate(idx):
            if found:
                value = fret_efficiency[i]
                if value.shape[0] > 0:
                    values.append(value)
        if values:
            values = np.concatenate(values, axis=0)
        with plt.style.context(spstyle.get_style('nature-reviews')):
            ax = axes[count]
            ax.hist(values, bins=50, range=[-0.3, 1.2], color=(85 / 255, 154 / 255, 209 / 255), alpha=1, edgecolor='w');
            ax.set_xlim([-0.3, 1.2])
            ax.set_title(l)
            ax.set_xlabel('FRET Value')
            ax.set_ylabel('Count')
            ax.text(x=ax.get_xlim()[0]+ 0.1, y=ax.get_ylim()[0] + 0.1, s=f'n={np.sum(idx)}', fontdict={'size': 12})
            count += 1
    return fig, axes



