import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spl
from scipy import stats
import tqdm
from test_MOU_largenet import MOUv2
import os
import pandas as pd
import seaborn as sb


def z_score_per_region(bold_signal: np.ndarray ) -> np.ndarray:
    """
    Z-score the BOLD signal for each region independently.
    
    Parameters
    ----------
    bold_signal (numpy.ndarray): BOLD signal, shape (time_points, n_regions).
    
    Returns
    ----------
    numpy.ndarray: The z-scored BOLD signal with the same shape as input.
    """
    # Compute mean and std for each region
    mean_per_region = np.mean(bold_signal, axis=0)
    std_per_region = np.std(bold_signal, axis=0, ddof=0)
    
    # Avoid division by zero
    std_per_region = np.where(std_per_region == 0, 1.0, std_per_region)
    
    # Z-score the signal
    z_scored_signal = (bold_signal - mean_per_region) / std_per_region
    
    return z_scored_signal

def load_and_process(data_path: str, folder_name: str, file_name: str, cut_transient: int) -> np.ndarray:    
    """Load and process BOLD and temporal average data for a given folder name (seed). Designed for the data output of MF AdEx simulations. 
    
    Parameters
    ----------
    data_path : str
        Path to the data directory.
    folder_name : str
        Name of the folder containing the data for a specific seed and condition.
    file_name : str
        Name of the file to load (either "bold.npy" or "temporal_average.npy").
    cut_transient : int
        Number of initial time points to cut from the data to remove transient effects.
    
    Returns
    -------
    np.ndarray
        Processed z-scored data array (of type BOLD or firing rate) for one subject (or seed if simuakated data) and condition.
        shape: (time_points - cut_transient, regions) 
    """
    # Load bold or temporal average for a given seed
    signal_array = np.load(os.path.join(data_path, folder_name, file_name))
    n_timepoints, _, n_regions = signal_array.shape
    # Reshape from (time_points, 1, regions) to (time_points, regions)
    signal_array = signal_array.reshape(n_timepoints, n_regions)
    # Cut initial transient
    signal_array = signal_array[cut_transient:, :]
    # Z-score signal
    signal_array = z_score_per_region(signal_array)

    return signal_array

def process_activity_multisub(bold_multisub: np.ndarray, cut_transient: int) -> np.ndarray:
    """Process BOLD or firing rates data for multiple subjects by cutting transient and z-scoring each subject independently.
    
    Parameters
    ----------
    bold_multisub : np.ndarray
        Raw BOLD or firing rate data for multiple subjects, shape (n_subjects, time_points, regions).
    cut_transient : int
        Number of initial time points to cut from the data to remove transient effects.
    
    Returns
    -------
    np.ndarray
        Processed activity data for multiple subjects, shape (n_subjects, time_points - cut_transient, regions).
    """
    n_subjects, n_timepoints, n_regions = bold_multisub.shape
    processed_activity = np.zeros((n_subjects, n_timepoints - cut_transient, n_regions))
    
    for i in range(n_subjects):
        processed_activity[i] = z_score_per_region(bold_multisub[i, cut_transient:, :])
    
    return processed_activity

def lagged_fc_matrices(X: np.ndarray, n_tau: int = 2, diag_zero: bool = True, z_score: bool = False
                       ) -> np.ndarray:
    """ Compute lagged functional connectivity matrices from time series data.
    
    Parameters
    ----------
    X : np.ndarray | jnp.ndarray
        Z-scored BOLD time series data of shape (time_points, n_nodes).
    n_tau : int
        Number of time lags to compute (default is 2, which computes FC0 and FC1).
    diag_zero : bool
        Whether to set diagonal elements to zero (default is True).

    Returns
    -------
    Q : np.ndarray
        Lagged FC matrices of shape (n_tau, n_nodes, n_nodes).
    """
    # Optionally z-score the input time series if required
    if z_score:
        X = z_score_per_region(X)

    # Get dimensions
    n_T, n_nodes = X.shape
    # Lag (time-shifted) FC matrices
    #Q_emp = np.zeros([n_tau, n_nodes, n_nodes], dtype=float)
    # Remove mean in the time series
    centered_X = X - np.mean(X, axis=0)
    n_T_span = n_T - n_tau + 1
    
    def one_tau(i_tau):
        return np.tensordot(
            centered_X[0:n_T_span],
            centered_X[i_tau:n_T_span + i_tau],
            axes=(0, 0)
        )

    Q = np.stack([one_tau(i) for i in range(n_tau)], axis=0)
    Q = Q / (n_T_span - 1)

    if diag_zero:
        Q = Q * (1.0 - np.eye(n_nodes)[None, :, :])

    return Q

def plot_bold(bold_onesub):
    cmap = plt.cm.PuRd_r
    n_nodes = np.shape(bold_onesub)[-1]
    norm = plt.Normalize(vmin=0, vmax=n_nodes - 1)

    fig, ax = plt.subplots(figsize=(14,6))

    for roi in range(n_nodes):
        ax.plot(
            range(bold_onesub.shape[0]),
            bold_onesub[ :, roi], color=cmap(norm(roi)), lw=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('ROI index', fontsize=12)

    #ax.set_xlim(0, new_array.shape[2] - 1)
    ax.set_xlabel('Time (TR)', fontsize=16)
    ax.set_ylabel('Z-score BOLD activity', fontsize=16)
    ax.set_title(f"Filtered BOLD activity one subject", fontsize = 16)

def corr_sc_fc(fc_mat, off_diag, sc_vec):
    """Compute the Pearson correlation between the structural connectivity vector and the functional connectivity vector 
    (off-diagonal elements only)."""
    fc_vec = fc_mat[off_diag]
    if np.std(sc_vec) == 0 or np.std(fc_vec) == 0:
        return np.nan
    return np.corrcoef(sc_vec, fc_vec)[0, 1]

def p_to_stars(p):
    """Convert a p-value to a string of stars for significance levels."""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"

def plot_sc_fc_corr(values, title, file_name, conds, n_sub, palette, presentation_figs_dir, graph_format='pdf'):
    """ Plot SC-FC correlation for two conditions with a violin plot and statistical comparison using Mann-Whitney U test."""
    # Build the dataframe with the correct subject-condition pairing
    data = pd.DataFrame({
        "Condition": [conds[0]] * n_sub + [conds[1]] * n_sub,
        "SC_FC_correlation": np.concatenate([values[:, 0], values[:, 1]])
    }).dropna(subset=["SC_FC_correlation"])

    g1 = data.loc[data["Condition"] == conds[0], "SC_FC_correlation"]
    g2 = data.loc[data["Condition"] == conds[1], "SC_FC_correlation"]
    stat, p_value = stats.mannwhitneyu(g1, g2, alternative="two-sided")

    plt.figure(figsize=(10, 6))
    ax = sb.violinplot(
        data=data,
        x="Condition",
        y="SC_FC_correlation",
        order=conds,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1,
        alpha=0.44
    )
    sb.stripplot(
        data=data,
        x="Condition",
        y="SC_FC_correlation",
        order=conds,
        color="gray",
        alpha=0.8,
        jitter=0.2,
        size=4,
        ax=ax
    )

    y_min = data["SC_FC_correlation"].min()
    y_max = data["SC_FC_correlation"].max()
    y_range = max(y_max - y_min, 1e-9)
    bar_height = y_max + 0.05 * y_range
    bar_height_diff = 0.02 * y_range

    x1, x2 = 0, 1
    ax.plot(
        [x1, x1, x2, x2],
        [bar_height, bar_height + bar_height_diff, bar_height + bar_height_diff, bar_height],
        "k-",
        linewidth=1.5
    )
    ax.text(
        (x1 + x2) * 0.5,
        bar_height + bar_height_diff,
        f"{p_to_stars(p_value)}\np = {p_value:.3g}",
        ha="center",
        va="bottom"
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Condition")
    ax.set_ylabel("SC-FC correlation")
    ax.set_ylim(y_min - 0.05 * y_range, bar_height + 0.15 * y_range)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(presentation_figs_dir, file_name), format=graph_format, dpi=300)
    plt.show()
    plt.close()

    print(title)
    print(f"Mann-Whitney U = {stat:.4f}, p = {p_value:.4g}")
    print()

def plot_sc_fc_corr_bold_fr(data, order, res_dir, palette, presentation_figs_dir, graph_format='pdf', stat_bars = True):
    """ Plot SC-FC correlation for both BOLD and firing rate conditions together, with optional statistical comparison bars.
    Each group is a situation (for ex, BOLD or firing rate) and condition (for ex, baseline or reduced gK)."""
    # Optional stats containers
    pairwise_df = None
    H_stat, H_pval = None, None
    
    if stat_bars:
        # Pairwise tests
        pairwise_results = []
        pairs_to_test = [
            (0, 1),  # baseline BOLD vs reduced gK BOLD
            (2, 3),  # baseline FR vs reduced gK FR
            (0, 2),  # baseline BOLD vs baseline FR
            (1, 3),  # reduced gK BOLD vs reduced gK FR
            (0, 3),  # baseline BOLD vs reduced gK FR
            (1, 2),  # reduced gK BOLD vs baseline FR
        ]

        for i, j in pairs_to_test:
            g1 = data.loc[data["Group"] == order[i], "SC_FC_correlation"]
            g2 = data.loc[data["Group"] == order[j], "SC_FC_correlation"]
            stat, pval = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            pairwise_results.append({
                "Group_1": order[i],
                "Group_2": order[j],
                "Mann-Whitney U": stat,
                "p-value": pval,
            })

        pairwise_df = pd.DataFrame(pairwise_results)
        pairwise_df.to_csv(os.path.join(res_dir, "SC_FC_pairwise_stats.csv"), index=False)

        # Global test across the 4 groups
        groups = [data.loc[data["Group"] == g, "SC_FC_correlation"].values for g in order]
        H_stat, H_pval = stats.kruskal(*groups)
        pd.DataFrame({
            "Kruskal-Wallis H": [H_stat],
            "p-value": [H_pval],
        }).to_csv(os.path.join(res_dir, "SC_FC_global_stats.csv"), index=False)


    plt.figure(figsize=(11, 6))
    ax = sb.violinplot(
        data=data,
        x="Group",
        y="SC_FC_correlation",
        order=order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1,
        alpha=0.44
    )
    sb.stripplot(
        data=data,
        x="Group",
        y="SC_FC_correlation",
        order=order,
        color="gray",
        alpha=0.8,
        jitter=0.2,
        size=4,
        ax=ax
    )

    if stat_bars:
        # Significance bars
        y_min = data["SC_FC_correlation"].min()
        y_max = data["SC_FC_correlation"].max()
        y_range = max(y_max - y_min, 1e-9)
        bar_base = y_max + 0.001 * y_range
        bar_step = 0.14 * y_range
        bar_height = 0.01 * y_range

        for k, row in enumerate(pairwise_results):
            x1 = order.index(row["Group_1"])
            x2 = order.index(row["Group_2"])
            y = bar_base + k * bar_step

            ax.plot(
                [x1, x1, x2, x2],
                [y, y + bar_height, y + bar_height, y],
                c="k",
                lw=1.2
            )
            ax.text(
                (x1 + x2) / 2,
                y + bar_height,
                f"{p_to_stars(row['p-value'])}\np = {row['p-value']:.3g}",
                ha="center",
                va="bottom",
                fontsize=8
            )

            ax.set_ylim(y_min - 0.05 * y_range, bar_base + len(pairwise_results) * bar_step + 0.12 * y_range)

    ax.set_title("SC-FC correlation by condition and modality - AdEx MF simulations", fontsize=16)
    ax.set_xlabel("Group")
    ax.set_ylabel("SC-FC correlation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if stat_bars:
        plt.savefig(os.path.join(presentation_figs_dir, f"SC_FC_correlation_all4_sig.{graph_format}"), format=graph_format, dpi=300)
    else:
        plt.savefig(os.path.join(presentation_figs_dir, f"SC_FC_correlation_all4.{graph_format}"), format=graph_format, dpi=300)
    plt.show()
    plt.close()

    if stat_bars:
        print("Global Kruskal-Wallis:")
        print(f"H = {H_stat:.4f}, p = {H_pval:.4g}")
        print()
        print("Pairwise Mann-Whitney results:")
        print(pairwise_df)

def run_MOU_fitting(n_sub, n_cond, Q_data, N, situations, situation_idx, mask_Sigma, min_iter,
                    presentation_figs_dir, graph_format, conds, 
                    res_dir = None, coef_S=None, coef_E=None, verbose=False):
    # Test: save number of iterations to see if algorithm converged
    iter_mod = np.zeros([n_sub, n_cond], dtype=int)
    # arrays to store results
    J_mod = np.zeros([n_sub,n_cond,N,N]) # model Jacobian (EC.) In the article of Entropy production B=-J
    Sigma_mod = np.zeros([n_sub,n_cond,N,N]) # model sigma. In the article of Entropy production sigma=2D
    FC0_mod = np.zeros([n_sub,n_cond,N,N]) # model zero-lad FC
    err_mod = np.zeros([n_sub,n_cond]) # model error
    fit_mod = np.zeros([n_sub,n_cond]) # model goodness of fit (Pearson corr)
    avg_dist_FC0 = 0
    avg_dist_FC1 = 0
    
    # loop over subject and conditions
    for i_sub in tqdm.tqdm(range(n_sub)):
        for i_cond in range(n_cond):
            # get time series
            Q_data_temp = Q_data[:,i_sub,:,:,i_cond]
            print("Q_data_temp shape:", Q_data_temp.shape)
            FC1 = Q_data_temp[1]
            lim_FC = 0.1 # limit DTI value to determine SC (only connections with larger values are tuned)
            mask_EC = np.zeros([N,N],dtype=bool) # EC weights to tune
            mask_EC[FC1>lim_FC] = True
            for i in range(N):
                mask_EC[i,i] = False # no self connection
                mask_EC[i,N-1-i] = True # additional interhemispherical connections
            ##debugging
            fig_dbg, ax_dbg = plt.subplots(1, 2, figsize=(12, 5))
            ax_dbg[0].imshow(Q_data_temp[0], vmin=-0.5, vmax=0.5, cmap='bwr')
            # add a colorbar
            plt.colorbar(ax_dbg[0].imshow(Q_data_temp[0], vmin=-0.5, vmax=0.5, cmap='cividis'), ax=ax_dbg[0], fraction=0.046, pad=0.04)
            ax_dbg[0].set_title(f"{situations[situation_idx].capitalize()} FC0 - sub {i_sub}, cond {i_cond}")
            ax_dbg[1].imshow(Q_data_temp[1], cmap='bwr')
            ax_dbg[1].set_title(f"{situations[situation_idx].capitalize()} FC1 - sub {i_sub}, cond {i_cond}")
            # add a colorbar
            plt.colorbar(ax_dbg[1].imshow(Q_data_temp[1], cmap='cividis'), ax=ax_dbg[1], fraction=0.046, pad=0.04)
            #
            # invert model
            mou_est = MOUv2()
            mou_est.fit_Q(Q_emp = Q_data_temp, mask_C=mask_EC, mask_Sigma=mask_Sigma, algo_version="true", min_iter=min_iter, epsilon_C=0.005, epsilon_Sigma=0.005*N,
                        track_eigs=True, eigs_every=1, eigs_store_full=False, verbose=True)

            # test : save nb of iterations
            iter_mod[i_sub, i_cond] = mou_est.d_fit["iterations"]

            # store results
            J_mod[i_sub,i_cond,:,:] = mou_est.J
            Sigma_mod[i_sub,i_cond,:,:] = mou_est.Sigma
            FC0_mod[i_sub,i_cond,:,:] = spl.solve_lyapunov(J_mod[i_sub,i_cond,:,:].T, -Sigma_mod[i_sub,i_cond,:,:])
            err_mod[i_sub,i_cond] = mou_est.d_fit['distance']
            fit_mod[i_sub,i_cond] = mou_est.d_fit['correlation']
            
            # Update average distances for FC0 and FC1
            avg_dist_FC0 += mou_est.d_fit['distFC0']
            avg_dist_FC1 += mou_est.d_fit['distFC1']

            if verbose:
                print('sub', i_sub, '; cond', i_cond, ':', mou_est.d_fit)
                
                d = mou_est.d_fit
                n_it = int(d["iterations"])

                # Error btw model and data for FC0 and FC1 (matrix distance) -> = 0.5 * (dist_Q0 + dist_Qtau)
                dist_hist = np.asarray(d["distance history"][:n_it])
                # Corr between model and empirical data for FC0 and FC1 -> =  0.5 * (Pearson_Q0  + Pearson_Qtau)
                corr_hist = np.asarray(d["correlation history"][:n_it])

                # Print eigenvalues
                #for iter_idx, eigs in enumerate(d['eigs_history']):
                #   print(f"Iteration {iter_idx}: min real part of eigenvalues = {np.min(np.real(eigs)):.4f}, max real part of eigenvalues = {np.max(np.real(eigs)):.4f}")    

                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
                ax[0].plot(dist_hist, 'o-', markersize=4, lw=2)
                ax[0].set_title("Average error between data and model FC")
                ax[0].set_xlabel("Iteration")
                ax[0].set_ylabel("Matrix distance")
                ax[0].grid(alpha=0.3)

                ax[1].plot(corr_hist, 'o-', markersize=4, lw=2, color="crimson")
                ax[1].set_title("Average correlation between data and model FC")
                
                ax[1].set_xlabel("Iteration")
                ax[1].set_ylabel("Pearson correlation")
                ax[1].grid(alpha=0.3)

                plt.suptitle(f"MOU fit evolution - subject {i_sub}, condition {conds[i_cond]} - {situations[situation_idx]} data", fontsize=16)
                plt.tight_layout()
                # Save figure
                if situation_idx == 2:
                    fig_name = f"MOU_fit_evolution_{situations[situation_idx]}_S_{coef_S}_E_{coef_E}_sub{i_sub}_cond{conds[i_cond]}.{graph_format}"
                else:
                    fig_name = f"MOU_fit_evolution_{situations[situation_idx]}_sub{i_sub}_cond{conds[i_cond]}.{graph_format}"
                plt.savefig(os.path.join(presentation_figs_dir, fig_name), format=graph_format, dpi=300)
                plt.show()
                plt.close()

                # Plot evolution of max and min real part of eigenvalues across iterations
                plt.figure(figsize=(6, 3))
                plt.plot(d['eig_iter'], d['eig_max_real'], 'o-', markersize=4, color = '#B9314F',label='Max Real Part')
                plt.plot(d['eig_iter'], d['eig_min_real'], 'x-', markersize=4, color = '#ACE4AA', label='Min Real Part')
                plt.axhline(0, color='gray', lw=0.5)
                plt.axvline(0, color='gray', lw=0.5)
                plt.title(f"Real part of J eigenvalues \nSubject {i_sub}, Condition {conds[i_cond]} - {situations[situation_idx]} data", fontsize=14)
                plt.ylabel("Real part")
                plt.xlabel("# Iteration")
                plt.grid(alpha=0.3)
                plt.legend()
                if situation_idx == 2:
                    plt.savefig(os.path.join(presentation_figs_dir, f"eigenvalues_real_part_{situations[situation_idx]}_S_{coef_S}_E_{coef_E}_sub{i_sub}_cond{conds[i_cond]}.{graph_format}"), format=graph_format, dpi=300)
                else:
                    plt.savefig(os.path.join(presentation_figs_dir, f"eigenvalues_real_part_{situations[situation_idx]}_sub{i_sub}_cond{conds[i_cond]}.{graph_format}"), format=graph_format, dpi=300)
                plt.show()
                plt.close()

                # Plot evolution of Q0 and Q1 consistency according to Q1 = Q0 * exp(J.T)
                plt.figure(figsize=(6, 3))
                plt.plot(np.arange(d['iterations']),d['cons_Q0_Q1_history'], 'x-', markersize = 4, color = '#7F95D1')
                plt.xlabel('# Iteration')
                plt.ylabel('|Q1 - Q0*exp(J)|/|Q1|')
                plt.axhline(0, color='gray', lw=0.5)
                plt.axvline(0, color='gray', lw=0.5)
                plt.title(f'Consistency between Q0 and Q1 during optimization\nSubject {i_sub}, condition {conds[i_cond]} - {situations[situation_idx]} data')
                plt.grid(alpha=0.3)
                plt.legend()
                if situation_idx == 2:
                    plt.savefig(os.path.join(presentation_figs_dir, f"consistency_Q0_Q1_{situations[situation_idx]}_S_{coef_S}_E_{coef_E}_sub{i_sub}_cond{conds[i_cond]}.{graph_format}"), format=graph_format, dpi=300)
                else:
                    plt.savefig(os.path.join(presentation_figs_dir, f"consistency_Q0_Q1_{situations[situation_idx]}_sub{i_sub}_cond{conds[i_cond]}.{graph_format}"), format=graph_format, dpi=300)
                plt.show()
                plt.close()

                # Save a figure with the final J for this subject and condition
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(mou_est.J, cmap='cividis', aspect='auto')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"Final Jacobian J - subject {i_sub}, condition {conds[i_cond]}")
                ax.set_xlabel("Source ROI")
                ax.set_ylabel("Target ROI")
                plt.tight_layout()

                if situation_idx == 2:
                    fig_name = f"J_final_{situations[situation_idx]}_S_{coef_S}_E_{coef_E}_sub{i_sub}_cond{conds[i_cond]}.{graph_format}"
                else:
                    fig_name = f"J_final_{situations[situation_idx]}_sub{i_sub}_cond{conds[i_cond]}.{graph_format}"

                plt.savefig(os.path.join(presentation_figs_dir, fig_name), format=graph_format, dpi=300)
                plt.close()

                print(f'Maximum value of the J matrix: {np.max(mou_est.J):.4f}')
                print(f'Minimum value of the J matrix: {np.min(mou_est.J):.4f}')
                print(f'Mean value of the J matrix: {np.mean(mou_est.J):.4f}')
                print(f'Standard deviation of the J matrix: {np.std(mou_est.J):.4f}')

    if verbose:
        print("Mean iterations:", iter_mod.mean())
        print("Max iterations:", iter_mod.max())
        print("Mean distance FC0:", avg_dist_FC0 / (n_sub * n_cond))
        print("Mean distance FC1:", avg_dist_FC1 / (n_sub * n_cond))

    # Save results if directory of results provided
    if res_dir is not None:
        np.save(os.path.join(res_dir,'J_mod.npy'),J_mod)
        np.save(os.path.join(res_dir, 'Sigma_mod.npy'),Sigma_mod)
        np.save(os.path.join(res_dir,'FC0_mod.npy'),FC0_mod)
        np.save(os.path.join(res_dir,'err_mod.npy'),err_mod)
        np.save(os.path.join(res_dir,'fit_mod.npy'),fit_mod)


def plot_mod_diag(array_mod, n_sub, n_cond, conds,
                   cols, y_min, y_max, plot_ylabel, file_name, graph_format, res_dir, xlabel=None):
    """Plot model array (e.g. error) as violin plot with swarmplot overlay"""
    # ensure condition strings and dataframe in the same order as `conds`
    conds_str = [str(c) for c in conds]
    array_mod_tmp = dict()
    array_mod_tmp['sum EC'] = array_mod.flatten()
    array_mod_tmp['sub'] = np.repeat(np.arange(n_sub), n_cond).flatten()
    array_mod_tmp['cond'] = np.repeat(np.array(conds_str).reshape([-1, n_cond]), n_sub, axis=0).flatten()

    array_mod_df = pd.DataFrame(array_mod_tmp)

    if y_min is None:
        y_min = array_mod_df['sum EC'].min() * 0.9
    if y_max is None:
        y_max = array_mod_df['sum EC'].max() * 1.1

    # Build palette mapping cond -> color (accept `cols` as list or dict)
    if isinstance(cols, dict):
        palette = {str(k): v for k, v in cols.items()}
    else:
        # cols is a list of colors; map first n_cond colors to conds order
        palette = {conds_str[i]: cols[i % len(cols)] for i in range(len(conds_str))}

    plt.figure()
    sb.violinplot(data=array_mod_df, x='cond', y='sum EC', palette=palette, order=conds_str)
    sb.swarmplot(data=array_mod_df, x='cond', y='sum EC', color=[0.7, 0.7, 0.7], order=conds_str)
    plt.axis(ymin=y_min, ymax=y_max)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if xlabel is None:
        xlabel = 'Condition'
    plt.xlabel(xlabel,fontsize=16)
    plt.ylabel(plot_ylabel, fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, file_name), format=graph_format)

def plot_mod_matrix(mod_matrix, mod_matrix_name, res_dir, graph_format):
    plt.figure(figsize=(5,4))
    plt.imshow(np.maximum(mod_matrix[0,0,:,:],0), cmap='Reds')   # Show only the non-negatives
    #plt.imshow(mod_matrix[0,0,:,:], cmap='Reds')   
    plt.colorbar()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Target ROI index', fontsize=16)
    plt.ylabel('Source ROI index', fontsize=16)
    plt.title(mod_matrix_name, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir,f"{mod_matrix_name}_mod.{graph_format}"), format=graph_format)