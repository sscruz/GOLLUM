import numpy as np
import matplotlib.pyplot as plt
import os

import os 
import sys
sys.path.insert(0, '..')

import fit.Likelihood as lh 
import pickle  as pck 
import common.user as user 
import json 
from fit.Modeling import Rotated


import numpy as np
import matplotlib.pyplot as plt
import os

from RunImpactPlots import fit_result_to_dict


import matplotlib.colors as mcolors

# Base colors
colorA = 'red'
colorB = 'blue'

# Shades
colorA_pos = mcolors.to_rgba(colorA, alpha=0.85)
colorA_neg = mcolors.to_rgba('darkred', alpha=0.85)

colorB_pos = mcolors.to_rgba(colorB, alpha=0.85)
colorB_neg = mcolors.to_rgba('navy', alpha=0.85)


def impact_table_plot_two_results(
    # result A
    nuis_names_A, nuis_vals_A, nuis_errs_A,
    impact_pos_A, impact_neg_A,
    # result B
    nuis_names_B, nuis_vals_B, nuis_errs_B,
    impact_pos_B, impact_neg_B,
    poi_names,
    top_n=None,
    figsize=(14, 0.40*20),
    outpath=None,
    name_col_width=0.20,
    pull_col_width=0.26,
    poi_col_width=0.54,
    color_A='red',
    color_B='blue'
):
    """
    Create a multi-column impact table plotting *two* impact results side-by-side.
    Nuisance parameters are matched by name.

    Inputs for each result:
      - nuis_names    : list of nuisance names
      - nuis_vals     : pull central values
      - nuis_errs     : pull uncertainties
      - impact_pos    : (M, N) POI impact (+1 sigma)
      - impact_neg    : (M, N) POI impact (-1 sigma)

    Both results must use the same set of nuisance names (any order ok).
    """

    # Convert to arrays
    nuis_names_A = np.asarray(nuis_names_A, dtype=object)
    nuis_vals_A  = np.asarray(nuis_vals_A, dtype=float)
    nuis_errs_A  = np.asarray(nuis_errs_A, dtype=float)
    impact_pos_A = np.asarray(impact_pos_A, dtype=float)
    impact_neg_A = np.asarray(impact_neg_A, dtype=float)

    nuis_names_B = np.asarray(nuis_names_B, dtype=object)
    nuis_vals_B  = np.asarray(nuis_vals_B, dtype=float)
    nuis_errs_B  = np.asarray(nuis_errs_B, dtype=float)
    impact_pos_B = np.asarray(impact_pos_B, dtype=float)
    impact_neg_B = np.asarray(impact_neg_B, dtype=float)

    poi_names = list(poi_names)
    N = len(poi_names)

    # --- Align nuisance parameters between the two results ---
    names = np.asarray(sorted(set(nuis_names_A) | set(nuis_names_B)), dtype=object)
    M = len(names)

    # Build index maps
    index_A = {n: i for i, n in enumerate(nuis_names_A)}
    index_B = {n: i for i, n in enumerate(nuis_names_B)}

    # Allocate aligned arrays
    vals_A = np.zeros(M)
    errs_A = np.zeros(M)
    pos_A  = np.zeros((M, N))
    neg_A  = np.zeros((M, N))

    vals_B = np.zeros(M)
    errs_B = np.zeros(M)
    pos_B  = np.zeros((M, N))
    neg_B  = np.zeros((M, N))

    for i, n in enumerate(names):
        ia = index_A[n]
        ib = index_B[n]
        vals_A[i] = nuis_vals_A[ia]
        errs_A[i] = nuis_errs_A[ia]
        pos_A[i]  = impact_pos_A[ia]
        neg_A[i]  = impact_neg_A[ia]

        vals_B[i] = nuis_vals_B[ib]
        errs_B[i] = nuis_errs_B[ib]
        pos_B[i]  = impact_pos_B[ib]
        neg_B[i]  = impact_neg_B[ib]

    # combined magnitude for sorting
    mags = np.mean(np.abs(pos_A) + np.abs(neg_A) +
                   np.abs(pos_B) + np.abs(neg_B), axis=1)
    order = np.argsort(mags)[::-1]  # largest first

    if top_n is not None:
        order = order[:top_n]

    names = names[order]
    vals_A, errs_A = vals_A[order], errs_A[order]
    vals_B, errs_B = vals_B[order], errs_B[order]
    pos_A, neg_A = pos_A[order], neg_A[order]
    pos_B, neg_B = pos_B[order], neg_B[order]

    Msel = len(names)

    # Column widths
    col_widths = np.array([name_col_width, pull_col_width] +
                          [poi_col_width]*N)
    col_widths = col_widths / np.sum(col_widths)

    # --- Figure layout ---
    fig = plt.figure(figsize=figsize)
    left = 0.02; right = 0.98
    bottom = 0.03; top = 0.97
    full_width = right - left
    full_height = top - bottom
    spacing = 0.005

    axes = []
    cum = left
    for w in col_widths:
        wpix = full_width * w
        ax = fig.add_axes([cum, bottom, wpix-spacing, full_height])
        axes.append(ax)
        cum += wpix

    ax_name, ax_pull, *ax_pois = axes
    y = np.arange(Msel)

    # --- Column 1: names ---
    ax_name.set_xlim(0, 1)
    ax_name.set_ylim(-0.5, Msel - 0.5)
    ax_name.invert_yaxis()
    ax_name.axis('off')
    ax_name.set_title("Nuisance", fontsize=10, pad=8)

    for i, nm in enumerate(names):
        ax_name.text(0.98, i, nm, ha='right', va='center', fontsize=9)

    # --- Column 2: pulls (two markers per nuisance) ---
    ax_pull.set_ylim(-0.5, Msel - 0.5)
    ax_pull.invert_yaxis()
    ax_pull.set_yticks([])

    # symmetric x-lims over both results
    all_vals = np.concatenate([vals_A + errs_A, vals_A - errs_A,
                               vals_B + errs_B, vals_B - errs_B])
    lim = max(np.nanmax(np.abs(all_vals))*1.2, 0.5)
    ax_pull.set_xlim(-lim, lim)
    ax_pull.axvline(0, ls='--', lw=0.6)

    # pulls for A and B
    ax_pull.errorbar(vals_A, y+0.15, xerr=errs_A, fmt='o',
                     markersize=4, capsize=3, color=color_A)
    ax_pull.errorbar(vals_B, y-0.15, xerr=errs_B, fmt='s',
                     markersize=4, capsize=3, color=color_B)

    ax_pull.set_title("Pull (A ○ , B □)", fontsize=10, pad=8)
    ax_pull.tick_params(axis='x', labelsize=8)

    for j, ax in enumerate(ax_pois):
        ax.set_ylim(-0.5, Msel - 0.5)
        ax.invert_yaxis()
        ax.set_yticks([])

        # Combined limits
        all_impacts = np.concatenate([
            pos_A[:, j], neg_A[:, j],
            pos_B[:, j], neg_B[:, j]
        ])
        span = max(np.nanmax(np.abs(all_impacts)), 1e-6)
        ax.set_xlim(-1.2*span, 1.2*span)

        ax.axvline(0, ls='--', lw=0.6)
        ax.xaxis.grid(True, linestyle=':', lw=0.4, alpha=0.6)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_title(poi_names[j], fontsize=10, pad=8)

        # draw A and B with vertical offsets
        for i in range(Msel):
            # ===== Result A =====
            pA, nA = pos_A[i, j], neg_A[i, j]

            # Positive shift (A)
            if pA >= 0:
                ax.hlines(i+0.15, 0, pA, lw=5, alpha=1.0, color=colorA_pos)
            else:
                ax.hlines(i+0.15, pA, 0, lw=5, alpha=1.0, color=colorA_pos)

            # Negative shift (A)
            if nA <= 0:
                ax.hlines(i+0.15, nA, 0, lw=5, alpha=1.0, color=colorA_neg)
            else:
                ax.hlines(i+0.15, 0, nA, lw=5, alpha=1.0, color=colorA_neg)

            # ===== Result B =====
            pB, nB = pos_B[i, j], neg_B[i, j]

            # Positive shift (B)
            if pB >= 0:
                ax.hlines(i-0.15, 0, pB, lw=5, alpha=1.0, color=colorB_pos)
            else:
                ax.hlines(i-0.15, pB, 0, lw=5, alpha=1.0, color=colorB_pos)

            # Negative shift (B)
            if nB <= 0:
                ax.hlines(i-0.15, nB, 0, lw=5, alpha=1.0, color=colorB_neg)
            else:
                ax.hlines(i-0.15, 0, nB, lw=5, alpha=1.0, color=colorB_neg)


    # --- Save or show ---
    if outpath:
        d = os.path.dirname(outpath)
        if d and not os.path.exists(d):
            os.makedirs(d)
        plt.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return outpath
    else:
        plt.show()
        plt.close(fig)
        return None


def get_fit_impacts( base, version, hyp_for_fit):

    # Results from the first fit
    in_path = os.path.join(user.output_directory, f"{base}_{version}_impacts_initialfit.json")

    with open(in_path) as inf: 
        initial_fit=json.load( inf )
        initial_fit_dict = fit_result_to_dict(initial_fit)

    nuisances_names  = []
    nuisances_values = []
    nuisances_errors = []
    impacts_up=[]
    impacts_dn=[]
    POIs= hyp_for_fit.POIs

    for p in hyp_for_fit.nuisances:
        param_name = p.name 
        param = initial_fit_dict[param_name]
        
        nuisances_names .append( param_name )
        nuisances_values.append( param['value'] )
        nuisances_errors.append( param['error'] )
        
        def get_impacts_for_nuisance( varied_fit_result ):
            ret=[]
            with open(varied_fit_result) as inf:
                fit_var=json.load( inf )
                fit_var_dict = fit_result_to_dict(fit_var)

            for poi in POIs:
                ret.append( fit_var_dict[poi.name]['value'] - initial_fit_dict[poi.name]['value'])
            return ret 
                
        impacts_up.append( get_impacts_for_nuisance( os.path.join(user.output_directory, f"{base}_{version}_impacts_{param_name}_up.json")))
        impacts_dn.append( get_impacts_for_nuisance( os.path.join(user.output_directory, f"{base}_{version}_impacts_{param_name}_down.json")))
        print( param_name, impacts_up[-1])
    
    return nuisances_names, nuisances_values, nuisances_errors, impacts_up, impacts_dn, POIs

import common.yaml_loader as yaml_loader 

cfg = yaml_loader.load_yaml("../configs/binned_merged.yaml")
like_info = lh.load_likelihood(cfg)


hyp  = lh.build_hypothesis_from_likelihood(like_info, name="SR")
hyp_rot = Rotated(hyp, f"/scratch-cbe/users/robert.schoefbeck/SBIPDF/output/orthogonal_basis_unbinned_merged.json", name="Fisher-basis")

nuisances_names_A, nuisances_values_A, nuisances_errors_A, impacts_up_A, impacts_dn_A, poi_names = get_fit_impacts("unbinned_merged_rotate", "v1", hyp_rot)
nuisances_names_B, nuisances_values_B, nuisances_errors_B, impacts_up_B, impacts_dn_B, poi_names = get_fit_impacts("binned_merged_rotate"  , "v1", hyp_rot)


impact_table_plot_two_results(nuisances_names_A, nuisances_values_A, nuisances_errors_A, impacts_up_A, impacts_dn_A,
                              nuisances_names_B, nuisances_values_B, nuisances_errors_B, impacts_up_B, impacts_dn_B,
                              poi_names, outpath="twoImpacts")

