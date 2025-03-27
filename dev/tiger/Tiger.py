# -*- coding: utf-8 -*-
"""
Python Version of TIGER Model
---------------------------
This code splits and converts the R code of TIGER into Python code:
  1. Data preprocessing: including filtering TFs and target genes, adjusting the prior network.
  2. Helper functions: including prior network preprocessing (prior_pp).
  3. Main function TIGER: constructs inputs required by the Stan model, calls the Stan model for inference, and extracts posterior results.
  4. Stan model code: loaded from an external TIGER_C.stan file.

Author: Chen Chen (Converted version by ChatGPT)
Date: 2022-10-05 (Example conversion date)
"""

import numpy as np
import pandas as pd
import os
import scipy.linalg
from cmdstanpy import CmdStanModel
from sklearn.covariance import GraphicalLasso

# ------------------------------------------------------------------------------
# Helper function: prior network preprocessing
# ------------------------------------------------------------------------------

def prior_pp(prior, expr):
    """
    Filter low-confidence edges in the prior network using partial correlation.
    Similar to the R implementation using GeneNet.
    
    Parameters:
      prior: pd.DataFrame
          Prior regulatory network (adjacency matrix), rows are TFs, columns are target genes.
      expr: pd.DataFrame
          Normalized and log-transformed gene expression matrix, rows are genes.
    
    Returns:
      pd.DataFrame: Filtered prior network.
    """
    # Filter TFs and target genes that exist in the expression matrix
    tf = prior.index.intersection(expr.index)
    tg = prior.columns.intersection(expr.index)
    all_genes = np.unique(np.concatenate([tf, tg]))
    
    # Extract expression data for these genes (transposed like in R GeneNet)
    expr_sub = expr.loc[all_genes].T  # Samples × genes
    
    try:
        # Calculate correlation matrix
        corr_matrix = expr_sub.corr().values
        
        # Apply shrinkage to correlation matrix - similar to what GeneNet does
        n_genes = corr_matrix.shape[0]
        n_samples = expr_sub.shape[0]
        
        # Apply regularization to ensure positive definiteness
        shrinkage = min(0.2, 1/np.sqrt(n_samples))
        
        # Calculate shrunk correlation using Ledoit-Wolf like approach
        shrunk_corr = (1 - shrinkage) * corr_matrix + shrinkage * np.eye(n_genes)
        
        # Calculate approximate partial correlation using matrix inversion
        # This is similar to how GeneNet calculates it without graphical lasso
        try:
            # More numerically stable approach for matrix inversion
            precision_mat = scipy.linalg.pinvh(shrunk_corr)  # Pseudo-inverse for better stability
            
            # Convert precision to partial correlation
            diag_precision = np.sqrt(np.diag(precision_mat))
            partial_corr = -precision_mat / np.outer(diag_precision, diag_precision)
            np.fill_diagonal(partial_corr, 0)
        except:
            # If matrix inversion fails, fall back to just using the correlation
            print("Warning: Matrix inversion failed. Using correlation instead of partial correlation.")
            partial_corr = shrunk_corr
            np.fill_diagonal(partial_corr, 0)
        
    except Exception as e:
        print(f"Warning: Correlation calculation failed: {str(e)}. Using simpler approach.")
        # Fall back to a very simple correlation approach if all else fails
        corr_values = np.zeros((len(all_genes), len(all_genes)))
        
        # Calculate pairwise correlations manually if necessary
        for i, gene_i in enumerate(all_genes):
            for j, gene_j in enumerate(all_genes):
                if i != j:
                    try:
                        # Simple Pearson correlation
                        correlation = np.corrcoef(expr.loc[gene_i], expr.loc[gene_j])[0, 1]
                        corr_values[i, j] = correlation
                    except:
                        corr_values[i, j] = 0
        
        partial_corr = corr_values
    
    # Convert partial correlation coefficients to DataFrame
    coexp = pd.DataFrame(partial_corr, index=all_genes, columns=all_genes)
    
    # Take the part of the prior matrix that corresponds to the partial correlation submatrix
    P_ij = prior.loc[tf, tg].copy().astype(np.float64)  # Convert to float64 explicitly
    C_ij = coexp.loc[tf, tg] * P_ij.abs()
    
    # Compare the sign of prior edges and partial correlation
    sign_P = np.sign(P_ij)
    sign_C = np.sign(C_ij)
    
    # For edges with inconsistent signs, adjust the weight to a very small value (fuzzy)
    inconsistent = (sign_P * sign_C) < 0
    P_ij[inconsistent] = 1e-6
    
    # Remove all-zero TFs and genes
    P_ij = P_ij.loc[(P_ij != 0).any(axis=1), (P_ij != 0).any(axis=0)]
    return P_ij

# ------------------------------------------------------------------------------
# Main function: TIGER
# ------------------------------------------------------------------------------

def TIGER(expr, prior, method="VB", TFexpressed=True, signed=True, baseline=True, 
          psis_loo=False, seed=123, out_path=None, out_size=300,
          a_sigma=1, b_sigma=1, a_alpha=1, b_alpha=1,
          sigmaZ=10, sigmaB=1, tol=0.005):
    """
    TIGER main function, uses Bayesian modeling to infer regulatory network (W) and TF activity (Z).
    
    Parameters:
      expr: pd.DataFrame
          Gene expression matrix (rows are genes, columns are samples).
      prior: pd.DataFrame
          Prior regulatory network (adjacency matrix), rows are TFs, columns are target genes.
      method: str
          Inference method: "VB" (Variational Bayes) or "MCMC".
      TFexpressed: bool
          If True, only retain TFs expressed in the expression matrix.
      signed: bool
          If True, use signed prior network.
      baseline: bool
          Whether to include a baseline term.
      psis_loo: bool
          Whether to perform PSIS-LOO model evaluation (placeholder here).
      seed: int
          Random seed.
      out_path: str or None
          Output path to save results.
      out_size: int
          Number of posterior samples.
      Other parameters are model hyperparameter settings.
    
    Returns:
      dict: Contains the following keys
            - "W": Regulatory network matrix (target genes × TFs).
            - "Z": TF activity matrix (TFs × samples).
            - "TF_name": List of used TFs.
            - "TG_name": List of used target genes.
            - "sample_name": List of sample names.
            - "loocv", "elpd_loo": Model evaluation results (if psis_loo is True).
    """
    # 1. Data checking and preprocessing: filter TFs and target genes
    sample_name = expr.columns.tolist()
    if TFexpressed:
        TF_names = sorted(list(set(prior.index).intersection(expr.index)))
    else:
        TF_names = sorted(prior.index.tolist())
    TG_names = sorted(list(set(expr.index).intersection(prior.columns)))
    if len(TF_names) == 0 or len(TG_names) == 0:
        raise ValueError("Gene names in the input expression matrix do not match with the prior network.")
    
    # 2. Prior network preprocessing (call prior_pp in signed case)
    if signed:
        if set(TG_names).intersection(set(TF_names)):
            prior_sub = prior.loc[TF_names, TG_names].copy()
            prior_filtered = prior_pp(prior_sub, expr)
            if prior_filtered.shape[0] != len(TF_names):
                missing_tfs = set(TF_names) - set(prior_filtered.index)
                if missing_tfs:
                    TF_not_expressed_edge = prior.loc[list(missing_tfs), prior_filtered.columns]
                    # 将所有非零值改为1e-6，不保留符号
                    non_zero = TF_not_expressed_edge != 0
                    TF_not_expressed_edge[non_zero] = 1e-6
                    prior_filtered = pd.concat([prior_filtered, TF_not_expressed_edge])
                    prior_filtered = prior_filtered.sort_index()
                    prior_filtered = prior_filtered.loc[(prior_filtered != 0).any(axis=1)]
            P = prior_filtered
            TF_names = list(P.index)
            TG_names = list(P.columns)
        else:
            P = prior.loc[TF_names, TG_names].copy()
    else:
        P = prior.loc[TF_names, TG_names].copy()
    
    # 3. Construct model input: expression matrix X, prior vector P, etc.
    X = expr.loc[TG_names, :].values  # (n_genes x n_samples)
    n_genes, n_samples = X.shape
    n_TFs = len(TF_names)
    
    # Note: Original prior matrix has rows as TFs, columns as genes; Stan model requires prior stored in (genes x TFs) form
    P_matrix = P.values.T  # After transposition: (n_genes x n_TFs)
    P_vec = P_matrix.flatten()  # Flattened by column
    # Note: In Stan, indices start from 1, so here we convert numpy's 0-index to 1-index
    P_zero = np.where(P_vec == 0)[0] + 1
    P_nonzero = np.where(P_vec != 0)[0] + 1
    P_negs = np.where(P_vec == -1)[0] + 1
    P_poss = np.where(P_vec == 1)[0] + 1
    P_blur = np.where(np.isclose(P_vec, 1e-6))[0] + 1
    n_zero = len(P_zero)
    n_ones = len(P_nonzero)
    n_negs = len(P_negs)
    n_poss = len(P_poss)
    n_blur = len(P_blur)
    n_all = len(P_vec)
    
    data_to_model = {
        "n_genes": n_genes,
        "n_samples": n_samples,
        "n_TFs": n_TFs,
        "X": X,
        "P": P_vec,
        "P_zero": P_zero.tolist(),
        "P_ones": P_nonzero.tolist(),
        "P_negs": P_negs.tolist(),
        "P_poss": P_poss.tolist(),
        "P_blur": P_blur.tolist(),
        "n_zero": n_zero,
        "n_ones": n_ones,
        "n_negs": n_negs,
        "n_poss": n_poss,
        "n_blur": n_blur,
        "n_all": n_all,
        "sign": int(signed),
        "baseline": int(baseline),
        "psis_loo": int(psis_loo),
        "sigmaZ": sigmaZ,
        "sigmaB": sigmaB,
        "a_alpha": a_alpha,
        "b_alpha": b_alpha,
        "a_sigma": a_sigma,
        "b_sigma": b_sigma
    }
    
    # 4. Compile and run Stan model
    # Get the path to the Stan model file (in the same directory as this script)
    stan_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TIGER_C.stan")
    
    # Check if the file exists
    if not os.path.exists(stan_file):
        raise FileNotFoundError(f"Stan model file not found: {stan_file}")
    
    # Create the model using CmdStanModel
    model = CmdStanModel(stan_file=stan_file)
    
    if method == "VB":
        fit = model.variational(data=data_to_model, algorithm="meanfield",
                                seed=seed, iter=50000, tol_rel_obj=tol, output_samples=out_size)
    elif method == "MCMC":
        fit = model.sample(data=data_to_model, chains=1, seed=seed, max_treedepth=10,
                           iter_warmup=1000, iter_sampling=out_size, adapt_delta=0.99)
    else:
        raise ValueError("Unknown inference method, please choose 'VB' or 'MCMC'.")
    
    if out_path is not None:
        fit.save_csvfiles(dir=out_path)
    
    # 5. Post-processing: extract W and Z parameters, and rescale
    W_pos = np.zeros(n_all)
    if signed:
        if n_negs > 0:
            W_negs = fit.stan_variable("W_negs").flatten()
            for idx, pos in enumerate(P_negs):
                W_pos[pos-1] = W_negs[idx]
        if n_poss > 0:
            W_poss = fit.stan_variable("W_poss").flatten()
            for idx, pos in enumerate(P_poss):
                W_pos[pos-1] = W_poss[idx]
        if n_blur > 0:
            W_blur = fit.stan_variable("W_blur").flatten()
            for idx, pos in enumerate(P_blur):
                W_pos[pos-1] = W_blur[idx]
    else:
        W_ones = fit.stan_variable("W_ones").flatten()
        for idx, pos in enumerate(P_nonzero):
            W_pos[pos-1] = W_ones[idx]
    
    # Reconstruct regulatory matrix W: shape (n_genes x n_TFs)
    W_matrix = W_pos.reshape(n_genes, n_TFs, order='F')
    # Extract TF activity matrix Z: shape (n_TFs x n_samples)
    Z_vec = fit.stan_variable("Z", mean=True).flatten()  # Add mean=True to avoid warning
    Z_matrix = Z_vec.reshape((n_TFs, n_samples), order='F')
    
    # Perform simple rescaling with proper broadcasting
    col_sum = np.sum(np.abs(W_matrix), axis=0)  # Shape: (n_TFs,)
    col_count = np.count_nonzero(W_matrix, axis=0)  # Shape: (n_TFs,)
    col_count[col_count == 0] = 1  # Avoid division by zero
    scale_factor = (col_sum / col_count)[:, np.newaxis]  # Shape: (n_TFs, 1)
    IZ = Z_matrix * scale_factor  # Broadcasting will work correctly now
    
    row_sum_Z = np.sum(Z_matrix, axis=1)  # Shape: (n_TFs,)
    scale_factor_W = (row_sum_Z / n_samples)[:, np.newaxis]  # Shape: (n_TFs, 1)
    IW = W_matrix * scale_factor_W.T  # Transpose for correct broadcasting
    
    IW_df = pd.DataFrame(IW, index=TG_names, columns=TF_names)
    IZ_df = pd.DataFrame(IZ, index=TF_names, columns=sample_name)
    
    loocv = None   # PSIS-LOO part needs additional implementation
    elpd_loo = None
    
    return {
        "W": IW_df,
        "Z": IZ_df,
        "TF_name": TF_names,
        "TG_name": TG_names,
        "sample_name": sample_name,
        "loocv": loocv,
        "elpd_loo": elpd_loo
    }

# ------------------------------------------------------------------------------
# Example: How to call the TIGER model
# ------------------------------------------------------------------------------


expr = pd.read_csv("data/holland_rna_expr.tsv", index_col=0, sep="\t")
prior = pd.read_csv("data/tf_gene_matrix.csv", index_col=0)

result = TIGER(expr, prior, method="VB")
print("Inferred regulatory network (W):")
print(result["W"])
result["W"].to_csv("compare/W.csv")
print("\nInferred TF activities (Z):")
print(result["Z"])
result["Z"].to_csv("compare/Z.csv")
