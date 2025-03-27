import numpy as np
import pandas as pd
import os
import scipy.linalg
from cmdstanpy import CmdStanModel 

def prior_pp(prior, expr, edge_type="default", enforce_negative=False):
    """
    处理先验网络，使用部分相关来过滤低置信度边。
    
    Parameters:
      prior: pd.DataFrame
          先验调控网络(邻接矩阵)，行为调控因子，列为靶标
      expr: pd.DataFrame
          标准化、对数转换后的表达矩阵，行为基因/miRNA
      edge_type: str
          边的类型：'tf_mirna', 'tf_gene', 'mirna_gene'或'default'
      enforce_negative: bool
          是否强制将所有非零边设为负值(不再默认使用)
          
    Returns:
      pd.DataFrame: 过滤后的先验网络
    """
    # 过滤在表达矩阵中存在的调控因子和靶标
    regulators = prior.index.intersection(expr.index)
    targets = prior.columns.intersection(expr.index)
    all_genes = np.unique(np.concatenate([regulators, targets]))
    
    # 提取这些基因的表达数据
    expr_sub = expr.loc[all_genes].T  # 样本 × 基因
    
    try:
        # 计算相关矩阵
        corr_matrix = expr_sub.corr().values
        
        # 应用收缩以确保正定性
        n_genes = corr_matrix.shape[0]
        n_samples = expr_sub.shape[0]
        
        shrinkage = min(0.2, 1/np.sqrt(n_samples))
        shrunk_corr = (1 - shrinkage) * corr_matrix + shrinkage * np.eye(n_genes)
        
        # 计算部分相关系数
        try:
            precision_mat = scipy.linalg.pinvh(shrunk_corr)
            diag_precision = np.sqrt(np.diag(precision_mat))
            partial_corr = -precision_mat / np.outer(diag_precision, diag_precision)
            np.fill_diagonal(partial_corr, 0)
        except:
            print("警告: 矩阵求逆失败。改用相关系数替代部分相关。")
            partial_corr = shrunk_corr
            np.fill_diagonal(partial_corr, 0)
            
    except Exception as e:
        print(f"警告: 相关计算失败: {str(e)}。使用简化方法。")
        # 简单相关计算的回退实现
        corr_values = np.zeros((len(all_genes), len(all_genes)))
        
        for i, gene_i in enumerate(all_genes):
            for j, gene_j in enumerate(all_genes):
                if i != j:
                    try:
                        correlation = np.corrcoef(expr.loc[gene_i], expr.loc[gene_j])[0, 1]
                        corr_values[i, j] = correlation
                    except:
                        corr_values[i, j] = 0
        
        partial_corr = corr_values
    
    # 转换部分相关系数为DataFrame
    coexp = pd.DataFrame(partial_corr, index=all_genes, columns=all_genes)
    
    # 提取先验矩阵对应于部分相关子矩阵的部分
    P_ij = prior.loc[regulators, targets].copy().astype(np.float64)
    C_ij = coexp.loc[regulators, targets].abs() * P_ij.abs()
    
    # 比较先验边和部分相关性的符号
    sign_P = np.sign(P_ij)
    sign_C = np.sign(coexp.loc[regulators, targets])
    
    # 对符号不一致的边，将权重调整为很小的值(模糊边)
    inconsistent = (sign_P * sign_C) < 0
    P_ij[inconsistent] = 1e-6
    
    # 只有当明确要求时才强制负值（移除mirna_gene的特殊处理）
    if enforce_negative:
        non_zero = P_ij != 0
        P_ij[non_zero] = -abs(P_ij[non_zero])
    
    # 移除全零的调控因子和靶标
    P_ij = P_ij.loc[(P_ij != 0).any(axis=1), (P_ij != 0).any(axis=0)]
    
    return P_ij

def prepare_network_data(prior_tf_mirna, prior_tf_gene, prior_mirna_gene, 
                        expr_gene, expr_mirna, TFexpressed=True, signed=True):
    """
    准备适合RegInsight模型的多层次网络数据
    """
    # 检查样本是否匹配
    if not np.all(expr_gene.columns == expr_mirna.columns):
        raise ValueError("基因和miRNA表达矩阵样本必须一致")
    
    sample_name = expr_gene.columns.tolist()
    
    # 1. 过滤数据阶段 - 类似TIGER的处理逻辑
    # 找出所有潜在的TF、miRNA和基因
    all_TFs = sorted(list(set(prior_tf_mirna.index).union(prior_tf_gene.index)))
    if TFexpressed:
        all_TFs = sorted(list(set(all_TFs).intersection(expr_gene.index)))
        
    all_miRNAs = sorted(list(set(prior_tf_mirna.columns).intersection(expr_mirna.index)))
    all_genes = sorted(list(set(prior_tf_gene.columns).intersection(expr_gene.index)))
    
    # 检查数据是否有效
    if len(all_TFs) == 0 or len(all_miRNAs) == 0 or len(all_genes) == 0:
        raise ValueError("输入数据中基因/miRNA名称不匹配")
    
    # 处理三种先验网络
    network_data = {}
    
    # 2. 先验网络处理和过滤
    if signed:
        # TF-miRNA网络
        tf_mz_common = sorted(list(set(all_TFs).intersection(prior_tf_mirna.index)))
        tf_mirna_net = prior_tf_mirna.loc[tf_mz_common, all_miRNAs].copy()
        P_mz = prior_pp(tf_mirna_net, pd.concat([expr_gene, expr_mirna]), edge_type='tf_mirna')
        
        # TF-gene网络
        tf_gz_common = sorted(list(set(all_TFs).intersection(prior_tf_gene.index)))
        tf_gene_net = prior_tf_gene.loc[tf_gz_common, all_genes].copy()
        P_gz = prior_pp(tf_gene_net, pd.concat([expr_gene, expr_mirna]), edge_type='tf_gene')
        
        # miRNA-gene网络 - 不再强制为负值
        mirna_common = sorted(list(set(all_miRNAs).intersection(prior_mirna_gene.index)))
        gene_common_for_mirna = sorted(list(set(all_genes).intersection(prior_mirna_gene.columns)))
        mirna_gene_net = prior_mirna_gene.loc[mirna_common, gene_common_for_mirna].copy()
        P_gm = prior_pp(mirna_gene_net, pd.concat([expr_gene, expr_mirna]), edge_type='mirna_gene')
        
        # 3. 补充处理 - 处理prior_pp可能丢失的TF和miRNA (类似TIGER)
        # 检查TF-miRNA网络中是否有TF被过滤掉
        missing_tfs_mz = set(tf_mz_common) - set(P_mz.index)
        if missing_tfs_mz:
            missing_tf_edges = prior_tf_mirna.loc[list(missing_tfs_mz), P_mz.columns].copy()
            missing_tf_edges = missing_tf_edges.replace(1, 1e-6)  # 模糊处理
            missing_tf_edges = missing_tf_edges.replace(-1, -1e-6)  # 模糊处理但保留符号
            P_mz = pd.concat([P_mz, missing_tf_edges])
            P_mz = P_mz.loc[(P_mz != 0).any(axis=1)]  # 移除全零行
        
        # 检查TF-gene网络中是否有TF被过滤掉
        missing_tfs_gz = set(tf_gz_common) - set(P_gz.index)
        if missing_tfs_gz:
            missing_tf_edges = prior_tf_gene.loc[list(missing_tfs_gz), P_gz.columns].copy()
            missing_tf_edges = missing_tf_edges.replace(1, 1e-6)
            missing_tf_edges = missing_tf_edges.replace(-1, -1e-6)
            P_gz = pd.concat([P_gz, missing_tf_edges])
            P_gz = P_gz.loc[(P_gz != 0).any(axis=1)]
        
        # 检查miRNA-gene网络中是否有miRNA被过滤掉
        missing_mirnas = set(mirna_common) - set(P_gm.index)
        if missing_mirnas:
            missing_mirna_edges = prior_mirna_gene.loc[list(missing_mirnas), P_gm.columns].copy()
            missing_mirna_edges = missing_mirna_edges.replace(1, 1e-6)
            missing_mirna_edges = missing_mirna_edges.replace(-1, -1e-6)
            P_gm = pd.concat([P_gm, missing_mirna_edges])
            P_gm = P_gm.loc[(P_gm != 0).any(axis=1)]
    else:
        # 无符号网络处理
        tf_mz_common = sorted(list(set(all_TFs).intersection(prior_tf_mirna.index)))
        P_mz = prior_tf_mirna.loc[tf_mz_common, all_miRNAs].copy()
        
        tf_gz_common = sorted(list(set(all_TFs).intersection(prior_tf_gene.index)))
        P_gz = prior_tf_gene.loc[tf_gz_common, all_genes].copy()
        
        mirna_common = sorted(list(set(all_miRNAs).intersection(prior_mirna_gene.index)))
        gene_common_for_mirna = sorted(list(set(all_genes).intersection(prior_mirna_gene.columns)))
        P_gm = prior_mirna_gene.loc[mirna_common, gene_common_for_mirna].copy()
    
    # 4. 确保三个网络的实体集合一致性
    # 更新处理后的TF、miRNA和基因名称
    TF_names = sorted(list(set(P_mz.index).union(P_gz.index)))
    miRNA_names = sorted(list(set(P_mz.columns).intersection(P_gm.index)))
    gene_names = sorted(list(set(P_gz.columns).intersection(P_gm.columns)))
    
    # 5. 如果有TF只出现在一个网络中，需要在另一个网络中为其添加零行
    for tf in TF_names:
        if tf not in P_mz.index:
            P_mz.loc[tf] = np.zeros(len(P_mz.columns))
        if tf not in P_gz.index:
            P_gz.loc[tf] = np.zeros(len(P_gz.columns))
    
    # 同样处理miRNA和gene
    for mirna in miRNA_names:
        if mirna not in P_mz.columns:
            P_mz[mirna] = np.zeros(len(P_mz.index))
        if mirna not in P_gm.index:
            P_gm.loc[mirna] = np.zeros(len(P_gm.columns))
    
    for gene in gene_names:
        if gene not in P_gz.columns:
            P_gz[gene] = np.zeros(len(P_gz.index))
        if gene not in P_gm.columns:
            P_gm[gene] = np.zeros(len(P_gm.index))
    
    # 按名称排序
    P_mz = P_mz.loc[TF_names, miRNA_names]
    P_gz = P_gz.loc[TF_names, gene_names]
    P_gm = P_gm.loc[miRNA_names, gene_names]
    
    network_data['TF_names'] = TF_names
    network_data['miRNA_names'] = miRNA_names
    network_data['gene_names'] = gene_names
    network_data['P_mz'] = P_mz
    network_data['P_gz'] = P_gz
    network_data['P_gm'] = P_gm
    network_data['sample_names'] = sample_name
    
    return network_data

def RegInsight(expr_gene, expr_mirna, 
               prior_tf_mirna, prior_tf_gene, prior_mirna_gene,
               method="VB", TFexpressed=True, signed=True, baseline=True,
               psis_loo=False, seed=123, out_path=None, out_size=300,
               a_sigma=1, b_sigma=1, a_alpha=1, b_alpha=1,
               sigmaZ=10, sigmaB=1, tol=0.005):
    """
    RegInsight主函数，使用贝叶斯建模推断多层次调控网络(TF-miRNA-gene)和TF活性(Z)。
    
    Parameters:
      expr_gene: pd.DataFrame
          基因表达矩阵(行为基因，列为样本)
      expr_mirna: pd.DataFrame
          miRNA表达矩阵(行为miRNA，列为样本)
      prior_tf_mirna: pd.DataFrame
          TF-miRNA先验调控网络(邻接矩阵)，行为TF，列为miRNA
      prior_tf_gene: pd.DataFrame
          TF-基因先验调控网络，行为TF，列为基因
      prior_mirna_gene: pd.DataFrame
          miRNA-基因先验调控网络，行为miRNA，列为基因
      method: str
          推断方法: "VB" (变分贝叶斯) 或 "MCMC"
      TFexpressed: bool
          如果为True，只保留表达矩阵中表达的TF
      signed: bool
          如果为True，使用有符号的先验网络
      baseline: bool
          是否包含基线表达项
      psis_loo: bool
          是否执行PSIS-LOO模型评估
      seed: int
          随机种子
      out_path: str or None
          保存结果的输出路径
      out_size: int
          后验采样数量
      其他参数为模型超参数设置
    
    Returns:
      dict: 包含以下键的字典
            - "W_mz": TF到miRNA的调控网络矩阵
            - "W_gz": TF到基因的调控网络矩阵
            - "W_gm": miRNA到基因的调控网络矩阵
            - "Z": TF活性矩阵(TF × 样本)
            - "TF_names": 使用的TF列表
            - "miRNA_names": 使用的miRNA列表
            - "gene_names": 使用的基因列表
            - "sample_names": 样本名称列表
            - "loocv_mirna", "loocv_gene": miRNA和基因模型评估结果(如果psis_loo为True)
    """
    # 1. 数据预处理：整合网络并过滤
    network_data = prepare_network_data(
        prior_tf_mirna, prior_tf_gene, prior_mirna_gene,
        expr_gene, expr_mirna, TFexpressed, signed
    )
    
    # 提取网络数据
    TF_names = network_data['TF_names']
    miRNA_names = network_data['miRNA_names']
    gene_names = network_data['gene_names']
    sample_names = network_data['sample_names']
    P_mz = network_data['P_mz']  # TF -> miRNA
    P_gz = network_data['P_gz']  # TF -> gene
    P_gm = network_data['P_gm']  # miRNA -> gene
    
    # 2. 构建模型输入：表达矩阵、先验向量等
    X_gene = expr_gene.loc[gene_names, sample_names].values  # (n_genes x n_samples)
    X_mirna = expr_mirna.loc[miRNA_names, sample_names].values  # (n_mirnas x n_samples)
    
    n_genes, n_samples = X_gene.shape
    n_mirnas = X_mirna.shape[0]
    n_TFs = len(TF_names)
    
    # 处理TF-miRNA网络
    P_mz_matrix = P_mz.values.T  # 转置: (n_mirnas x n_TFs)
    P_mz_vec = P_mz_matrix.flatten()  # 按列展平
    # Stan中索引从1开始，这里转换numpy的0索引为1索引
    P_mz_zero = np.where(P_mz_vec == 0)[0] + 1
    P_mz_nonzero = np.where(P_mz_vec != 0)[0] + 1
    P_mz_negs = np.where(P_mz_vec == -1)[0] + 1
    P_mz_poss = np.where(P_mz_vec == 1)[0] + 1
    P_mz_blur = np.where(np.isclose(P_mz_vec, 1e-6))[0] + 1
    n_mz_zero = len(P_mz_zero)
    n_mz_ones = len(P_mz_nonzero)
    n_mz_negs = len(P_mz_negs)
    n_mz_poss = len(P_mz_poss)
    n_mz_blur = len(P_mz_blur)
    n_mz_all = len(P_mz_vec)
    
    # 处理TF-gene网络
    P_gz_matrix = P_gz.values.T  # 转置: (n_genes x n_TFs)
    P_gz_vec = P_gz_matrix.flatten()  # 按列展平
    P_gz_zero = np.where(P_gz_vec == 0)[0] + 1
    P_gz_nonzero = np.where(P_gz_vec != 0)[0] + 1
    P_gz_negs = np.where(P_gz_vec == -1)[0] + 1
    P_gz_poss = np.where(P_gz_vec == 1)[0] + 1
    P_gz_blur = np.where(np.isclose(P_gz_vec, 1e-6))[0] + 1
    n_gz_zero = len(P_gz_zero)
    n_gz_ones = len(P_gz_nonzero)
    n_gz_negs = len(P_gz_negs)
    n_gz_poss = len(P_gz_poss)
    n_gz_blur = len(P_gz_blur)
    n_gz_all = len(P_gz_vec)
    
    # 处理miRNA-gene网络
    P_gm_matrix = P_gm.values.T  # 转置: (n_genes x n_mirnas)
    P_gm_vec = P_gm_matrix.flatten()  # 按列展平
    P_gm_zero = np.where(P_gm_vec == 0)[0] + 1
    P_gm_nonzero = np.where(P_gm_vec != 0)[0] + 1
    n_gm_zero = len(P_gm_zero)
    n_gm_ones = len(P_gm_nonzero)
    n_gm_all = len(P_gm_vec)
    
    # 3. 构建Stan模型的数据输入
    data_to_model = {
        # 维度信息
        "n_genes": n_genes,
        "n_mirnas": n_mirnas,
        "n_samples": n_samples,
        "n_TFs": n_TFs,
        
        # 表达数据
        "X_gene": X_gene,
        "X_mirna": X_mirna,
        
        # TF-miRNA网络
        "n_mz_zero": n_mz_zero,
        "n_mz_ones": n_mz_ones,
        "n_mz_negs": n_mz_negs,
        "n_mz_poss": n_mz_poss,
        "n_mz_blur": n_mz_blur,
        "n_mz_all": n_mz_all,
        "P_mz": P_mz_vec,
        "P_mz_zero": P_mz_zero.tolist(),
        "P_mz_ones": P_mz_nonzero.tolist(),
        "P_mz_negs": P_mz_negs.tolist(),
        "P_mz_poss": P_mz_poss.tolist(),
        "P_mz_blur": P_mz_blur.tolist(),
        
        # TF-gene网络
        "n_gz_zero": n_gz_zero,
        "n_gz_ones": n_gz_ones,
        "n_gz_negs": n_gz_negs,
        "n_gz_poss": n_gz_poss,
        "n_gz_blur": n_gz_blur,
        "n_gz_all": n_gz_all,
        "P_gz": P_gz_vec,
        "P_gz_zero": P_gz_zero.tolist(),
        "P_gz_ones": P_gz_nonzero.tolist(),
        "P_gz_negs": P_gz_negs.tolist(),
        "P_gz_poss": P_gz_poss.tolist(),
        "P_gz_blur": P_gz_blur.tolist(),
        
        # miRNA-gene网络
        "n_gm_zero": n_gm_zero,
        "n_gm_ones": n_gm_ones,
        "n_gm_all": n_gm_all,
        "P_gm": P_gm_vec,
        "P_gm_zero": P_gm_zero.tolist(),
        "P_gm_ones": P_gm_nonzero.tolist(),
        
        # 模型配置
        "sign": int(signed),
        "baseline": int(baseline),
        "psis_loo": int(psis_loo),
        
        # 超参数
        "sigmaZ": sigmaZ,
        "sigmaB": sigmaB,
        "a_alpha": a_alpha,
        "b_alpha": b_alpha,
        "a_sigma": a_sigma,
        "b_sigma": b_sigma
    }
    
    # 4. 编译并运行Stan模型
    # 获取Stan模型文件路径（与此脚本位于同一目录）
    stan_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RegInsight_C.stan")
    
    
    # 使用CmdStanModel创建模型
    model = CmdStanModel(stan_file=stan_file)
    
    if method == "VB":
        fit = model.variational(data=data_to_model, algorithm="meanfield",
                                seed=seed, iter=50000, tol_rel_obj=tol, output_samples=out_size)
    elif method == "MCMC":
        fit = model.sample(data=data_to_model, chains=1, seed=seed, max_treedepth=10,
                           iter_warmup=1000, iter_sampling=out_size, adapt_delta=0.99)
    else:
        raise ValueError("未知的推断方法，请选择'VB'或'MCMC'。")
    
    if out_path is not None:
        fit.save_csvfiles(dir=out_path)
    
    # 5. 后处理：提取参数并重新缩放
    # 提取TF-miRNA网络权重
    W_mz_pos = np.zeros(n_mz_all)
    if signed:
        if n_mz_negs > 0:
            W_mz_negs = fit.stan_variable("W_mz_negs").flatten()
            for idx, pos in enumerate(P_mz_negs):
                W_mz_pos[pos-1] = W_mz_negs[idx]
        if n_mz_poss > 0:
            W_mz_poss = fit.stan_variable("W_mz_poss").flatten()
            for idx, pos in enumerate(P_mz_poss):
                W_mz_pos[pos-1] = W_mz_poss[idx]
        if n_mz_blur > 0:
            W_mz_blur = fit.stan_variable("W_mz_blur").flatten()
            for idx, pos in enumerate(P_mz_blur):
                W_mz_pos[pos-1] = W_mz_blur[idx]
    else:
        W_mz_ones = fit.stan_variable("W_mz_ones").flatten()
        for idx, pos in enumerate(P_mz_nonzero):
            W_mz_pos[pos-1] = W_mz_ones[idx]
    
    # 提取TF-gene网络权重
    W_gz_pos = np.zeros(n_gz_all)
    if signed:
        if n_gz_negs > 0:
            W_gz_negs = fit.stan_variable("W_gz_negs").flatten()
            for idx, pos in enumerate(P_gz_negs):
                W_gz_pos[pos-1] = W_gz_negs[idx]
        if n_gz_poss > 0:
            W_gz_poss = fit.stan_variable("W_gz_poss").flatten()
            for idx, pos in enumerate(P_gz_poss):
                W_gz_pos[pos-1] = W_gz_poss[idx]
        if n_gz_blur > 0:
            W_gz_blur = fit.stan_variable("W_gz_blur").flatten()
            for idx, pos in enumerate(P_gz_blur):
                W_gz_pos[pos-1] = W_gz_blur[idx]
    else:
        W_gz_ones = fit.stan_variable("W_gz_ones").flatten()
        for idx, pos in enumerate(P_gz_nonzero):
            W_gz_pos[pos-1] = W_gz_ones[idx]
    
    # 提取miRNA-gene网络权重
    W_gm_pos = np.zeros(n_gm_all)
    W_gm_ones = fit.stan_variable("W_gm_ones").flatten()
    for idx, pos in enumerate(P_gm_nonzero):
        W_gm_pos[pos-1] = W_gm_ones[idx]
    
    # 重构权重矩阵
    W_mz_matrix = W_mz_pos.reshape(n_mirnas, n_TFs, order='F')  # 按列重构
    W_gz_matrix = W_gz_pos.reshape(n_genes, n_TFs, order='F')
    W_gm_matrix = W_gm_pos.reshape(n_genes, n_mirnas, order='F')
    
    # 提取TF活性矩阵Z
    Z_vec = fit.stan_variable("Z", mean=True).flatten()
    Z_matrix = Z_vec.reshape((n_TFs, n_samples), order='F')
    
    # 执行简单的缩放与TIGER类似
    # 对TF-miRNA网络缩放
    mz_col_sum = np.sum(np.abs(W_mz_matrix), axis=0)  # Shape: (n_TFs,)
    mz_col_count = np.count_nonzero(W_mz_matrix, axis=0)  # Shape: (n_TFs,)
    mz_col_count[mz_col_count == 0] = 1  # 避免除零
    mz_scale_factor = (mz_col_sum / mz_col_count)[:, np.newaxis]  # Shape: (n_TFs, 1)
    
    # 对TF-gene网络缩放
    gz_col_sum = np.sum(np.abs(W_gz_matrix), axis=0)
    gz_col_count = np.count_nonzero(W_gz_matrix, axis=0)
    gz_col_count[gz_col_count == 0] = 1
    gz_scale_factor = (gz_col_sum / gz_col_count)[:, np.newaxis]
    
    # 计算TF活性缩放因子
    row_sum_Z = np.sum(Z_matrix, axis=1)  # Shape: (n_TFs,)
    scale_factor_Z = (row_sum_Z / n_samples)[:, np.newaxis]  # Shape: (n_TFs, 1)
    
    # 应用缩放
    IZ = Z_matrix * np.maximum(mz_scale_factor, gz_scale_factor)  # 使用较大的缩放因子
    IW_mz = W_mz_matrix * scale_factor_Z.T  # 转置以正确广播
    IW_gz = W_gz_matrix * scale_factor_Z.T
    
    # 对miRNA-gene网络不做额外缩放，因为它们已经通过其他网络隐式缩放
    IW_gm = W_gm_matrix
    
    # 转换为DataFrame
    IW_mz_df = pd.DataFrame(IW_mz, index=miRNA_names, columns=TF_names)
    IW_gz_df = pd.DataFrame(IW_gz, index=gene_names, columns=TF_names)
    IW_gm_df = pd.DataFrame(IW_gm, index=gene_names, columns=miRNA_names)
    IZ_df = pd.DataFrame(IZ, index=TF_names, columns=sample_names)
    
    # 模型评估结果
    loocv_mirna = None
    loocv_gene = None
    elpd_loo_mirna = None
    elpd_loo_gene = None
    
    if psis_loo:
        # 这部分需要根据实际需要实现PSIS-LOO计算
        # 这里只是占位
        pass
    
    return {
        "W_mz": IW_mz_df,             # TF→miRNA网络
        "W_gz": IW_gz_df,             # TF→gene网络
        "W_gm": IW_gm_df,             # miRNA→gene网络
        "Z": IZ_df,                   # TF活性矩阵
        "TF_names": TF_names,
        "miRNA_names": miRNA_names,
        "gene_names": gene_names, 
        "sample_names": sample_names,
        "loocv_mirna": loocv_mirna,   # miRNA模型评估
        "loocv_gene": loocv_gene,     # gene模型评估
        "elpd_loo_mirna": elpd_loo_mirna,
        "elpd_loo_gene": elpd_loo_gene
    }

expr_gene = pd.read_csv("data/holland_rna_expr.tsv", index_col=0, sep="\t")
expr_mirna = pd.read_csv("data/holland_miRNA_exp.csv", index_col=0).T
prior_tf_mirna = pd.read_csv("data/tf_mir_matrix.csv", index_col=0)
prior_tf_gene = pd.read_csv("data/tf_gene_matrix.csv", index_col=0)
prior_mirna_gene = pd.read_csv("data/mir_gene_matrix.csv", index_col=0)



# 使用示例
result = RegInsight(expr_gene, expr_mirna, 
                    prior_tf_mirna, prior_tf_gene, prior_mirna_gene,
                    method="VB")

# 分析结果
W_mz = result["W_mz"]  # TF到miRNA的调控网络
W_gz = result["W_gz"]  # TF到gene的调控网络
W_gm = result["W_gm"]  # miRNA到gene的调控网络
Z = result["Z"]        # TF活性矩阵