data {
  // 样本和数量信息
  int<lower=0> n_genes;                       // 基因数量
  int<lower=0> n_mirnas;                      // miRNA数量
  int<lower=0> n_samples;                     // 样本数量
  int<lower=0> n_TFs;                         // 转录因子数量
  
  // TF-miRNA边的信息
  int<lower=0> n_mz_zero;                     // TF-miRNA零边数量
  int<lower=0> n_mz_ones;                     // TF-miRNA非零边数量
  int<lower=0> n_mz_negs;                     // TF-miRNA抑制边数量
  int<lower=0> n_mz_poss;                     // TF-miRNA激活边数量
  int<lower=0> n_mz_blur;                     // TF-miRNA模糊边数量
  int<lower=0> n_mz_all;                      // TF-miRNA边总数 (n_mirnas * n_TFs)
  
  // TF-gene边的信息 
  int<lower=0> n_gz_zero;                     // TF-gene零边数量
  int<lower=0> n_gz_ones;                     // TF-gene非零边数量
  int<lower=0> n_gz_negs;                     // TF-gene抑制边数量
  int<lower=0> n_gz_poss;                     // TF-gene激活边数量
  int<lower=0> n_gz_blur;                     // TF-gene模糊边数量
  int<lower=0> n_gz_all;                      // TF-gene边总数 (n_genes * n_TFs)
  
  // miRNA-gene边的信息 (应全部为负调控)
  int<lower=0> n_gm_zero;                     // miRNA-gene零边数量
  int<lower=0> n_gm_ones;                     // miRNA-gene非零边数量
  int<lower=0> n_gm_all;                      // miRNA-gene边总数 (n_genes * n_mirnas)
  
  // 表达数据
  matrix[n_genes, n_samples] X_gene;          // 基因表达矩阵
  matrix[n_mirnas, n_samples] X_mirna;        // miRNA表达矩阵
  
  // 先验连接概率
  vector[n_mz_all] P_mz;                      // TF到miRNA的先验概率
  vector[n_gz_all] P_gz;                      // TF到gene的先验概率
  vector[n_gm_all] P_gm;                      // miRNA到gene的先验概率
  
  // 各类型边的索引
  array[n_mz_zero] int P_mz_zero;             // TF-miRNA零边索引
  array[n_mz_ones] int P_mz_ones;             // TF-miRNA非零边索引
  array[n_mz_negs] int P_mz_negs;             // TF-miRNA抑制边索引
  array[n_mz_poss] int P_mz_poss;             // TF-miRNA激活边索引
  array[n_mz_blur] int P_mz_blur;             // TF-miRNA模糊边索引
  
  array[n_gz_zero] int P_gz_zero;             // TF-gene零边索引
  array[n_gz_ones] int P_gz_ones;             // TF-gene非零边索引
  array[n_gz_negs] int P_gz_negs;             // TF-gene抑制边索引
  array[n_gz_poss] int P_gz_poss;             // TF-gene激活边索引
  array[n_gz_blur] int P_gz_blur;             // TF-gene模糊边索引
  
  array[n_gm_zero] int P_gm_zero;             // miRNA-gene零边索引
  array[n_gm_ones] int P_gm_ones;             // miRNA-gene非零边索引
  
  // 模型配置选项
  int sign;                                   // 使用有符号网络或无符号网络
  int baseline;                               // 是否包含基线表达项
  int psis_loo;                               // 是否计算LOO用于模型检验
  
  // 先验超参数
  real sigmaZ;                                // Z的先验标准差
  real sigmaB;                                // 基线项的先验标准差
  real a_alpha;                               // 逆gamma分布超参数
  real b_alpha;
  real a_sigma;                               // 逆gamma分布超参数 
  real b_sigma;
}

transformed data {
  vector[n_genes * n_samples] X_gene_vec = to_vector(X_gene);
  vector[n_mirnas * n_samples] X_mirna_vec = to_vector(X_mirna);
}

parameters {
  // TF活性矩阵 (非负)
  matrix<lower=0>[n_TFs, n_samples] Z;        // TF活性矩阵
  
  // 噪声方差
  vector<lower=0>[n_genes] sigma2_gene;       // 基因表达噪声方差
  vector<lower=0>[n_mirnas] sigma2_mirna;     // miRNA表达噪声方差
  
  // 基线表达项
  vector[baseline ? n_genes : 0] b0_gene;     // 基因基线表达
  vector[baseline ? n_mirnas : 0] b0_mirna;   // miRNA基线表达
  
  // TF到miRNA的权重参数
  vector<lower=0>[sign ? n_mz_blur : 0] alpha_mz_blur;
  vector<lower=0>[sign ? n_mz_poss : 0] alpha_mz_poss;
  vector<lower=0>[sign ? n_mz_negs : 0] alpha_mz_negs;
  vector[sign ? n_mz_blur : 0] beta_mz_blur;
  vector<upper=0>[sign ? n_mz_negs : 0] beta_mz_negs;
  vector<lower=0>[sign ? n_mz_poss : 0] beta_mz_poss;
  vector<lower=0>[sign ? 0 : n_mz_ones] alpha_mz_ones;
  vector[sign ? 0 : n_mz_ones] beta_mz_ones;
  
  // TF到gene的权重参数
  vector<lower=0>[sign ? n_gz_blur : 0] alpha_gz_blur;
  vector<lower=0>[sign ? n_gz_poss : 0] alpha_gz_poss;
  vector<lower=0>[sign ? n_gz_negs : 0] alpha_gz_negs;
  vector[sign ? n_gz_blur : 0] beta_gz_blur;
  vector<upper=0>[sign ? n_gz_negs : 0] beta_gz_negs;
  vector<lower=0>[sign ? n_gz_poss : 0] beta_gz_poss;
  vector<lower=0>[sign ? 0 : n_gz_ones] alpha_gz_ones;
  vector[sign ? 0 : n_gz_ones] beta_gz_ones;
  
  // miRNA到gene的权重参数 (应全部为负调控)
  vector<lower=0>[n_gm_ones] alpha_gm;        // 噪声精度
  vector<upper=0>[n_gm_ones] beta_gm;         // 负权重 (miRNA抑制基因)
}

transformed parameters {
  // TF到miRNA的网络权重
  vector[sign ? n_mz_negs : 0] W_mz_negs;
  vector[sign ? n_mz_poss : 0] W_mz_poss;
  vector[sign ? n_mz_blur : 0] W_mz_blur;
  vector[sign ? 0 : n_mz_ones] W_mz_ones;
  
  // TF到gene的网络权重
  vector[sign ? n_gz_negs : 0] W_gz_negs;
  vector[sign ? n_gz_poss : 0] W_gz_poss;
  vector[sign ? n_gz_blur : 0] W_gz_blur;
  vector[sign ? 0 : n_gz_ones] W_gz_ones;
  
  // miRNA到gene的网络权重 (负值)
  vector[n_gm_ones] W_gm_ones;
  
  // 计算各种网络权重
  if (sign) {
    // TF-miRNA权重
    W_mz_negs = beta_mz_negs .* sqrt(alpha_mz_negs);
    W_mz_poss = beta_mz_poss .* sqrt(alpha_mz_poss);
    W_mz_blur = beta_mz_blur .* sqrt(alpha_mz_blur);
    
    // TF-gene权重
    W_gz_negs = beta_gz_negs .* sqrt(alpha_gz_negs);
    W_gz_poss = beta_gz_poss .* sqrt(alpha_gz_poss);
    W_gz_blur = beta_gz_blur .* sqrt(alpha_gz_blur);
  } else {
    // 无符号网络
    W_mz_ones = beta_mz_ones .* sqrt(alpha_mz_ones);
    W_gz_ones = beta_gz_ones .* sqrt(alpha_gz_ones);
  }
  
  // miRNA到gene权重 (应全部为负)
  W_gm_ones = beta_gm .* sqrt(alpha_gm);
}

model {
  // 构建权重向量和矩阵
  vector[n_mz_all] W_mz_vec;    // TF到miRNA的权重向量
  vector[n_gz_all] W_gz_vec;    // TF到gene的权重向量
  vector[n_gm_all] W_gm_vec;    // miRNA到gene的权重向量
  
  // 设置TF到miRNA的权重
  W_mz_vec[P_mz_zero] = rep_vector(0, n_mz_zero);
  if (sign) {
    W_mz_vec[P_mz_negs] = W_mz_negs;
    W_mz_vec[P_mz_poss] = W_mz_poss;
    W_mz_vec[P_mz_blur] = W_mz_blur;
  } else {
    W_mz_vec[P_mz_ones] = W_mz_ones;
  }
  
  // 设置TF到gene的权重
  W_gz_vec[P_gz_zero] = rep_vector(0, n_gz_zero);
  if (sign) {
    W_gz_vec[P_gz_negs] = W_gz_negs;
    W_gz_vec[P_gz_poss] = W_gz_poss;
    W_gz_vec[P_gz_blur] = W_gz_blur;
  } else {
    W_gz_vec[P_gz_ones] = W_gz_ones;
  }
  
  // 设置miRNA到gene的权重
  W_gm_vec[P_gm_zero] = rep_vector(0, n_gm_zero);
  W_gm_vec[P_gm_ones] = W_gm_ones;
  
  // 将权重向量转换为矩阵
  matrix[n_mirnas, n_TFs] W_mz = to_matrix(W_mz_vec, n_mirnas, n_TFs);  // 按列
  matrix[n_genes, n_TFs] W_gz = to_matrix(W_gz_vec, n_genes, n_TFs);    // 按列
  matrix[n_genes, n_mirnas] W_gm = to_matrix(W_gm_vec, n_genes, n_mirnas);  // 按列
  
  // 计算miRNA表达均值: X_mirna = W_mz * Z + b0_mirna
  matrix[n_mirnas, n_samples] mu_mirna = W_mz * Z;
  if (baseline) {
    mu_mirna = mu_mirna + rep_matrix(b0_mirna, n_samples);
  }
  
  // 计算基因表达均值: X_gene = W_gz * Z + W_gm * X_mirna + b0_gene
  matrix[n_genes, n_samples] mu_gene = W_gz * Z + W_gm * X_mirna;
  if (baseline) {
    mu_gene = mu_gene + rep_matrix(b0_gene, n_samples);
  }
  
  // 转换为向量形式以便计算概率
  vector[n_mirnas * n_samples] mu_mirna_vec = to_vector(mu_mirna);
  vector[n_genes * n_samples] mu_gene_vec = to_vector(mu_gene);
  vector[n_mirnas * n_samples] sigma_mirna_vec = to_vector(rep_matrix(sqrt(sigma2_mirna), n_samples));
  vector[n_genes * n_samples] sigma_gene_vec = to_vector(rep_matrix(sqrt(sigma2_gene), n_samples));
  
  // 先验分布
  sigma2_mirna ~ inv_gamma(a_sigma, b_sigma);
  sigma2_gene ~ inv_gamma(a_sigma, b_sigma);
  
  if (baseline) {
    b0_mirna ~ normal(0, sigmaB);
    b0_gene ~ normal(0, sigmaB);
  }
  
  // TF-miRNA权重先验
  if (sign) {
    alpha_mz_poss ~ inv_gamma(a_alpha, b_alpha);
    beta_mz_poss ~ normal(0, 1);
    
    alpha_mz_negs ~ inv_gamma(a_alpha, b_alpha);
    beta_mz_negs ~ normal(0, 1);
    
    alpha_mz_blur ~ inv_gamma(a_alpha, b_alpha);
    beta_mz_blur ~ normal(0, 1);
  } else {
    alpha_mz_ones ~ inv_gamma(a_alpha, b_alpha);
    beta_mz_ones ~ normal(0, 1);
  }
  
  // TF-gene权重先验
  if (sign) {
    alpha_gz_poss ~ inv_gamma(a_alpha, b_alpha);
    beta_gz_poss ~ normal(0, 1);
    
    alpha_gz_negs ~ inv_gamma(a_alpha, b_alpha);
    beta_gz_negs ~ normal(0, 1);
    
    alpha_gz_blur ~ inv_gamma(a_alpha, b_alpha);
    beta_gz_blur ~ normal(0, 1);
  } else {
    alpha_gz_ones ~ inv_gamma(a_alpha, b_alpha);
    beta_gz_ones ~ normal(0, 1);
  }
  
  // miRNA-gene权重先验 (全部为负)
  alpha_gm ~ inv_gamma(a_alpha, b_alpha);
  beta_gm ~ normal(0, 1);
  
  // TF活性先验
  to_vector(Z) ~ normal(0, sigmaZ);
  
  // 似然函数
  X_mirna_vec ~ normal(mu_mirna_vec, sigma_mirna_vec);  // miRNA表达似然
  X_gene_vec ~ normal(mu_gene_vec, sigma_gene_vec);     // 基因表达似然
}

generated quantities {
  // 用于模型检验的日志似然
  vector[psis_loo ? n_mirnas * n_samples : 0] log_lik_mirna;
  vector[psis_loo ? n_genes * n_samples : 0] log_lik_gene;
  
  if (psis_loo) {
    // 重新计算均值和标准差
    vector[n_mz_all] W_mz_vec;    // TF到miRNA的权重向量
    vector[n_gz_all] W_gz_vec;    // TF到gene的权重向量
    vector[n_gm_all] W_gm_vec;    // miRNA到gene的权重向量
    
    // 设置TF到miRNA的权重
    W_mz_vec[P_mz_zero] = rep_vector(0, n_mz_zero);
    if (sign) {
      W_mz_vec[P_mz_negs] = W_mz_negs;
      W_mz_vec[P_mz_poss] = W_mz_poss;
      W_mz_vec[P_mz_blur] = W_mz_blur;
    } else {
      W_mz_vec[P_mz_ones] = W_mz_ones;
    }
    
    // 设置TF到gene的权重
    W_gz_vec[P_gz_zero] = rep_vector(0, n_gz_zero);
    if (sign) {
      W_gz_vec[P_gz_negs] = W_gz_negs;
      W_gz_vec[P_gz_poss] = W_gz_poss;
      W_gz_vec[P_gz_blur] = W_gz_blur;
    } else {
      W_gz_vec[P_gz_ones] = W_gz_ones;
    }
    
    // 设置miRNA到gene的权重
    W_gm_vec[P_gm_zero] = rep_vector(0, n_gm_zero);
    W_gm_vec[P_gm_ones] = W_gm_ones;
    
    // 将权重向量转换为矩阵
    matrix[n_mirnas, n_TFs] W_mz = to_matrix(W_mz_vec, n_mirnas, n_TFs);
    matrix[n_genes, n_TFs] W_gz = to_matrix(W_gz_vec, n_genes, n_TFs);
    matrix[n_genes, n_mirnas] W_gm = to_matrix(W_gm_vec, n_genes, n_mirnas);
    
    // 计算miRNA表达均值
    matrix[n_mirnas, n_samples] mu_mirna = W_mz * Z;
    if (baseline) {
      mu_mirna = mu_mirna + rep_matrix(b0_mirna, n_samples);
    }
    
    // 计算基因表达均值
    matrix[n_genes, n_samples] mu_gene = W_gz * Z + W_gm * X_mirna;
    if (baseline) {
      mu_gene = mu_gene + rep_matrix(b0_gene, n_samples);
    }
    
    // 转换为向量
    vector[n_mirnas * n_samples] mu_mirna_vec = to_vector(mu_mirna);
    vector[n_genes * n_samples] mu_gene_vec = to_vector(mu_gene);
    vector[n_mirnas * n_samples] sigma_mirna_vec = to_vector(rep_matrix(sqrt(sigma2_mirna), n_samples));
    vector[n_genes * n_samples] sigma_gene_vec = to_vector(rep_matrix(sqrt(sigma2_gene), n_samples));
    
    // 计算每个元素的日志似然
    for (i in 1:(n_mirnas * n_samples)) {
      log_lik_mirna[i] = normal_lpdf(X_mirna_vec[i] | mu_mirna_vec[i], sigma_mirna_vec[i]);
    }
    
    for (i in 1:(n_genes * n_samples)) {
      log_lik_gene[i] = normal_lpdf(X_gene_vec[i] | mu_gene_vec[i], sigma_gene_vec[i]);
    }
  }
}