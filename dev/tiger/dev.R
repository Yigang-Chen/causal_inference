prior.pp = function(prior,expr){
  
  # filter tfs and tgs
  tf = intersect(rownames(prior),rownames(expr)) ## TF needs to express
  tg = intersect(colnames(prior),rownames(expr))
  all.gene = unique(c(tf,tg))
  
  # create coexp net
  coexp = GeneNet::ggm.estimate.pcor(t(expr[all.gene,]), method = "static")
  diag(coexp)= 0
  
  # prior and coexp nets
  P_ij = prior[tf,tg] ## prior ij
  C_ij = coexp[tf,tg]*abs(P_ij) ## coexpression ij
  
  # signs
  sign_P = sign(P_ij) ## signs in prior
  sign_C = sign(C_ij) ## signs in coexp
  
  # blurred edge index
  blurs = which((sign_P*sign_C)<0,arr.ind = T) ## inconsistent edges
  P_ij[blurs] = 1e-6
  
  # remove all zero TFs (in case prior has all zero TFs)
  A_ij = P_ij
  A_ij = A_ij[rowSums(A_ij!=0)>0,]
  A_ij = A_ij[,colSums(A_ij!=0)>0]
  
  return(A_ij)
}

# 读取基因表达矩阵
expr <- read.csv("data/holland_rna_expr.tsv", row.names=1, sep="\t")  # 假设第一列是基因名

# 读取先验网络
prior <- read.csv("data/tf_gene_matrix.csv", row.names=1)  # 假设第一列是TF名

sample.name = colnames(expr)

TF.name = sort(intersect(rownames(prior),rownames(expr))) # TF needs to express

TG.name = sort(intersect(rownames(expr),colnames(prior)))
if (length(TG.name)==0 | length(TF.name)==0){
  stop("No matched gene names in the two inputs...")
}

print(paste("Prior dimensions:", nrow(prior), "x", ncol(prior), 
            ", row names:", length(rownames(prior))))
print(paste("Expr dimensions:", nrow(expr), "x", ncol(expr), 
            ", row names:", length(rownames(expr))))
tf_set = intersect(rownames(prior), rownames(expr))
print(paste("TF intersection size:", length(tf_set)))
print(length(TG.name))
print(length(TF.name))

signed = TRUE

#0. prepare stan input
if (signed){
  if (length(intersect(TG.name,TF.name))!=0){
    prior2 = prior.pp(prior[TF.name,TG.name],expr)
    write.csv(prior2,"compare/prior_pp_r.csv")
    if (nrow(prior2)!=length(TF.name)){
      TFnotExp = setdiff(TF.name,rownames(prior2))
      TFnotExpEdge = prior[TFnotExp,colnames(prior2),drop=F]
      TFnotExpEdge[TFnotExpEdge==1] = 1e-6
      TFnotExpEdge[TFnotExpEdge==-1] = 1e-6
      prior2 = rbind(prior2,TFnotExpEdge)
      prior2 = prior2[order(rownames(prior2)),]
      prior2 = prior2[rowSums(prior2!=0)>0,]  # remove all zero TFs
    }
    P = prior2
    write.csv(P,"compare/prior_r.csv")
    TF.name = rownames(P)
    TG.name = colnames(P)
  }else{
    P = prior[TF.name,TG.name]
  }
}else{
  P = prior[TF.name,TG.name]
}
X = expr[TG.name,]
n_genes = dim(X)[1]
n_samples = dim(X)[2]
n_TFs = dim(P)[1]
