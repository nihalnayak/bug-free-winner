
library(Seurat)
library(ggplot2)
library(patchwork)

dataset_path <- "data/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_"
par <- list(
  input_train_mod1 = paste0(dataset_path, "train_mod1.h5ad"),
  input_train_mod2 = paste0(dataset_path, "train_mod2.h5ad"),
  input_train_sol = paste0(dataset_path, "train_sol.h5ad"),
  input_test_mod1 = paste0(dataset_path, "test_mod1.h5ad"),
  input_test_mod2 = paste0(dataset_path, "test_mod2.h5ad"),
  input_test_sol = paste0(dataset_path, "test_sol.h5ad"),
  output = "output.h5ad",
  n_neighbors = 5L
)

input_test_mod1 <- anndata::read_h5ad(par$input_test_mod1)
input_test_mod2 <- anndata::read_h5ad(par$input_test_mod2)
input_test_sol <- anndata::read_h5ad(par$input_test_sol)


rna<-t(read.csv("GEX.csv",header = FALSE))
colnames(rna) <- paste0(colnames(rna), 1:ncol(rna)) 
rna<-CreateSeuratObject(counts=rna)
#rna<-CreateSeuratObject(counts=input_test_mod1$layers["counts"])
rna<-NormalizeData(rna)
rna<-FindVariableFeatures(rna)
rna<-ScaleData(rna)

rna<- RunPCA(rna,npcs=25,verbose = FALSE)

DimHeatmap(rna, dims = 1:15, cells = 500, balanced = TRUE)
ElbowPlot(rna, ndims = 50)

rna<- FindNeighbors(rna, dims = 1:15)
rna<- FindClusters(rna, resolution = 0.8)

rna<- RunTSNE(rna, dims = 1:15, method = "FIt-SNE",check_duplicates = FALSE)
rna.markers <- FindAllMarkers(rna, max.cells.per.ident = 100, min.diff.pct = 0.3, only.pos = TRUE)
new.cluster.ids <- c("A", "B", "C", "D", "E", "F", "G", 
                     "H", "I", "J", "K", "L", "M")
names(new.cluster.ids) <- levels(rna)
rna <- RenameIdents(rna, new.cluster.ids)
DimPlot(rna, label = TRUE) + NoLegend()


print(dim(rna))

rna[["ADT"]] <- CreateAssayObject(counts = cbmc.adt)

adt<-CreateAssayObject(counts=input_test_mod2$layers["counts"],assay="ADT")
adt<- NormalizeData(adt, assay = "ADT", normalization.method = "CLR")
#variable_adt<- FindVariableFeatures(norm_adt)
adt<- ScaleData(adt, assay = "ADT")
#pca_adt<- RunPCA(scale_adt, verbose = FALSE)
#tse_adt<-RunTSNE(scale_adt, dims = 1:25, method = "FIt-SNE",check_duplicates = FALSE)
FeaturePlot(adt, features = c("CD86", "CD274"), reduction='umap', min.cutoff = "q05", max.cutoff = "q95", ncol =2)

RidgePlot(adt, features = c("CD86", "CD274"), ncol = 2)



