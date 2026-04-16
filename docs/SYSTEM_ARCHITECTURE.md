# StreetClean AI: System Architecture

## Problem Statement
Traditional object detection pipelines map bounding boxes equally without respecting environmental topology. This system extends raw detection into a context-aware and spatially-intelligent tracking application.

## 1. Object Detection Engine
Our foundation utilises YOLOv8 fine-tuned precisely on an optimized subset of the TACO dataset. Categories are collapsed into 6 highly impactful severity vectors: `plastic`, `metal`, `paper`, `organic`, `glass`, and `other`. 

## 2. Scene-Dependent Contextual Normalisation
We hypothesise that absolute object counts must be divided by an expected normal. Using MobileNetV2, an image $I$ is mapped to $S$, a scene label from $\{\text{road}, \text{park}, \text{street}, \text{indoor}\}$. The normalisation is:
$$ C_{context} = 5.0 - \left( \frac{N}{ \beta_S } \right) $$
Where $\beta_S$ is the scene baseline (e.g., 15 for streets, 8 for parks).

## 3. Weighted Semantic Impact
Not all detritus degrades environments equally. We allocate an ecological harm penalty $w_c$:
$$ S_{semantic} = 5.0 - \left( 100 \sum_{i=0}^{N} w_{ci} \times \frac{A_i}{A_{img}} \times conf_i \right) $$
Plastic ($w=1.0$) inherently drops the score more severely than Paper ($w=0.7$).

## 4. Spatial Point-Spread Heatmap
Using a Gaussian kernel mapped over detection centroids $(x_c, y_c)$, the system generates a thermal matrix:
$$ H(x,y) = \sum_{i=0}^{N} \exp \left( - \frac{(x-x_{ci})^2 + (y-y_{ci})^2}{2\sigma^2} \right) $$
Clusters exceeding threshold $\tau$ generate Hotspot warnings, natively isolating critical cleanup grids regardless of overall image capacity.

---
*End of Blueprint Document*
