# Exploiting Structural Repetition for Synthetic Training Data Generation in LiDAR Point Cloud Segmentation

Implementation of the paper **Exploiting Structural Repetition for Synthetic Training Data Generation in LiDAR Point Cloud Segmentation**, published in Automation in Construction, Volume 192, 2026.


DOI: [10.1016/j.autcon.2026.107248](https://doi.org/10.1016/j.autcon.2026.107248)

## Authors

**Michael Brunklaus**<sup>1,2</sup>, **Alexander Reiterer**<sup>1,2</sup>  
<sup>1</sup> University of Freiburg, Department of Sustainable Systems Engineering (INATECH), Freiburg, Germany  
<sup>2</sup> Fraunhofer Institute for Physical Measurement Techniques IPM, Freiburg, Germany  


## Abstract

Maintaining aging tunnel infrastructure requires inspection supported by digital tools such as Building Information Models (BIMs). Scan-to-BIM methods based on deep learning automate BIM creation from Light Detection and Ranging (LiDAR) point clouds. However, supervised training requires large datasets that are costly to acquire. This paper presents an automatic synthetic data generation pipeline for subway tunnels. Tunnel segment meshes are assembled along stochastic 3D splines to generate 66.1 km of tunnel meshes, scanned by a LiDAR simulator to yield annotated point clouds. Data realism is assessed, and training strategies are benchmarked using three deep learning architectures evaluated on real-world tunnel scans. In certain cases, synthetic-only training outperforms training on small real datasets. Mixing synthetic and real data outperforms synthetic-only training, achieving 86.2% mIoU on real scans. The results demonstrate that synthetic data generation can reduce dataset creation costs, addressing a key barrier to the adoption of deep learning in construction.

## Repository content

* For details regarding synthetic data generation, please refer to the `part1_data_generation/` directory.
* For details regarding synthetic data evaluation, please refer to the `part2_data_evaluation/` directory.
