# README

本仓库是对三个图异常检测框架DOMINANT, CONAD, ADA-GAD基于Pytorch+PyG的重写版本。

# Environmental installation

```shell
conda env create -f environment.yml
```

# Usage

假设要运行DOMINANT或CONAD的代码（以CONAD举例）：

```shell
cd conad/
python main.py --dataset inj_cora
```

假设要运行ADA-GAD的代码：

```shell
cd ada-gad/
python main.py --use_cfg --dataset inj_cora
```

# Reference

[1] Ding K, Li J, Bhanushali R, et al. Deep anomaly detection on attributed networks[C]//Proceedings of the 2019 SIAM international conference on data mining. Society for Industrial and Applied Mathematics, 2019: 594-602.

[2] Xu Z, Huang X, Zhao Y, et al. Contrastive attributed network anomaly detection with data augmentation[C]//Pacific-Asia conference on knowledge discovery and data mining. Cham: Springer International Publishing, 2022: 444-457.

[3] He J, Xu Q, Jiang Y, et al. ADA-GAD: Anomaly-Denoised Autoencoders for Graph Anomaly Detection[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(8): 8481-8489.

‍
