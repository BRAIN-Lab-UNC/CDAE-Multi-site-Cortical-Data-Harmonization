# CDAE: Multi-site Cortical Data Harmonization

Official PyTorch implementation of **Cycle-Consistent Disentangled Autoencoder (CDAE)** for multi-site cortical data harmonization, introduced in:
 *Fenqiang Zhao, Zhengwang Wu, Dajiang Zhu, Tianming Liu, John Gilmore, Weili Lin, Li Wang, Gang Li. "Disentangling Site Effects with Cycle-Consistent Adversarial Autoencoder for Multi-site Cortical Data Harmonization," MICCAI 2023.* [Paper link](https://link.springer.com/chapter/10.1007/978-3-031-43993-3_36)

![Demo](https://github.com/BRAIN-Lab-UNC/CDAE-Multi-site-Cortical-Data-Harmonization/imgs/demo.png)

------

## 🌟 Highlights

- **Vertex-wise harmonization** of cortical surface maps.
- **Disentangled representation learning** to separate site-related vs. site-unrelated features.
- **Cycle consistency constraints** for controllable and meaningful mapping.
- **Large-scale validation** on **2,342 infant cortical scans from 4 sites**, achieving state-of-the-art performance in removing site effects while preserving biological variability.

## 🔍 Inference

To harmonize cortical maps from a source site (e.g., S2) to a target site (e.g., S1):

```python
python ./scripts/harmonize.py \
  --input subjX_thickness.npy \
  --source_site S2 \
  --target_site S1 \
  --checkpoint ./trained_models/cdae_model.pth \
  --output subjX_harmonized.npy
```

## 🚀 Training (Coming soon)

## 📖 Citation

If you find the paper or repository useful, please consider citing:

```yaml
@inproceedings{zhao2019harmonization,
  title={Harmonization of infant cortical thickness using surface-to-surface cycle-consistent adversarial networks},
  author={Zhao, Fenqiang and Wu, Zhengwang and Wang, Li and Lin, Weili and Xia, Shunren and Shen, Dinggang and Li, Gang and UNC/UMN Baby Connectome Project Consortium},
  booktitle={International conference on medical image computing and computer-assisted intervention},
  pages={475--483},
  year={2019},
  organization={Springer}
}
@inproceedings{zhao2023disentangling,
  title={Disentangling site effects with cycle-consistent adversarial autoencoder for multi-site cortical data harmonization},
  author={Zhao, Fenqiang and Wu, Zhengwang and Zhu, Dajiang and Liu, Tianming and Gilmore, John and Lin, Weili and Wang, Li and Li, Gang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={369--379},
  year={2023},
  organization={Springer}
}
```

