# F2GNN

Code for ICASSP 2024 paper: "F2GNN: AN ADAPTIVE FILTER WITH FEATURE SEGMENTATION FOR GRAPH-BASED FRAUD DETECTION".

Guanghui Hu, Yang Liu, Qing He, Xiang Ao

## Dependencies

The code has been successfully tested in the following environment:

- python==3.9.0
- pytorch==1.12.1
- dgl_cuda11.3==0.9.1
- numpy==1.23.5
- scikit_learn==1.2.2
- scipy==1.9.3
- pyyaml==6.0

# Usage

For the Amazon dataset:

```bash
python run.py --dataset amazon
```

For the YelpChi dataset:

```bash
python run.py --dataset yelpchi
```
