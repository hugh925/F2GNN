# F2GNN

Code for ICASSP 2024 paper: "F2GNN: An Adaptive Filter with Feature Segmentation for Graph-Based Fraud Detection".

Guanghui Hu, [Yang Liu](https://yliu.site/) , [Qing He](https://people.ucas.ac.cn/~heqing?language=en) , [Xiang Ao](https://aoxaustin.github.io/)

## Dependencies

The code has been successfully tested in the following environment:

- python==3.9.0
- dgl_cuda11.3==0.9.1
- pytorch==1.12.1
- numpy==1.23.5
- scikit_learn==1.2.2
- scipy==1.9.3
- pyyaml==6.0

## Usage

For the Amazon dataset:

```bash
python run.py --dataset amazon
```

For the YelpChi dataset:

```bash
python run.py --dataset yelpchi
```

## Cite

```
@inproceedings{hu2024f2gnn,
  title={F2GNN: An Adaptive Filter with Feature Segmentation for Graph-Based Fraud Detection},
  author={Hu, Guanghui and Liu, Yang and He, Qing and Ao, Xiang},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6335--6339},
  year={2024},
  organization={IEEE}
}
```
