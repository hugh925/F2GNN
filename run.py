import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import time
import argparse
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from sklearn.model_selection import train_test_split
from F2GNN import *
from utils import *
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, confusion_matrix
# import dgl
# import torch
# import numpy

parser = argparse.ArgumentParser(description='F2GNN')
parser.add_argument('--dataset', type=str, default='amazon', help='Dataset for this model (amazon/yelpchi)')
parser.add_argument('--train_ratio', type=float, help='Training ratio')
parser.add_argument('--lr', type=float, help='Initial learning rate.')
parser.add_argument('--layer_num', type=int, help='Number of layers')
parser.add_argument('--hid_dim', type=int, help='Hidden layer dimension')
parser.add_argument('--head_num', type=int, help='Number of attention heads')
parser.add_argument('--dropMessage', type=float, help='DropMessage rate (1 - keep probability).')
parser.add_argument('--eps', type=float, help='Fixed scalar or learnable weight.')
parser.add_argument('--weight_decay', type=float, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--homo', type=int, help='1 for (Homo) and 0 for (Hetero)')
parser.add_argument('--epochs', type=int, help='Number of epochs to train.')
parser.add_argument('--cuda', type=int, help='which gpu device to use.')
parser.add_argument('--run', type=int, help='Number of Training Runs.')
parser.add_argument('--seed', '--rand-seed', type=int, help='The random seed (default: 925).')
parser.add_argument('--add', '--addition', type=str, help='additional information and usage')
args = parser.parse_args()
dataset_config = load_config(args.dataset)
update_args_from_config(args, dataset_config)
print(args)


if args.dataset == 'amazon':
    dataset = FraudAmazonDataset()
    graph = dataset[0]
    if args.homo == 1:
        graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph = dgl.add_self_loop(graph)
elif args.dataset == 'yelpchi':
    dataset = FraudYelpDataset()
    graph = dataset[0]
    # if args.homo == 1:
    #     graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
    #     graph = dgl.add_self_loop(graph)
else:
    print('no such dataset')
    exit(1)


# device selection
if torch.cuda.is_available() and args.cuda < torch.cuda.device_count():
    device = torch.device('cuda:{}'.format(args.cuda))
else:
    device = torch.device('cpu')


set_random_seed(args.seed)
graph.ndata['feature'] = graph.ndata['feature'].float()
features = graph.ndata['feature']
features = normalize_features(features).to(device)
in_feats = features.shape[1]
graph = graph.to(device)
graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
labels = graph.ndata['label']
num_classes = 2


def train_run():
    index = list(range(len(labels)))
    if args.dataset == 'amazon':
        index = list(range(3305, len(labels)))
    labels_cpu = labels.cpu()
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels_cpu[index], stratify=labels_cpu[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_epoch = 1
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc, final_gmean = 0., 0., 0., 0., 0., 0.,0.
    weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    wt=torch.tensor([1., weight]).to(device)
    print('cross entropy weight: ', weight)

    time_start = time.time()
    for e in range(1,args.epochs+1):
        model.train()
        logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=wt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = model(features).softmax(1)
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        preds = torch.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        trec = recall_score(labels[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy(), zero_division=1)
        tpre = precision_score(labels[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy(), zero_division=0)
        tmf1 = f1_score(labels[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy(), average='macro')
        tauc = roc_auc_score(labels[test_mask].cpu().numpy(), probs[test_mask][:, 1].detach().cpu().numpy())
        preds[probs[:, 1] > 0.5] = 1
        tn, fp, fn, tp = confusion_matrix(labels[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy()).ravel()
        tgmean = (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5

        if best_f1 < f1:
            best_f1 = f1
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            final_gmean = tgmean
            best_epoch = e
            print('---------------------------------------------------------------------------------------------------')
        print('Epoch {}, loss: {:.4f}, Val mf1: {:.4f}, (best {:.4f}), Test f1:{:.4f}, auc:{:.4f}, gmean:{:.4f}'.format(
                                                       e, loss, f1, best_f1,tmf1,tauc,tgmean))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('best_epoch:', best_epoch)
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} GMean {:.2f}'.format(final_trec * 100,final_tpre * 100,
                                                                final_tmf1 * 100, final_tauc * 100,final_gmean* 100))
    result = 'MF1 {:.2f} AUC {:.2f} GMean {:.2f}'.format(final_tmf1 * 100, final_tauc * 100, final_gmean* 100)
    with open('result.txt', 'a+') as f:
        f.write(f'{args}\n'
                +'{}\t'.format(result)
                +'best_epoch:{}\n'.format(best_epoch))
    return final_tmf1, final_tauc, final_gmean, best_epoch

if args.run == 1:
    if args.homo == 1:
        model = F2GNN(graph, in_feats, args.hid_dim, num_classes, args.dropMessage, args.eps, args.head_num, args.layer_num).to(device)
    if args.homo == 0:
        model = F2GNNhetero(graph, in_feats, args.hid_dim, num_classes, args.dropMessage, args.eps,args.head_num, args.layer_num).to(device)
    train_run()
else:
    final_mf1s, final_aucs, final_gmeans, best_epochs= [], [], [], []
    for i in range(args.run):
        if args.homo:
            model = F2GNN(graph, in_feats, args.hid_dim, num_classes, args.dropMessage, args.eps,args.head_num, args.layer_num).to(device)
        if args.homo == 0:
            model = F2GNNhetero(graph, in_feats, args.hid_dim, num_classes, args.dropMessage, args.eps,args.head_num, args.layer_num).to(
                device)
        mf1, auc, gmean, best_epoch = train_run()
        final_mf1s.append(mf1)
        final_aucs.append(auc)
        final_gmeans.append(gmean)
        best_epochs.append(best_epoch)
    final_mf1s = np.array(final_mf1s)
    final_aucs = np.array(final_aucs)
    final_gmeans = np.array(final_gmeans)
    best_epochs = np.array(best_epochs)
    print('mf1s:', 100 * (final_mf1s))
    print('aucs:', 100 * (final_aucs))
    print('gmeans:', 100 * (final_gmeans))
    print('best_epoch:',best_epochs)
    result = 'MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}, GMean-mean: {:.2f}, GMean-mean: {:.2f}'.format(
                                                                              100 * np.mean(final_mf1s),
                                                                              100 * np.std(final_mf1s),
                                                                              100 * np.mean(final_aucs),
                                                                              100 * np.std(final_aucs),
                                                                              100 * np.mean(final_gmeans),
                                                                              100 * np.std(final_gmeans))
    print(result)
    with open('result.txt', 'a+') as f:
        f.write(f'{args}\n{final_mf1s}\n{final_aucs}\n{final_gmeans}\n{best_epochs}\t{result}')

if torch.cuda.is_available():
    torch.cuda.empty_cache()