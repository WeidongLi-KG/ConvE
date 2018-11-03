import torch
import numpy as np
import argparse
import os
import time
from datetime import datetime
from config import config
from logger_init import get_logger
from read_data import index_ent_rel, graph_size, read_data, read_data_with_rel_reverse, read_reverse_data
from data_utils import inplace_shuffle, heads_tails, batch_by_num, batch_by_size
from evaluation import ranking_and_hits
from model import ConvE, DistMult, Complex

np.set_printoptions(precision=3)
logger = get_logger('train', True)
logger.info('START TIME : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
Config = config()
#model_name = 'DistMult_{0}_{1}'.format(Config.input_dropout, Config.dropout)
model_name = '{2}_{0}_{1}'.format(Config.input_dropout, Config.dropout, Config.model_name)
epochs = 1000
load = False
if Config.dataset is None:
    Config.dataset = 'FB15k-237'
save_dir = 'saved_models'
if not os.path.exists(save_dir): os.makedirs(save_dir)
model_path = 'saved_models/{0}_{1}.model'.format(Config.dataset, model_name)

task_dir = config().task_dir
kb_index = index_ent_rel(os.path.join(task_dir, 'train.txt'),
                         os.path.join(task_dir, 'valid.txt'),
                         os.path.join(task_dir, 'test.txt'))
n_ent, n_rel = graph_size(kb_index)

train_data_with_reverse = read_data_with_rel_reverse(os.path.join(task_dir, 'train.txt'), kb_index)
inplace_shuffle(*train_data_with_reverse)
heads, tails = heads_tails(n_ent, train_data_with_reverse)

train_data = read_data(os.path.join(task_dir, 'train.txt'), kb_index)
valid_data = read_data(os.path.join(task_dir, 'valid.txt'), kb_index)
test_data = read_data(os.path.join(task_dir, 'test.txt'), kb_index)
eval_h, eval_t = heads_tails(n_ent, train_data, valid_data, test_data)

valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data = [torch.LongTensor(vec) for vec in test_data]
train_data_with_reverse = [torch.LongTensor(vec) for vec in train_data_with_reverse]

parser = argparse.ArgumentParser(description='Conv2E argparse')
parser.add_argument('--model_name', type=str, default='ConvE', help='specific the model name')

args = parser.parse_args()

def main():
    if args.model_name is None:
        model = ConvE(n_ent, n_rel)
    elif args.model_name == 'ConvE':
        model = ConvE(n_ent, n_rel)
    elif args.model_name == 'DistMult':
        model = DistMult(n_ent, n_rel)
    elif args.model_name == 'ComplEx':
        model = Complex(n_ent, n_rel)
    else:
        logger.info('Unknown model: {0}', args.model_name)
        raise Exception("Unknown model!")

    if Config.cuda:
        model.cuda()

    model.init()
    params = [value.numel() for value in model.parameters()]
    print(params)
    print(sum(params))
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        epoch_loss = 0
        start = time.time()
        model.train()
        h, r, t = train_data_with_reverse
        n_train = h.size(0)
        rand_idx = torch.randperm(n_train)
        h = h[rand_idx].cuda()
        r = r[rand_idx].cuda()
        tot = 0.0

        for bh, br in batch_by_num(Config.n_batch, h, r):
            opt.zero_grad()
            batch_size = bh.size(0)
            e2_multi = torch.empty(batch_size, n_ent)
            # label smoothing
            for i, (head, rel) in enumerate(zip(bh, br)):
                head = head.item()
                rel = rel.item()
                e2_multi[i] = tails[head, rel].to_dense()
            e2_multi = ((1.0-Config.label_smoothing_epsilon)*e2_multi) + (1.0/e2_multi.shape[1])
            e2_multi = e2_multi.cuda()
            pred = model.forward(bh, br)
            loss = model.loss(pred, e2_multi)
            loss.backward()
            opt.step()
            batch_loss = torch.sum(loss)
            epoch_loss += batch_loss
            tot += bh.size(0)
            print('\r{:>10} progress {} loss: {}'.format('', tot/n_train, batch_loss), end='')
        logger.info('')
        end = time.time()
        time_used = end - start
        logger.info('one epoch time: {} minutes'.format(time_used/60))
        logger.info('epoch {} loss: {}'.format(epoch+1, epoch_loss))
        logger.info('saving to {0}'.format(model_path))
        torch.save(model.state_dict(), model_path)

        model.eval()
        with torch.no_grad():
            start = time.time()
            ranking_and_hits(model, Config.batch_size, valid_data, eval_h, eval_t,'dev_evaluation')
            end = time.time()
            logger.info('eval time used: {} minutes'.format((end - start)/60))
            if epoch % 3 == 0:
                if epoch > 0:
                    ranking_and_hits(model, Config.batch_size, test_data, eval_h, eval_t, 'test_evaluation')


if __name__ == '__main__':
    main()
