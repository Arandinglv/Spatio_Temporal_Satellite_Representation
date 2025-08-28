# TODO: 去掉for loop, 提高效率
import torch 
import os
import shutil
import yaml



def accuracy(output, target, topk=None):
    """
    output: batch_size, 取决于输出
    target: batch_size, 1
    如果topk = (1, 5), 则返回top1, top5的准确率
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True) # pred为index -> batchsize*2, maxk
        pred = pred.t()  # transpose  -> maxk, batch-size*2
        correct = pred.eq(target.view(1, -1).expand_as(pred))  
        # target: [batch_size, 1] -> [1, batch_size]
        # correct: [maxk, batch_size*2] bool
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # [k, batch_size*2] -> [k*batch_size*2] -> scalar -> [1]
            res.append(correct_k.mul_(100.0 / batch_size))  # mul_原地操作, 逐乘, 转为百分比
        return res

    

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
        
def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)
            
    


# def spatial_accuracy(output, target, topk=(1,)):
#     """
#     spatial accuracy
#     output shape: batch-size * n_views, batch-size * n_views - 1  (logits)
#     target shape: batch-size * n_views   1D 行向量  (labels)
#     """
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
        
#         _, idx = output.topk(maxk, 1, True, True)  # idx≈label, 每个图片都是一个类别
#         # dim = 1, pred为index -> batchsize*2, maxk
#         idx = idx.t()  # transpose  -> maxk, batch-size*2
#         correct = (idx.eq(target.view(1, -1).expand_as(idx))).float()
#         # bool - > float  maxk, batch-size*2
        
#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).sum(0, keepdim=True)  # 在maxk中取前k个
#             res.append(correct_k.mul_(100.0 / batch_size))  

#     return  res 



# def temporal_accuracy_neg(output, target, topk=(1,), threshold=0.5):
#     """
#     适用于只计算负样本相似度的时间对比学习损失
#     topk选取前k个相似度最低的样本
#     output: batch-size, neg * years  (temporal_similarity)
#     target: batch-size, neg * years  (temporal_labels_neg)
#     """
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = output.size(0)

#         _, idx = output.topk(maxk, 1, False, False)  # batch-size, maxk
#         # 选取最小的k个index, 升序排列

#         res = []
#         for i, k in enumerate(topk):
#             topk_pred = idx[:, :k]  # batch_size, k

#             # 得到对应相似度
#             correct_k = output.gather(dim=1, index=topk_pred)
#             threshold_tensor = torch.full_like(correct_k, threshold)
#             correct_k = (correct_k < threshold_tensor).float()

#             res.append(correct_k.sum() * 100.0 / batch_size / k)

#     return res



# def temporal_accuracy_pos_neg(output, target, topk=(1, 5), threshold=0.5):
#     """
#     适用于计算在同时有正样本和负样本时的时间对比学习输出的准确率。
#     参数:
#         output: [batch_size, 1 + num_neg] 的 logits，logits[:,0] 为正样本分数
#         target: [batch_size] 的整型标签，全为0
#         topk:   需要计算的top-k列表, 如(1,5)
#     返回:
#         一个list, 依次为top1, top5, ... 的准确率
#     """
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)  # = output.size(0)

#         # output: [batch_size, 1 + num_neg]
#         # 取分数最高的 maxk 个index
#         # largest=True 表示从大到小排列
#         _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # pred: [batch_size, maxk]

#         # pred 的形状: [batch_size, maxk]
#         # target 全是0，表示正样本的下标=0
#         # 比较 pred[:,:k] 是否包含 0
#         res = []
#         for k in topk:
#             correct_k = (pred[:, :k] == 0).float().sum()
#             # correct_k 表示在前k个里预测到“0”类别的数量
#             # 如果 pred[i, :k] 中包含0，则表示该样本top-k正确
#             res.append(correct_k * 100.0 / batch_size)  # 转为百分比
#         return res
