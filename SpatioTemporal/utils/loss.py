import torch 
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss:
    
    def __init__(self, args):
        self.args = args
        self.criterion_ce = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.criterion_bce = torch.nn.BCEWithLogitsLoss().to(self.args.device)
    
    def group_spatial_temporal_loss(self, anchors, temporal_negatives, indices):
        """
        Args: 
            anchors: [group_num * group_element, dim=128]
            temporal_negatives: [group_num * group_element, num_neg, dim=128]
            indices: [batch_size, 1], 也就是temporal_negatives中距离最远的索引
            loss = (1 - sim((A - A'), (B - B')) * sim(A, B))
        """
        device = self.args.device
        group_num = self.args.group_num
        group_element = self.args.group_element
        batch_size = group_num * group_element
        
        anchors = F.normalize(anchors, dim=1)  # [batchsize, 128]
        temporal_negatives = F.normalize(temporal_negatives, dim=2)
        
        group_ids = torch.arange(group_num, device=device).repeat_interleave(group_element).view(-1, 1)  
        # [batchsize,] -> [batchsize, 1]
        group_mask = (group_ids == group_ids.t())  # 不用device [batchsize, batchsize]
        similarity = torch.matmul(anchors, anchors.t())  # [batchsize, batchsize]
        # 把对角线设置成无穷小
        self_mask = torch.eye(batch_size, device=device) * 1000
        # 把不是一个组的sim设成无穷小
        masked_sim = similarity.masked_fill(~group_mask, 0) - self_mask
        min_sim_indices = torch.argmax(masked_sim ,dim=1).squeeze()   # batchsize, 1 -> batchsize
        # min_sim_indices: 记录的是每一个anchor距离最远的group内的B所在的行位置(相当于行位置)
        # indices记录的是每一行中temporal negative的位置, 相当于列位置
        batch_indices = torch.arange(batch_size, device=device)
        
        indices = indices.squeeze(1)
        # 取temporal negatives中距离最远的, 也就是A'  三维, 要多索引
        A_prime = temporal_negatives[batch_indices, indices]  # [batchsize, 128]
        B = anchors[min_sim_indices]  # [batchsize, 128]
        B_prime = temporal_negatives[min_sim_indices, indices[min_sim_indices]]  # [batchsize, 8]
        # 锁定B的位置后找到相应的负样本
        
        diff_A = F.normalize((anchors - A_prime), dim=1)   # [batch_size, 128]
        diff_B = F.normalize((B - B_prime), dim=1)   # [batch_size, 128]
        loss = torch.mean((1 - torch.sum(diff_A * diff_B, dim=1)) * torch.sum(anchors * B, dim=1))
        
        return loss


    def temporal_loss(self, features, temporal_negatives):
        """
        JIABO LIU
        计算时间上的对比损失。
        Args:
            features (Tensor): [batch_size, n_views, feature_dim=128]
            temporal_negatives (Tensor): [batch_size, num_neg, feature_dim=128]

        Returns:
            Tensor: 时间损失
        """
        temperature = self.args.temperature
        device = self.args.device
        
        features = F.normalize(features, dim=2)  # [batch_size, n_views, feature_dim=128]
        temporal_negatives = F.normalize(temporal_negatives, dim=2)  # [batch_size, num_neg, feature_dim=128]

        # [batch_size, n_views, feature_dim=128]的摆放方式可以让后面交叉计算
        features_pos = features[:,1:]        # [batchsize, n_views-1, feature_dim]
        features_anchor = features[:,:1]        # [batchsize, 1, feature_dim]

        features = torch.cat([features_pos, temporal_negatives], dim=1)     # [batchsize, 1+num_neg, feature_dim]

        similarity_matrix = torch.matmul(features_anchor, features.transpose(1, 2)) # [batchsize, 1, 1+num_neg]
        similarity_matrix = similarity_matrix.squeeze(1)  # [batchsize, 1+num_neg]
        logits = similarity_matrix / temperature

        labels_ce = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        loss = self.criterion_ce(logits, labels_ce)
        # loss = torch.nn.CrossEntropyLoss(label_smoothing=0.2)(logits, labels_ce).to(device)
        return logits, labels_ce, loss


    def temporal_soft_loss(self, anchors, temporal_negatives, negative_ratios, label_emb):
        """
        Args: 
            anchors: [batch_size, dim=128],  # 只取一个view，不取positive
            temporal_negatives: [batch_size, num_neg, dim=128],
            negative_ratios: [batch_size, num_neg, label_num=12],
            label_emb: [label_num=12, out_dim=768]
        Loss: 
            \sum_i {\sum_j[y_ij * log(sigmoid(logits_ij)) + (1-y_ij) * log(1-sigmoid(logits_ij))]}
        Return:
            logits: 
            labels: soft scores
            loss: scalar
        """
        device = self.args.device
        if self.args.group:
            batch_size = self.args.group_num * self.args.group_element
        else:
            batch_size = self.args.batch_size
        num_neg = self.args.num_neg
        
        anchors = F.normalize(anchors, dim=1)  # [batch_size, 128]
        temporal_negatives = F.normalize(temporal_negatives, dim=2)  # [batch_size, num_neg, 128]
        soft_matrix = torch.matmul(negative_ratios, label_emb)  # [batch_size, num_neg, 768]
        soft_matrix = soft_matrix.view(batch_size*num_neg, -1)  # [batch_size*num_neg, 768]
        # 计算L2范数
        soft_labels = soft_matrix.norm(p=2, dim=1)  # [batch_size*num_neg]
        soft_labels = soft_labels.view(batch_size, num_neg)  # [batch_size, num_neg]
        
        # softlabel最大说明距离最远, 计算batchsize个最大的softlabel和相应的索引
        _, indices = torch.topk(soft_labels, k=1, dim=1)  # [batch_size, 1] 
        
        # todo: 虽然label_emb已经做好归一化了，但是乘上ratio可能存在大于1的问题，所以用torch.tanh来约束
        soft_labels = 1 - torch.tanh(2 * soft_labels)
        # print(soft_labels)
        similarity_matrix = torch.matmul(anchors.unsqueeze(1), temporal_negatives.transpose(1, 2))  
        # [batch_size, 1, num_neg]
        similarity_matrix = similarity_matrix.squeeze(1)  # [batch_size, num_neg]
        logits = similarity_matrix / self.args.temperature  # [batch_size, num_neg]
        
        # F.binary_cross_entropy_with_logits: -[y \cdot \log(\sigma(x)) + (1 - y) \cdot \log(1 - \sigma(x))]
        loss = F.binary_cross_entropy_with_logits(
            input=logits, 
            target=soft_labels,
            reduction='mean'
            )
        return logits, soft_labels, loss, indices, soft_matrix


    # tag:spatial
    def spatial_group_smoothing_loss(self, features):
        """
        features:  [batch_size, n_views, feature_dim=128]
        """
        device = self.args.device
        group_num = self.args.group_num
        group_element = self.args.group_element
        batch_size = group_element * group_num
        epsilon = self.args.epsilon
        # pdb.set_trace()
        
        features = F.normalize(features, dim=2)
        features_pos = features[:, 0]  # [group_num * group_element, dim]
        # print(f"features_pos 维度: {features_pos.shape}")
        features_anchor = features[:, 1]  # [group_num * group_element, dim]
        
        similarity = (torch.matmul(features_anchor, features_pos.T)) / self.args.temperature  # [batch_size, batch_size]
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        group_ids = torch.arange(group_num, device=device).repeat_interleave(group_element).view(-1, 1)  
        # [batchsize,] -> [batchsize, 1]
        group_mask = (group_ids == group_ids.t())
        intra_group_mask = group_mask & (~diag_mask)

        inter_group_mask = ~group_mask
        
        positive = similarity[diag_mask].view(-1, 1)  # [group_num * group_element, 1]
        soft_positive = similarity[intra_group_mask].view(-1, (group_element - 1))  
        # [group_num * group_element, (group_element - 1)]
        negative = similarity[inter_group_mask].view(-1, (batch_size - group_element))  
        # [(group_num * group_element)^2 - soft_positive.shape[0], (batch_size - group_element)]
        logits = torch.cat([positive, soft_positive, negative], dim=1)  # batch_size, batch_size
        labels_smooth = torch.zeros_like(logits) 
        labels_smooth[:, 0] = 1
        labels_smooth[:, 1:group_element] = epsilon
        logits_softmax = torch.softmax(logits, dim=1)  # [batch_size, batch_size]
        loss = -torch.mean(torch.sum(labels_smooth * torch.log(logits_softmax + 1e-6), dim=1))  # [batch_size,]
        
        return loss


    
    
    # def spatial_loss(self, features):
    #     """
    #     计算空间上的对比损失（标准 SimCLR 的 NT-Xent 损失）。
    #     还应当排除自身的相似度
    #     Args:
    #         features: [batch_size, n_views, feature_dim=128]
        
    #     仿写simclr的方法
    #     需要重新堆叠成[n_views, batch_size, feature_dim], 也就是前面一个batch后面一个batch
    #     """
    #     device = self.args.device
    #     batch_size = self.args.batch_size
    #     temperature = self.args.temperature
        
    #     features = features.transpose(0, 1).contiguous().view(self.args.n_views * batch_size, -1)
    #     # features: [batch_size, n_views, feature_dim] -> 
    #     # [n_views, batch_size, feature_dim] ->
    #     # [n_views * batch_size, feature_dim]
        
    #     features = F.normalize(features, dim=1)

    #     labels = torch.cat([torch.arange(batch_size) for i in range(self.args.n_views)], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #     labels = labels.to(device)

    #     similarity_matrix = torch.matmul(features, features.T)  # [batch_size, batch_size]

    #     mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
    #     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size, 1]

    #     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    #     # [batch_size, batch_size - 1]
        
    #     logits = torch.cat([positives, negatives], dim=1)  # [batch_size, batch_size]
    #     labels_ce = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

    #     logits = logits / temperature

    #     loss = self.criterion_ce(logits, labels_ce)

    #     return logits, labels_ce, loss

    # def temporal_loss(self, features, temporal_negatives):
    #     """
    #     JIABO LIU
    #     计算时间上的对比损失。
    #     Args:
    #         features (Tensor): [batch_size, n_views, feature_dim=128]
    #         temporal_negatives (Tensor): [batch_size, num_neg, feature_dim=128]

    #     Returns:
    #         Tensor: 时间损失
    #     """
    #     temperature = self.args.temperature
    #     device = self.args.device
        
    #     features = F.normalize(features, dim=2)  # [batch_size, n_views, feature_dim=128]
    #     temporal_negatives = F.normalize(temporal_negatives, dim=2)  # [batch_size, num_neg, feature_dim=128]

    #     # [batch_size, n_views, feature_dim=128]的摆放方式可以让后面交叉计算
    #     features_pos = features[:,1:]        # [batchsize, n_views-1, feature_dim]
    #     features_anchor = features[:,:1]        # [batchsize, 1, feature_dim]

    #     features = torch.cat([features_pos, temporal_negatives], dim=1)     # [batchsize, 1+num_neg, feature_dim]

    #     similarity_matrix = torch.matmul(features_anchor, features.transpose(1, 2)) # [batchsize, 1, 1+num_neg]
    #     similarity_matrix = similarity_matrix.squeeze(1)  # [batchsize, 1+num_neg]
    #     logits = similarity_matrix / temperature

    #     labels_ce = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    #     loss = self.criterion_ce(logits, labels_ce)
    #     return logits, labels_ce, loss
    
    
    # def spatial_temporal_loss(self, features, temporal_negatives):
    #     """
    #     计算空间上和时间上的对比损失
    #     Args:
    #         features (Tensor): [batch_size, n_views, feature_dim=128]
    #         temporal_negatives (Tensor): [batch_size. num_neg, feature_dim=128]
    #     """
    #     device = self.args.device
    #     temperature = self.args.temperature
    #     batch_size = self.args.batch_size
        
    #     features = features.transpose(0, 1).contiguous().view(self.args.n_views * batch_size, -1)
    #     # [batch_size, n_views, feature_dim=128] ->
    #     # [n_views, batch_size, feature_dim=128] ->
    #     # [n_views * batch_size, feature_dim=128]
    #     temporal_negatives = temporal_negatives.transpose(0, 1).contiguous()  
    #     # [num_neg, batch_size, feature_dim]      
        
    #     anchors = features[:self.args.batch_size]
    #     positives = features[self.args.batch_size:]
    #     negatives = temporal_negatives.reshape(-1, temporal_negatives.size(2))  # [num_neg*batch_size, dim]    

    #     anchors = F.normalize(anchors, dim=1)   # [batch_size, dim]
    #     positives = F.normalize(positives, dim=1)  # [batch_size, dim]   
    #     negatives = F.normalize(negatives, dim=1)  # [num_neg*batch_size, dim]
        
    #     anchor_pos_sim = torch.sum(anchors * positives, dim=1).unsqueeze(1)  # [batch_size, 1]
    #     anchor_neg_sim = torch.matmul(
    #         anchors, negatives.transpose(0, 1)
    #     )  # [batch_size, num_neg*batch_size]
        
    #     pos_anchor_sim = torch.sum(positives * anchors, dim=1).unsqueeze(1)  # [batch_size, 1]
    #     pos_neg_sim = torch.matmul(
    #         positives, negatives.transpose(0, 1)
    #     )  # [batch_size, num_neg*batch_size]
        
    #     anchor_logits = torch.cat([anchor_pos_sim, anchor_neg_sim], dim=1)  # [batch_size, 1+num_neg]
    #     pos_logits = torch.cat([pos_anchor_sim, pos_neg_sim], dim=1)  # [batch_size, 1+num_neg]
        
    #     logits = torch.cat([anchor_logits, pos_logits], dim=0)   # [2*batch_size, 1+num_neg]
    #     logits = logits / temperature
        
    #     # 第0行是正样本
    #     labels_ce = torch.zeros((2*batch_size,), dtype=torch.long, device=device)  # [2*batch_size, 1]
        
    #     loss = self.criterion_ce(logits, labels_ce)

    #     return logits, labels_ce, loss    
    
    


    # def temporal_loss_neg(self, features_anchor, features_negatives):
    #     """
    #     计算时间上的对比损失。
    #     只计算anchor与时间负样本的计算, anchor与正样本的计算只需要在空间上计算即可
    #
    #     Args:
    #         features_anchor (Tensor): [batch_size, feature_dim=128]
    #         features_negatives (Tensor): [batch_size, num_neg, feature_dim=128]
    #
    #     Returns:
    #         Tensor: 时间损失
    #     """
    #     # TODO: 由于时间对比损失只考虑了anchor与时间负样本之间的对比
    #     # 后期采用双encoder和双对称loss的时候也不用考虑anchor与时间正样本的对比
    #     device = self.args.device
    #     temperature = self.args.temperature
    #
    #     if features_negatives.numel() == 0:
    #         return torch.tensor(0.0).to(device)
    #
    #     features_anchor = F.normalize(features_anchor, dim=1)  # [batch_size, 128]
    #     features_negatives = F.normalize(features_negatives, dim=2)  # [batch_size, num_neg, 128]
    #
    #     similarity = torch.bmm(features_negatives, features_anchor.unsqueeze(2)).squeeze(2) / temperature
    #     # 负样本标签全为0
    #     labels_neg = torch.zeros_like(similarity).to(device)  # [batch_size, num_neg]
    #     loss = self.criterion_bce(similarity, labels_neg)
    #     return similarity, labels_neg, loss
    #
    #
    # def temporal_loss_pos_neg(self, features, features_negatives):
    #     """
    #     计算时间上的对比巡视
    #     计算了anchor与正样本与负样本的损失
    
    #     Args:
    #         features (Tensor): [2*batch_size, feature_dim=128]
    #         features_negatives (Tensor): [batch_size, num_neg, feature_dim=128]
    #     """
    #     device = self.args.device
    #     temperature = self.args.temperature
    
    #     anchors = features[:self.args.batch_size]
    #     temporal_positives = features[self.args.batch_size:]
    
    #     anchors = F.normalize(anchors, dim=1)                    # [batch_size, embedding_dim]
    #     temporal_positives = F.normalize(temporal_positives, dim=1) # [batch_size, embedding_dim]
    #     features_negatives = F.normalize(features_negatives, dim=2) # [batch_size, num_neg, embedding_dim]
    
    #     sim_pos = torch.sum(anchors * temporal_positives, dim=1, keepdim=True) # [batch_size, 1]
    #     sim_pos = sim_pos / temperature
    
    #     sim_neg = torch.bmm(features_negatives, anchors.unsqueeze(2)).squeeze(2) # [batch_size, num_neg]
    #     sim_neg = sim_neg / temperature
    
    #     logits = torch.cat([sim_pos, sim_neg], dim=1) # [batch_size, 1 + num_neg]
    
    #     labels_ce = torch.zeros(self.args.batch_size, dtype=torch.long).to(device) # [batch_size]
    
    #     loss = self.criterion_ce(logits, labels_ce)
    
    #     return logits, labels_ce, loss

    # def temporal_loss(self, features, features_negatives):
    #     """
    #     YUTIAN JIANG
    #     anchor要与所有负样本都计算一次相似度
    #     features: [2*batch_size, dim]
    #     features_negatives: [batch_size, num_neg, dim]
        
    #     return:    
    #         logits: [2*batch_size, 1+num_neg], 第0行是正样本
    #         labels_ce: [2*batch_size]
    #         loss: scalar
            
    #     anchor与positive是正样本对
    #     anchor与negative, positive与negative是负样本对
    #     """
    #     # TODO: 这个代码并没有利用batch内的所有负样本, 仍然属于单独计算每个样本对应的负样本的相似度
    #     # TODO: 如果是要计算每个样本对应负样本的相似度, 需要把负样本给拉平, 也就是[batch_size, batch_size * num_neg, dim]
    #     device = self.args.device
    #     temperature = self.args.temperature
    #     batch_size = self.args.batch_size
        
    #     anchors = features[:batch_size]  # [batch_size, dim]
    #     positives = features[batch_size:]  # [batch_size, dim]

    #     anchors = F.normalize(anchors, dim=1)         # [batch_size, dim]
    #     positives = F.normalize(positives, dim=1)     # [batch_size, dim]
    #     negatives = F.normalize(features_negatives, dim=2)     # [num_neg, batch_size, dim]
        
    #     # anchor-positive相似度
    #     anchor_pos_sim = torch.sum(anchors * positives, dim=1).unsqueeze(1)  # [batch_size, 1]
        
    #     # positive-anchor相似度
    #     pos_anchor_sim = torch.sum(positives * anchors, dim=1).unsqueeze(1)  # [batch_size, 1]
        
    #     # 计算每个anchor与所有的负样本的相似度
    #     anchors = anchors.unsqueeze(1)  # [batch_size, 1, dim]
    #     negatives = negatives.transpose(0, 1)  # [batch_size, num_neg, dim]
    #     anchor_neg_sim = torch.matmul(
    #         anchors, negatives.transpose(-1, -2)
    #         ).squeeze(1)  # [batch_size, num_neg]
        
    #     # 计算每个positive与所有的负样本的相似度
    #     positives = positives.unsqueeze(1)  # [batch_size, 1, dim]
    #     pos_neg_sim = torch.matmul(
    #         positives, negatives.transpose(-1, -2)
    #         ).squeeze(1)  # [batch_size, num_neg]
        
    #     anchor_logits = torch.cat([anchor_pos_sim, anchor_neg_sim], dim=1)  # [batch_size, 1+num_neg]
    #     pos_logits = torch.cat([pos_anchor_sim, pos_neg_sim], dim=1)  # [batch_size, 1+num_neg]
        
    #     logits = torch.cat([anchor_logits, pos_logits], dim=0)   # [2*batch_size, 1+num_neg]
    #     logits = logits / temperature
        
    #     # 第0行是正样本
    #     labels_ce = torch.zeros((2*batch_size,), dtype=torch.long, device=device)  # [2*batch_size, 1]
        
    #     loss = self.criterion_ce(logits, labels_ce)

    #     return logits, labels_ce, loss