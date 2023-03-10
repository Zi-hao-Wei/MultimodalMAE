import numpy as np
from torch import nn
import torch
import model.clip as clip 
import yaml
import os
from torch.utils.data import DataLoader
from torchvision.datasets.coco import CocoCaptions
from utils.simple_tokenizer import SimpleTokenizer
from model.models_mae import mae_vit_base_patch16_dec512d8b

single_caption = True # choose if evalating only using the first caption
model_name = "ViT-B/16" #"RN50" #"RN50x4" #"RN101" #

def compute_similarity(image_features, text_features, bs = 1000):
    # compute similarity
    max_pairs = image_features.shape[0]
    similarity_scores = torch.zeros(max_pairs, max_pairs)
    for v in range(0, max_pairs, bs):
        for t in range(0, max_pairs, bs):
            print('Processing Visual '+str(v)+' Text '+str(t), end='\r')
            batch_visual_emb = image_features[v:v+bs]
            batch_caption_emb = text_features[t:t+bs]
            print(batch_visual_emb)
            logits = batch_visual_emb @ batch_caption_emb.t()
            similarity_scores[v:v+bs,t:t+bs] = logits

    print('Done similarity')
    return similarity_scores

def tokenize(text):
    context_length=77
    tokenizer = SimpleTokenizer()
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    text = text[0]
    tokens = [sot_token] + tokenizer.encode(text) + [eot_token]
    # print(tokens)
    result = torch.zeros(context_length, dtype=torch.long)
    result[:len(tokens)] = torch.tensor(tokens)
    mask = torch.zeros((context_length + 197, context_length + 197))
    mask[(197 + len(tokens)):, :] = -1e9
    mask[:, (197 + len(tokens)):] = -1e9
    return result, mask

def compute_retrieval(a2b_sims, return_ranks=True):
    """
    Args:
        a2b_sims: Result of computing similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # loop source embedding indices
    for index in range(npts):
        # get order of similarities to target embeddings
        inds = np.argsort(a2b_sims[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    report_dict = {"r1": r1, "r5": r5, "r10": r10, "r50": r50, "medr": medr, "meanr": meanr, "sum": r1 + r5 + r10}

    if return_ranks:
        return report_dict, (ranks, top1)
    else:
        return report_dict

print(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_name, device=device)
model = mae_vit_base_patch16_dec512d8b()
model_parameter = torch.load(r"F:\MultimodalMAE\clip-training\saved_checkpoints\checkpoint_0_1000.pt")["model_state_dict"]
model.load_state_dict(model_parameter,strict=True)
model.to(device)

data_root = "F:\mscoco"
train_root = os.path.join(data_root, 'train2017')
valid_root = os.path.join(data_root, 'val2017')
train_captions = os.path.join(data_root, 'annotations/captions_train2017.json')
valid_captions = os.path.join(data_root, 'annotations/captions_val2017.json')

valid_dataset = CocoCaptions(root = valid_root,
                        annFile = valid_captions,
                        transform = preprocess)
valid_dataloader = DataLoader(valid_dataset, batch_size = 1)

# fwd all samples
image_features = []
text_features = []
for batch_idx, batch in enumerate(valid_dataloader):
    print('Evaluating batch {}/{}'.format(batch_idx, len(valid_dataloader)), end = "\r")
    images, texts = batch
    if single_caption:
        texts = [texts[0][0]]
    else:
        texts = [txt[0] for txt in texts]
    images = images.cuda()
    texts, mask = tokenize(texts)
    texts = texts.cuda().unsqueeze(0)
    mask = mask.cuda().unsqueeze(0)
    image_emb, text_emb, loss = model.forward_finetune(images,texts,mask)

    # text_emb = texts  #embed with text encoder
    if not single_caption:
        text_emb = text_emb.unsqueeze(0)
    # images = images.cuda()
    # image_emb = model.encode_image(images) #embed with image encoder
    # print(batch_idx)
    # print(text_emb.squeeze().shape)
    text_features.append(text_emb.detach().cpu())
    image_features.append(image_emb.detach().cpu())
    if(batch_idx==100):
        break 


image_features = torch.cat(image_features, 0)
text_features = torch.cat(text_features, 0)
print('Done forward')

print(torch.std(image_features))
print(torch.std(text_features))



# normalized features
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True) 

if not single_caption:
    for cap_idx in range(text_features.shape[1]):
        similarity_scores = compute_similarity(image_features, text_features[:,cap_idx,:])
        i2t_dict = compute_retrieval(similarity_scores.numpy())
        t2i_dict = compute_retrieval(similarity_scores.t().numpy())
        print(cap_idx, 'i2t', i2t_dict)
        print(cap_idx, 't2i', t2i_dict)
else:
    # print(image_features.shape)
    # print(text_features.shape)
    image_features = image_features.float()
    text_features = text_features.float()

    similarity_scores = compute_similarity(image_features, text_features)
    i2t_dict = compute_retrieval(similarity_scores.numpy())
    t2i_dict = compute_retrieval(similarity_scores.t().numpy())
    print('i2t', i2t_dict)
    print('t2i', t2i_dict)