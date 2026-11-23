"""Batched inference with per-game parameters and one native attention dispatch.

Same Transformer equations and learned tensors; no vmap attention fallback that
would run each active network sequentially on MPS. Inference has no autograd.
"""
import torch
from torch.nn import functional as F


def forward(model,params,batch):
    if model.representation == 'player-relative-v1':
        from .spatial import forward_spatial
        return forward_spatial(model,batch,params=params)
    b,d = batch['board'].shape[0],model.tile.embedding_dim
    def embedding(name,indices):
        if indices.ndim==1:
            indices=indices[None].expand(b,-1)
        return params[name+'.weight'].gather(1,indices[:,:,None].expand(-1,-1,d))
    def linear(name,x):
        output=x.bmm(params[name+'.weight'].transpose(1,2))
        bias=params.get(name+'.bias')
        return output if bias is None else output+bias[:,None]
    def norm(name,x):
        return F.layer_norm(x,(d,),eps=1e-5)*params[name+'.weight'][:,None]+params[name+'.bias'][:,None]
    spatial=embedding('tile',batch['board'])+embedding('garden',model.point_types)+embedding('row',model.rows)+embedding('column',model.columns)
    context=linear('context',batch['context'][:,None])+embedding('phase',batch['phase'][:,None])+embedding('player',batch['player'][:,None])
    context=context+embedding('action_tile',batch['setup'][:,0,None]+1)+linear('setup_context',batch['setup'][:,None,1:].float()/2.0)
    tokens=torch.cat((params['global_token'].squeeze(1)+context,spatial),dim=1)
    history=batch['history']
    padding=None
    if history.shape[1]:
        def coordinates(indices):
            value=embedding('row',(indices//17).clamp(max=16))+embedding('column',indices%17)
            return value*(indices!=289)[:,:,None]
        past=embedding('family',history[:,:,0])+embedding('action_tile',history[:,:,1])+embedding('player',history[:,:,4])
        past=past+linear('history_source',coordinates(history[:,:,2]))+linear('history_destination',coordinates(history[:,:,3]))+params['history_type'].squeeze(1)
        dimensions=torch.arange(0,d,2,device=tokens.device,dtype=torch.float32)
        angles=batch['history_positions'][:,:,None]*torch.exp(-dimensions*(9.210340371976184/d))
        chronological=torch.stack((angles.sin(),angles.cos()),dim=-1).flatten(-2)
        padding=torch.cat((torch.zeros((b,tokens.shape[1]),dtype=torch.bool,device=tokens.device),~batch['history_mask']),dim=1)
        tokens=torch.cat((tokens,past+chronological[:,:,:d]),dim=1)
    attention_mask=None if padding is None else (~padding)[:,None,None,:]
    for i,block in enumerate(model.blocks):
        name=f'blocks.{i}'
        heads=block.self_attn.num_heads
        x=norm(name+'.norm1',tokens)
        q,k,v=linear(name+'.self_attn.in_proj',x).chunk(3,dim=-1)
        def split(value):
            return value.reshape(b,-1,heads,d//heads).transpose(1,2)
        output=F.scaled_dot_product_attention(split(q),split(k),split(v),attn_mask=attention_mask,dropout_p=0.0)
        output=output.transpose(1,2).reshape(b,-1,d)
        tokens=tokens+linear(name+'.self_attn.out_proj',output)
        x=norm(name+'.norm2',tokens)
        tokens=tokens+linear(name+'.linear2',F.gelu(linear(name+'.linear1',x)))
    tokens=norm('final_norm',tokens)
    actions=batch['actions']
    family=embedding('family',actions[:,:,0])+embedding('action_tile',actions[:,:,1])
    def gather(indices):
        mapped=model.dense_to_token[indices]
        return tokens.gather(1,mapped[:,:,None].expand(-1,-1,d))
    features=torch.cat((tokens[:,:1].expand(-1,actions.shape[1],-1),gather(actions[:,:,2]),gather(actions[:,:,3]),family),dim=-1)
    logits=linear('action_score.2',F.gelu(linear('action_score.0',features))).squeeze(-1)
    return logits.masked_fill(~batch['mask'],float('-inf'))


def stack(states,model):
    # MultiheadAttention names its packed projection without a dot before
    # weight/bias; normalize those two keys for the common linear helper.
    result={name:torch.stack([state[name] for state in states]) for name,_ in model.named_parameters()}
    for i in range(len(model.blocks)):
        for suffix in ('weight','bias'):
            original=f'blocks.{i}.self_attn.in_proj_{suffix}'
            result[f'blocks.{i}.self_attn.in_proj.{suffix}']=result.pop(original)
    return result
