"""Player-oriented geometry and board-relative attention without a T-by-T bias."""
import torch
from torch import nn
from torch.nn import functional as F


class StableRelativeBias(torch.autograd.Function):
    """Repeated offset gradients reduce in fixed order instead of scatter-add."""
    @staticmethod
    def forward(ctx,weight,pairs,inverse):
        ctx.save_for_backward(inverse)
        ctx.pair_count=pairs.numel()
        return weight.index_select(-1,pairs).reshape(*weight.shape[:-1],249,249)

    @staticmethod
    def backward(ctx,gradient):
        inverse,=ctx.saved_tensors
        values=F.pad(gradient.flatten(-2),(0,1))
        reduced=values[...,inverse].sum(-1)
        return reduced,None,None


def attach_geometry(model,points,heads):
    index={point[0]:i for i,point in enumerate(points)}
    permutation=torch.tensor([index[288-point[0]] for point in points],dtype=torch.long)
    assert all(points[i][1]==points[j][1] for i,j in enumerate(permutation.tolist()))
    rows=torch.tensor([point[2] for point in points])
    columns=torch.tensor([point[3] for point in points])
    offsets=((rows[None]-rows[:,None]+16)*33+(columns[None]-columns[:,None]+16)).flatten()
    groups=[[] for _ in range(1089)]
    for pair,offset in enumerate(offsets.tolist()):
        groups[offset].append(pair)
    maximum=max(map(len,groups))
    inverse=torch.full((1089,maximum),len(offsets),dtype=torch.long)
    for offset,pairs in enumerate(groups):
        inverse[offset,:len(pairs)]=torch.tensor(pairs,dtype=torch.long)
    model.register_buffer('_canonical_index',permutation,persistent=False)
    model.register_buffer('_relative_pairs',offsets,persistent=False)
    model.register_buffer('_relative_inverse',inverse,persistent=False)
    model.relative_bias=nn.ParameterList([nn.Parameter(torch.zeros(heads,1089)) for _ in model.blocks])


def canonicalize(model,batch):
    """Self is always at South; first-move initiative remains explicit."""
    result=dict(batch)
    original_player=batch['player']
    rotate=original_player==0
    result['board']=torch.where(rotate[:,None],batch['board'].index_select(1,model._canonical_index),batch['board'])
    def points(indices):
        return torch.where(rotate[:,None] & (indices!=289),288-indices,indices)
    actions=batch['actions'].clone()
    actions[:,:,2]=points(actions[:,:,2])
    actions[:,:,3]=points(actions[:,:,3])
    result['actions']=actions.masked_fill(~batch['mask'][:,:,None],0)
    history=batch['history'].clone()
    if history.shape[1]:
        first=batch.get('first_player',batch['history'][:,0,4])
        first=torch.where(batch['history_mask'].any(1),first,torch.ones_like(first))
        history[:,:,2]=points(history[:,:,2])
        history[:,:,3]=points(history[:,:,3])
        history[:,:,4]=(batch['history'][:,:,4]==original_player[:,None]).long()
        history=history.masked_fill(~batch['history_mask'][:,:,None],0)
    else:
        first=batch.get('first_player',torch.ones_like(original_player))
    result['history']=history
    initiative=(original_player==first).long()
    result['player']=initiative
    result['first_player']=initiative
    setup=batch['setup']
    swapped=torch.cat((setup[:,:1],setup[:,5:9],setup[:,1:5]),dim=1)
    result['setup']=torch.where(rotate[:,None],swapped,setup)
    return result


def forward_spatial(model,batch,params=None):
    """Same learned trunk/head, plus canonical inputs and relative board scores."""
    from paisho_ppo.model import _StableGather
    batch=canonicalize(model,batch)
    b,d=batch['board'].shape[0],model.tile.embedding_dim
    def embedding(name,indices):
        if params is None:
            return model.get_submodule(name)(indices)
        if indices.ndim==1:
            indices=indices[None].expand(b,-1)
        return params[name+'.weight'].gather(1,indices[:,:,None].expand(-1,-1,d))
    def linear(name,x):
        if params is None:
            if name.endswith('.self_attn.in_proj'):
                attention=model.get_submodule(name.removesuffix('.in_proj'))
                return F.linear(x,attention.in_proj_weight,attention.in_proj_bias)
            return model.get_submodule(name)(x)
        output=x.bmm(params[name+'.weight'].transpose(1,2))
        bias=params.get(name+'.bias')
        return output if bias is None else output+bias[:,None]
    def norm(name,x):
        if params is None:
            return model.get_submodule(name)(x)
        return F.layer_norm(x,(d,),eps=1e-5)*params[name+'.weight'][:,None]+params[name+'.bias'][:,None]
    spatial=embedding('tile',batch['board'])+embedding('garden',model.point_types)+embedding('row',model.rows)+embedding('column',model.columns)
    context=linear('context',batch['context'][:,None])+embedding('phase',batch['phase'][:,None])+embedding('player',batch['player'][:,None])
    context=context+embedding('action_tile',batch['setup'][:,0,None]+1)+linear('setup_context',batch['setup'][:,None,1:].float()/2.0)
    global_token=model.global_token if params is None else params['global_token'].squeeze(1)
    tokens=torch.cat((global_token+context,spatial),dim=1)
    history=batch['history']
    padding=None
    if history.shape[1]:
        def coordinates(indices):
            value=embedding('row',(indices//17).clamp(max=16))+embedding('column',indices%17)
            return value*(indices!=289)[:,:,None]
        past=embedding('family',history[:,:,0])+embedding('action_tile',history[:,:,1])+embedding('player',history[:,:,4])
        history_type=model.history_type if params is None else params['history_type'].squeeze(1)
        past=past+linear('history_source',coordinates(history[:,:,2]))+linear('history_destination',coordinates(history[:,:,3]))+history_type
        dimensions=torch.arange(0,d,2,device=tokens.device,dtype=torch.float32)
        angles=batch['history_positions'][:,:,None]*torch.exp(-dimensions*(9.210340371976184/d))
        chronological=torch.stack((angles.sin(),angles.cos()),dim=-1).flatten(-2)
        padding=torch.cat((torch.zeros((b,250),dtype=torch.bool,device=tokens.device),~batch['history_mask']),dim=1)
        tokens=torch.cat((tokens,past+chronological[:,:,:d]),dim=1)
    key_mask=None if padding is None else (~padding)[:,None,None,:]
    for i,block in enumerate(model.blocks):
        name=f'blocks.{i}'
        heads=block.self_attn.num_heads
        x=norm(name+'.norm1',tokens)
        q,k,v=linear(name+'.self_attn.in_proj',x).chunk(3,dim=-1)
        def split(value):
            return value.reshape(b,-1,heads,d//heads).transpose(1,2)
        q,k,v=split(q),split(k),split(v)
        weights=model.relative_bias[i] if params is None else params[f'relative_bias.{i}']
        board_bias=StableRelativeBias.apply(weights,model._relative_pairs,model._relative_inverse)
        bias=F.pad(board_bias,(1,0,1,0))
        bias=F.pad(bias,(0,tokens.shape[1]-250))
        if params is None:
            bias=bias[None]
        if padding is not None:
            bias=bias.masked_fill(padding[:,None,None,:],float('-inf'))
        # Only the 250 state queries carry a spatial bias. History queries keep
        # fused attention and a broadcast padding mask, avoiding a T*T bias.
        state=F.scaled_dot_product_attention(q[:,:,:250],k,v,attn_mask=bias,dropout_p=0.0)
        if tokens.shape[1]>250:
            past=F.scaled_dot_product_attention(q[:,:,250:],k,v,attn_mask=key_mask,dropout_p=0.0)
            output=torch.cat((state,past),dim=2)
        else:
            output=state
        output=output.transpose(1,2).reshape(b,-1,d)
        tokens=tokens+linear(name+'.self_attn.out_proj',output)
        x=norm(name+'.norm2',tokens)
        tokens=tokens+linear(name+'.linear2',F.gelu(linear(name+'.linear1',x)))
    tokens=norm('final_norm',tokens)
    actions=batch['actions']
    family=embedding('family',actions[:,:,0])+embedding('action_tile',actions[:,:,1])
    def gather(indices):
        mapped=model.dense_to_token[indices]
        if params is None and tokens.device.type=='mps' and torch.is_grad_enabled():
            return _StableGather.apply(tokens,mapped,250)
        return tokens.gather(1,mapped[:,:,None].expand(-1,-1,d))
    features=torch.cat((tokens[:,:1].expand(-1,actions.shape[1],-1),gather(actions[:,:,2]),gather(actions[:,:,3]),family),dim=-1)
    logits=linear('action_score.2',F.gelu(linear('action_score.0',features))).squeeze(-1)
    return logits.masked_fill(~batch['mask'],float('-inf'))
