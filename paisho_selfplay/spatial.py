"""Identical spatial equations; immutable behavior biases may be reused."""
import torch
from torch.nn import functional as F
from paisho_ppo.spatial import canonicalize, StableRelativeBias


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
        cached=None if params is None else params.get(f'_board_bias.{i}')
        if cached is None:
            board_bias=StableRelativeBias.apply(weights,model._relative_pairs,model._relative_inverse)
            bias=F.pad(board_bias,(1,0,1,0))
        else:
            bias=cached
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
