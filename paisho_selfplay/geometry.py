"""Identical spatial equations; immutable geometry and behavior biases may be reused."""
import torch
from torch.nn import functional as F
from paisho_ppo.spatial import canonicalize, StableRelativeBias


def forward_spatial(model,batch,params=None,canonical=False):
    """Same learned trunk/head, plus canonical inputs and relative board scores."""
    from paisho_ppo.model import _StableGather
    if not canonical:batch=canonicalize(model,batch)
    b,d=batch['board'].shape[0],model.tile.embedding_dim
    def embedding(name,indices):
        if params is not None:
            for label,constant in (('garden',model.point_types),('row',model.rows),('column',model.columns)):
                if name==label and indices is constant and f'_geometry.{label}' in params:
                    return params[f'_geometry.{label}']
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
    def family_tile(fields):
        table=None if params is None else params.get('_geometry.family_tile')
        if table is None:return embedding('family',fields[:,:,0])+embedding('action_tile',fields[:,:,1])
        indices=fields[:,:,0]*13+fields[:,:,1]
        return table.gather(1,indices[:,:,None].expand(-1,-1,d))
    spatial=embedding('tile',batch['board'])+embedding('garden',model.point_types)+embedding('row',model.rows)+embedding('column',model.columns)
    context=linear('context',batch['context'][:,None])+embedding('phase',batch['phase'][:,None])+embedding('player',batch['player'][:,None])
    context=context+embedding('action_tile',batch['setup'][:,0,None]+1)+linear('setup_context',batch['setup'][:,None,1:].float()/2.0)
    global_token=model.global_token if params is None else params['global_token'].squeeze(1)
    tokens=torch.cat((global_token+context,spatial),dim=1)
    history=batch['history']
    padding=None
    if history.shape[1]:
        def coordinates(indices):
            table=None if params is None else params.get('_geometry.coordinates')
            if table is not None:return table.gather(1,indices[:,:,None].expand(-1,-1,d))
            value=embedding('row',(indices//17).clamp(max=16))+embedding('column',indices%17)
            return value*(indices!=289)[:,:,None]
        past=family_tile(history)+embedding('player',history[:,:,4])
        history_type=model.history_type if params is None else params['history_type'].squeeze(1)
        past=past+linear('history_source',coordinates(history[:,:,2]))+linear('history_destination',coordinates(history[:,:,3]))+history_type
        table=None if params is None else params.get('_geometry.chronology')
        if table is None:
            dimensions=torch.arange(0,d,2,device=tokens.device,dtype=torch.float32)
            frequencies=(params.get('_geometry.frequencies') if params is not None else None)
            if frequencies is None:frequencies=torch.exp(-dimensions*(9.210340371976184/d))
            angles=batch['history_positions'][:,:,None]*frequencies
            chronological=torch.stack((angles.sin(),angles.cos()),dim=-1).flatten(-2)
        else:
            chronological=table[batch['history_positions'].long()]
        padding=torch.cat((torch.zeros((b,250),dtype=torch.bool,device=tokens.device),~batch['history_mask']),dim=1)
        tokens=torch.cat((tokens,past+chronological[:,:,:d]),dim=1)
    key_mask=None if padding is None else (~padding)[:,None,None,:]
    layer_bias=None if params is None else params.get('_geometry.layer_biases')
    if layer_bias is not None:
        layer_bias=F.pad(layer_bias,(0,tokens.shape[1]-250))
        if padding is not None:
            layer_bias=layer_bias.masked_fill(padding[None,:,None,None,:],float('-inf'))
    for i,block in enumerate(model.blocks):
        name=f'blocks.{i}'
        heads=block.self_attn.num_heads
        x=norm(name+'.norm1',tokens)
        q,k,v=linear(name+'.self_attn.in_proj',x).chunk(3,dim=-1)
        def split(value):
            return value.reshape(b,-1,heads,d//heads).transpose(1,2)
        q,k,v=split(q),split(k),split(v)
        weights=model.relative_bias[i] if params is None else params[f'relative_bias.{i}']
        if layer_bias is not None:bias=layer_bias[i]
        else:
            cached=None if params is None else params.get(f'_board_bias.{i}')
            if cached is None:
                board_bias=StableRelativeBias.apply(weights,model._relative_pairs,model._relative_inverse)
                bias=F.pad(board_bias,(1,0,1,0))
            else:bias=cached
            bias=F.pad(bias,(0,tokens.shape[1]-250))
            if params is None:bias=bias[None]
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
    family=family_tile(actions)
    def gather(indices):
        mapped=model.dense_to_token[indices]
        if params is None and tokens.device.type=='mps' and torch.is_grad_enabled():
            return _StableGather.apply(tokens,mapped,250)
        return tokens.gather(1,mapped[:,:,None].expand(-1,-1,d))
    features=torch.cat((tokens[:,:1].expand(-1,actions.shape[1],-1),gather(actions[:,:,2]),gather(actions[:,:,3]),family),dim=-1)
    logits=linear('action_score.2',F.gelu(linear('action_score.0',features))).squeeze(-1)
    return logits.masked_fill(~batch['mask'],float('-inf'))


def geometry(model,params):
    b=params['tile.weight'].shape[0];d=model.tile.embedding_dim
    for name,indices in (('garden',model.point_types),('row',model.rows),('column',model.columns)):
        key=f'_geometry.{name}'
        if key not in params:
            params[key]=params[name+'.weight'].gather(1,indices[None,:,None].expand(b,-1,d))
    if '_geometry.frequencies' not in params:
        dimensions=torch.arange(0,d,2,device=model.global_token.device,dtype=torch.float32)
        params['_geometry.frequencies']=torch.exp(-dimensions*(9.210340371976184/d))


def chronology(model,params,maximum_position=12000):
    # Pure positional encoding: independent of learned weights and game state.
    # Positions and padding are looked up unchanged, including padded position 0.
    if not hasattr(model,'_selfplay_chronology') or model._selfplay_chronology.shape[0]<maximum_position:
        d=model.tile.embedding_dim
        positions=torch.arange(max(12000,maximum_position),device=model.global_token.device,dtype=torch.float32)
        dimensions=torch.arange(0,d,2,device=positions.device,dtype=torch.float32)
        angles=positions[:,None]*torch.exp(-dimensions*(9.210340371976184/d))
        model._selfplay_chronology=torch.stack((angles.sin(),angles.cos()),dim=-1).flatten(-2)
    params['_geometry.chronology']=model._selfplay_chronology


def layer_biases(model,params):
    if '_geometry.layer_biases' not in params:
        params['_geometry.layer_biases']=torch.stack([params[f'_board_bias.{i}'] for i in range(len(model.blocks))])


def history_inputs(model,params):
    if '_geometry.coordinates' in params:return
    b,d=params['row.weight'].shape[0],model.tile.embedding_dim
    indices=torch.arange(290,device=model.global_token.device)
    def gather(name,index):
        return params[name+'.weight'].gather(1,index[None,:,None].expand(b,-1,d))
    coordinates=(gather('row',(indices//17).clamp(max=16))+gather('column',indices%17))*(indices!=289)[None,:,None]
    params['_geometry.coordinates']=coordinates
    params['_geometry.family_tile']=(params['family.weight'][:,:,None]+params['action_tile.weight'][:,None]).flatten(1,2)
