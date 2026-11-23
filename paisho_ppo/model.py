import torch
from torch import nn


class _StableEmbedding(torch.autograd.Function):
    """MPS embedding gradients via fixed-order matrix reduction, not atomics."""
    @staticmethod
    def forward(ctx, weight, indices):
        ctx.save_for_backward(indices)
        ctx.rows = weight.shape[0]
        return torch.nn.functional.embedding(indices, weight)

    @staticmethod
    def backward(ctx, gradient):
        indices, = ctx.saved_tensors
        selector = (indices.reshape(-1, 1) == torch.arange(ctx.rows, device=indices.device)[None]).to(gradient.dtype)
        return selector.transpose(0, 1).matmul(gradient.reshape(-1, gradient.shape[-1])), None


class StableEmbedding(nn.Embedding):
    def forward(self, indices):
        if self.weight.device.type == "mps" and torch.is_grad_enabled():
            return _StableEmbedding.apply(self.weight, indices)
        return super().forward(indices)


class _StableGather(torch.autograd.Function):
    """Only the 250 state tokens are action endpoints; histories remain intact."""
    @staticmethod
    def forward(ctx, tokens, indices, state_tokens):
        ctx.save_for_backward(indices)
        ctx.shape, ctx.state_tokens = tokens.shape, state_tokens
        return tokens.gather(1, indices[:, :, None].expand(-1, -1, tokens.shape[-1]))

    @staticmethod
    def backward(ctx, gradient):
        indices, = ctx.saved_tensors
        selector = (indices[:, :, None] == torch.arange(ctx.state_tokens, device=indices.device)[None, None]).to(gradient.dtype)
        prefix = selector.transpose(1, 2).bmm(gradient)
        suffix = gradient.new_zeros(ctx.shape[0], ctx.shape[1]-ctx.state_tokens, ctx.shape[2])
        return torch.cat((prefix, suffix), dim=1), None, None


def pack(observations, device, context_moves=12000):
    if type(context_moves) is not int or not 1 <= context_moves <= 12000:
        raise ValueError("context_moves must be in 1..12000")
    if not observations or any(not o["actions"] for o in observations):
        raise ValueError("model inputs must be nonterminal positions with legal actions")
    batch, maximum = len(observations), max(len(o["actions"]) for o in observations)
    actions = torch.zeros((batch, maximum, 4), dtype=torch.long)
    mask = torch.zeros((batch, maximum), dtype=torch.bool)
    for i, obs in enumerate(observations):
        actions[i, :len(obs["actions"])] = torch.as_tensor(obs["actions"], dtype=torch.long)
        mask[i, :len(obs["actions"])] = True
    result = {
        "board": torch.tensor([o["board"] for o in observations], dtype=torch.long),
        "phase": torch.tensor([o["phase"] for o in observations], dtype=torch.long),
        "player": torch.tensor([o["player"] for o in observations], dtype=torch.long),
        "context": torch.tensor([o["reserves"] + [o["turns"]] for o in observations], dtype=torch.float32),
        "actions": actions, "mask": mask,
        "setup": torch.tensor([o.get("setup", [0]+[1]*8) for o in observations], dtype=torch.long),
    }
    histories = [o.get("history", []) for o in observations]
    lengths = [o.get("history_length", len(h)) for o, h in zip(observations, histories)]
    if any(type(n) is not int or n < 0 or n > len(h) for n, h in zip(lengths, histories)):
        raise ValueError("invalid chronological history prefix")
    starts = [max(0, n-context_moves) for n in lengths]
    first_players = [h[0][4] if n else 1 for h,n in zip(histories,lengths)]
    if any(player not in (0,1) for player in first_players):
        raise ValueError("invalid first player in observed history")
    result["first_player"] = torch.tensor(first_players,dtype=torch.long)
    visible_lengths = [n-start for n, start in zip(lengths, starts)]
    maximum_history = max(visible_lengths)
    history = torch.zeros((batch, maximum_history, 5), dtype=torch.long)
    history_mask = torch.zeros((batch, maximum_history), dtype=torch.bool)
    history_positions = torch.zeros((batch, maximum_history), dtype=torch.float32)
    for i, (h, n, start, visible) in enumerate(zip(histories, lengths, starts, visible_lengths)):
        if visible:
            history[i, :visible] = torch.as_tensor(h[start:n], dtype=torch.long)
            history_mask[i, :visible] = True
            history_positions[i, :visible] = torch.arange(start, n, dtype=torch.float32)
    result.update(history=history, history_mask=history_mask, history_positions=history_positions)
    result["context"][:, :24] /= 3.0
    result["context"][:, 24] = torch.log1p(result["context"][:, 24]) / 8.0
    return {key: value.to(device) for key, value in result.items()}


class TransformerPlayer(nn.Module):
    """Board and chronological past-move Transformer; one legal-action output."""
    def __init__(self, config, points):
        super().__init__()
        d = config.width
        self.context_moves = config.context_moves
        self.representation = config.representation
        self.tile = StableEmbedding(25, d)
        self.garden = StableEmbedding(9, d)
        self.row = StableEmbedding(17, d)
        self.column = StableEmbedding(17, d)
        self.phase = StableEmbedding(2, d)
        self.player = StableEmbedding(2, d)
        self.context = nn.Linear(25, d)
        self.global_token = nn.Parameter(torch.zeros(1, 1, d))
        # Create blocks independently; no identical cloned-block initialization.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, config.heads, config.feedforward,
                                       dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(config.layers)
        ])
        self.final_norm = nn.LayerNorm(d)
        self.family = StableEmbedding(7, d)
        self.action_tile = StableEmbedding(13, d)
        self.action_score = nn.Sequential(nn.Linear(4*d, d), nn.GELU(), nn.Linear(d, 1))
        self.history_source = nn.Linear(d, d, bias=False)
        self.history_destination = nn.Linear(d, d, bias=False)
        self.history_type = nn.Parameter(torch.zeros(1, 1, d))
        self.setup_context = nn.Linear(8, d)
        points = torch.as_tensor(points, dtype=torch.long)
        self.register_buffer("point_types", points[:, 1])
        self.register_buffer("rows", points[:, 2])
        self.register_buffer("columns", points[:, 3])
        mapping = torch.zeros(290, dtype=torch.long)
        mapping[points[:, 0]] = torch.arange(1, len(points)+1)
        self.register_buffer("dense_to_token", mapping)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        # Start near a uniform distribution, while preserving gradient flow.
        nn.init.normal_(self.action_score[-1].weight, std=0.01)
        nn.init.zeros_(self.action_score[-1].bias)
        if self.representation == "player-relative-v1":
            from .spatial import attach_geometry
            attach_geometry(self,points.tolist(),config.heads)

    def forward(self, batch):
        if self.representation == "player-relative-v1":
            from .spatial import forward_spatial
            return forward_spatial(self,batch)
        spatial = (self.tile(batch["board"]) + self.garden(self.point_types)
                   + self.row(self.rows) + self.column(self.columns))
        context = (self.context(batch["context"]) + self.phase(batch["phase"])
                   + self.player(batch["player"]) + self.action_tile(batch["setup"][:, 0]+1)
                   + self.setup_context(batch["setup"][:, 1:].float()/2.0))
        tokens = torch.cat((self.global_token + context[:, None], spatial), dim=1)
        history = batch["history"]
        padding = None
        if history.shape[1]:
            def coordinates(indices):
                value = self.row((indices//17).clamp(max=16)) + self.column(indices%17)
                return value*(indices != 289)[:, :, None]
            past = (self.family(history[:, :, 0]) + self.action_tile(history[:, :, 1])
                    + self.player(history[:, :, 4])
                    + self.history_source(coordinates(history[:, :, 2]))
                    + self.history_destination(coordinates(history[:, :, 3])) + self.history_type)
            # Absolute move positions preserve order within the visible prefix.
            positions = batch["history_positions"][:, :, None]
            dimensions = torch.arange(0, tokens.shape[-1], 2, device=tokens.device, dtype=torch.float32)
            angles = positions*torch.exp(-dimensions*(9.210340371976184/tokens.shape[-1]))
            chronological = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
            past = past+chronological[:, :, :tokens.shape[-1]]
            prefix = tokens.shape[1]
            tokens = torch.cat((tokens, past), dim=1)
            padding = torch.cat((torch.zeros((len(tokens), prefix), dtype=torch.bool, device=tokens.device),
                                 ~batch["history_mask"]), dim=1)
        for block in self.blocks:
            tokens = block(tokens, src_key_padding_mask=padding)
        tokens = self.final_norm(tokens)
        actions = batch["actions"]
        family = self.family(actions[:, :, 0]) + self.action_tile(actions[:, :, 1])
        source = self.dense_to_token[actions[:, :, 2]]
        destination = self.dense_to_token[actions[:, :, 3]]
        def gather(index):
            if tokens.device.type == "mps" and torch.is_grad_enabled():
                return _StableGather.apply(tokens, index, len(self.rows)+1)
            return tokens.gather(1, index[:, :, None].expand(-1, -1, tokens.shape[-1]))
        global_context = tokens[:, :1].expand(-1, actions.shape[1], -1)
        features = torch.cat((global_context, gather(source), gather(destination), family), dim=-1)
        logits = self.action_score(features).squeeze(-1)
        return logits.masked_fill(~batch["mask"], float("-inf"))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    return torch.device(name)
