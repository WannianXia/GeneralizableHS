import torch
import torch.nn as nn
import torch.nn.functional as F


class Representation(nn.Module):
    def __init__(self, card_dim=16, lm_dim=768, embed_dim=256, dim_ff=256, num_entity=32, dropout=0.1):
        super(Representation, self).__init__()

        self.card_dim = card_dim
        self.embed_dim = embed_dim
        self.entity_dim = self.card_dim + self.embed_dim

        # Embedding layers
        self.lm_embedding = nn.Linear(lm_dim, embed_dim)
        self.secret_embedding = nn.Linear(lm_dim, self.entity_dim)
        
        self.hand_card_feat_embed = nn.Linear(23, card_dim)
        self.minion_embedding = nn.Linear(26, card_dim)  # 26+9 = 35
        self.hero_embedding = nn.Linear(31, card_dim)    # 31+16 = 47

        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.entity_dim, nhead=8, dim_feedforward=dim_ff, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=4)

        # Position encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_entity, self.entity_dim))

        # Initialize dump_secret_stat without gradient
        dump_secret_stat_init = torch.zeros(5, self.card_dim)
        self.register_buffer('dump_secret_stat', dump_secret_stat_init)

    def forward(self, hand_lm_embed, minion_lm_embed, secret_feat, weapon_lm_embed, hand_stats, minion_stats, hero_stats):
        # Embed LM outputs
        hand_lm_embed = self.lm_embedding(hand_lm_embed)
        minion_lm_embed = self.lm_embedding(minion_lm_embed)
        secret_feat = self.lm_embedding(secret_feat)
        weapon_lm_embed = self.lm_embedding(weapon_lm_embed)

        # Embed scalar features
        hand_card_feat = self.hand_card_feat_embed(hand_stats)
        minions_feat = self.minion_embedding(minion_stats)
        heros_feat = self.hero_embedding(hero_stats)


        # Concatenate embeddings
        hand_card_feat = torch.cat((hand_card_feat, hand_lm_embed), dim=-1)
        minions_feat = torch.cat((minions_feat, minion_lm_embed), dim=-1)
        heros_feat = torch.cat((heros_feat, weapon_lm_embed), dim=-1)
        if len(secret_feat.shape) == 3:
            secret_feat = torch.cat((secret_feat, self.dump_secret_stat.expand(secret_feat.size(0), -1, -1)), dim=-1)
        elif len(secret_feat.shape) == 2:
            secret_feat = torch.cat((secret_feat, self.dump_secret_stat), dim=-1)
        elif len(secret_feat.shape) == 4:
            secret_feat = torch.cat((secret_feat, self.dump_secret_stat.unsqueeze(0).expand(secret_feat.size(0), secret_feat.size(1), -1, -1)), dim=-1)

        entities = torch.cat((hand_card_feat, minions_feat, heros_feat, secret_feat), dim=-2)
        # Add position embedding
        entities = entities + self.pos_embedding


        # Transformer encoding
        obs_repr = self.transformer(entities.reshape(-1, 32, 272))
        

        return obs_repr



class WorldModel(nn.Module):
    def __init__(self, state_dim, dim_ff):
        super(WorldModel, self).__init__()
        self.state_dim = state_dim

        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=state_dim, nhead=8, dim_feedforward=dim_ff, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=4)


    def forward(self, x):

        obs_repr = self.transformer(x.reshape(-1, 32, self.state_dim))
        return obs_repr


class Actor(nn.Module):
    def __init__(self, embed_dim, dim_ff, llm_dim = 768):
        '''
            state_dim: input observation dimension, default=512,
            action_dim: the sentence embedding of chosen action, default=768. We use cosine similarity to compare with available action embeddings
        '''
        super(Actor, self).__init__()
        print('check deck strategy')

        self.embed_dim  = embed_dim
        self.llm_dim = llm_dim
        self.deck_embed = nn.Linear(self.llm_dim, embed_dim)
                
        self.output_layer = nn.Linear(embed_dim, embed_dim)
        self.scale_out = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=-2)
        )
        self.fn_out = nn.Linear(embed_dim, 1)

        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=dim_ff, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=4)


    
    def forward(self, x, deck):

        if deck is not None:
            deck = self.deck_embed(deck).reshape(-1, 1, self.embed_dim)
            x = torch.cat((deck, x), dim=-2)
        x = self.transformer(x)
        
        out = self.output_layer(x)
        out_scale = self.scale_out(x)
        out = out * out_scale
        out = torch.sum(out, dim=-2)
        out = self.fn_out(out)
        
        return out


class LanguageEncoder:
    def __init__(self, model, tokenizer, device):
        self.encoder = model
        self.tokenizer = tokenizer
        self.cache = {}
        self.device = device
        self.encoder.to(device)

    def to(self, device):
        self.device = device
        self.encoder = self.encoder.to(device)

    def tokens_to_device(self, tokens):
        tok_device = {}
        for key in tokens:
            tok_device[key] = tokens[key].to(self.device)
        return tok_device


    def encode_by_names(self, names):
        txt = []
        for name in names:
            if name is None:
                txt.append(None)
            else:
                description = name
                txt.append(description)

        encoded = []
        for sent in txt:
            if sent in self.cache.keys():
                encoded.append(self.cache[sent])
            elif sent is None:
                encoded.append(torch.zeros((1, 768)).to(self.device))
            else:
                encoded_input = self.tokenizer(sent, padding=True, truncation=True, max_length = 64, return_tensors='pt')
                encoded_input = self.tokens_to_device(encoded_input)
                with torch.no_grad():
                    model_output = self.encoder.encoder(**encoded_input)
                sent_embed = mean_pooling(model_output, encoded_input['attention_mask'])
                sent_embed = F.normalize(sent_embed, p=2, dim=1)
                encoded.append(sent_embed)
                self.cache[sent] = sent_embed
        if len(encoded) == 0:
            return None
        else:
            return torch.cat(encoded, dim=0)  # n * max_length * 768


    def encode_action(self, action):
        if action is not None:
            # TODO action.FullPrint() is not a good text representation of the action
            action = action.FullPrint()

            tokens = self.tokenizer(action, padding=True, truncation=True, return_tensors='pt')
            tokens = self.tokens_to_device(tokens)
            with torch.no_grad():
                model_output = self.encoder(**tokens)
            action_embed = mean_pooling(model_output, tokens['attention_mask'])
            action_embed = F.normalize(action_embed, p=2, dim=1)
        else:
            action_embed = torch.zeros(1, 768).to(self.device)
        return action_embed

    
    def encode_options(self, options):
        available_action_embed = []
        for option in options:
            action_embed = self.encode_action(option)
            available_action_embed.append(action_embed)
        available_action_embed = torch.stack(available_action_embed, dim=0)
        return available_action_embed




def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
