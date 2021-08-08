import torch
import torch.nn as nn
import copy

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, input_key_padding_mask):
        # local attention
        src2, local_attention_weights = self.self_attn(src, src, src, key_padding_mask=input_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, local_attention_weights


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.multihead2 = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)


        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, global_input, input_key_padding_mask, position_embed):

        tgt2, global_attention_weights = self.multihead2(query=global_input+position_embed, key=global_input+position_embed,
                                                         value=global_input, key_padding_mask=input_key_padding_mask)
        tgt = global_input + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt, global_attention_weights


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, input, input_key_padding_mask):
        output = input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):
            output, local_attention_weights = layer(output, input_key_padding_mask)
            weights[i] = local_attention_weights
        if self.num_layers > 0:
            return output, weights
        else:
            return output, None


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, embed_dim):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers


    def forward(self, global_input, input_key_padding_mask, position_embed):

        output = global_input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):
            output, global_attention_weights = layer(output, input_key_padding_mask, position_embed)
            weights[i] = global_attention_weights

        if self.num_layers>0:
            return output, weights
        else:
            return output, None


class transformer(nn.Module):
    ''' Spatial Temporal Transformer
        local_attention: spatial encoder
        global_attention: temporal decoder
        position_embedding: frame encoding (window_size*dim)
        mode: both--use the features from both frames in the window
              latter--use the features from the latter frame in the window
    '''
    def __init__(self, enc_layer_num=1, dec_layer_num=3, embed_dim=1936, nhead=8, dim_feedforward=2048,
                 dropout=0.1, mode=None):
        super(transformer, self).__init__()
        self.mode = mode

        encoder_layer = TransformerEncoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout)
        self.local_attention = TransformerEncoder(encoder_layer, enc_layer_num)

        decoder_layer = TransformerDecoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout)

        self.global_attention = TransformerDecoder(decoder_layer, dec_layer_num, embed_dim)

        self.position_embedding = nn.Embedding(2, embed_dim) #present and next frame
        nn.init.uniform_(self.position_embedding.weight)


    def forward(self, features, im_idx):
        rel_idx = torch.arange(im_idx.shape[0])

        l = torch.sum(im_idx == torch.mode(im_idx)[0])  # the highest box number in the single frame
        b = int(im_idx[-1] + 1)
        rel_input = torch.zeros([l, b, features.shape[1]]).to(features.device)
        masks = torch.zeros([b, l], dtype=torch.uint8).to(features.device)
        # TODO Padding/Mask maybe don't need for-loop
        for i in range(b):
            rel_input[:torch.sum(im_idx == i), i, :] = features[im_idx == i]
            masks[i, torch.sum(im_idx == i):] = 1

        # spatial encoder
        local_output, local_attention_weights = self.local_attention(rel_input, masks)
        local_output = (local_output.permute(1, 0, 2)).contiguous().view(-1, features.shape[1])[masks.view(-1) == 0]

        global_input = torch.zeros([l * 2, b - 1, features.shape[1]]).to(features.device)
        position_embed = torch.zeros([l * 2, b - 1, features.shape[1]]).to(features.device)
        idx = -torch.ones([l * 2, b - 1]).to(features.device)
        idx_plus = -torch.ones([l * 2, b - 1], dtype=torch.long).to(features.device) #TODO

        # sliding window size = 2
        for j in range(b - 1):
            global_input[:torch.sum((im_idx == j) + (im_idx == j + 1)), j, :] = local_output[(im_idx == j) + (im_idx == j + 1)]
            idx[:torch.sum((im_idx == j) + (im_idx == j + 1)), j] = im_idx[(im_idx == j) + (im_idx == j + 1)]
            idx_plus[:torch.sum((im_idx == j) + (im_idx == j + 1)), j] = rel_idx[(im_idx == j) + (im_idx == j + 1)] #TODO

            position_embed[:torch.sum(im_idx == j), j, :] = self.position_embedding.weight[0]
            position_embed[torch.sum(im_idx == j):torch.sum(im_idx == j)+torch.sum(im_idx == j+1), j, :] = self.position_embedding.weight[1]

        global_masks = (torch.sum(global_input.view(-1, features.shape[1]),dim=1) == 0).view(l * 2, b - 1).permute(1, 0)
        # temporal decoder
        global_output, global_attention_weights = self.global_attention(global_input, global_masks, position_embed)

        output = torch.zeros_like(features)

        if self.mode == 'both':
            # both
            for j in range(b - 1):
                if j == 0:
                    output[im_idx == j] = global_output[:, j][idx[:, j] == j]

                if j == b - 2:
                    output[im_idx == j+1] = global_output[:, j][idx[:, j] == j+1]
                else:
                    output[im_idx == j + 1] = (global_output[:, j][idx[:, j] == j + 1] +
                                               global_output[:, j + 1][idx[:, j + 1] == j + 1]) / 2

        elif self.mode == 'latter':
            # later
            for j in range(b - 1):
                if j == 0:
                    output[im_idx == j] = global_output[:, j][idx[:, j] == j]

                output[im_idx == j + 1] = global_output[:, j][idx[:, j] == j + 1]

        return output, global_attention_weights, local_attention_weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

