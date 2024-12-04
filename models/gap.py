import torch.nn as nn
import torch
from models.functions import ReverseLayerF
import torch.nn.functional as F
from einops import rearrange, repeat

# def entropy(predictions: torch.Tensor, reduction='mean') -> torch.Tensor:
#     r"""Entropy of prediction.
#     The definition is:

#     .. math::
#         entropy(p) = - \sum_{c=1}^C p_c \log p_c

#     where C is number of classes.

#     Args:
#         predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
#         reduction (str, optional): Specifies the reduction to apply to the output:
#           ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
#           ``'mean'``: the sum of the output will be divided by the number of
#           elements in the output. Default: ``'mean'``

#     Shape:
#         - predictions: :math:`(minibatch, C)` where C means the number of classes.
#         - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
#     """
#     epsilon = 1e-5
#     H = -predictions * torch.log(predictions + epsilon)
#     H = H.sum(dim=1)
#     if reduction == 'mean':
#         return H.mean()
#     else:
#         return H

def merged_features(x1,x2):
    x = torch.cat((x1, x2), dim=1)
    return x

class Feature(nn.Module):
    def __init__(self, is_cat=True):
        super(Feature, self).__init__()
        self.is_cat = is_cat
        self.gru1 = nn.GRU(14, 64, 2, batch_first=True)
        self.fc_1 = nn.Linear(64, 128)
        self.fc_2 = nn.Linear(256, 128)

    # def forward(self, x, fx):
    #     feat, _ = self.gru1(x)
    #     x = F.relu(self.fc_1(feat[:, -1, :]))
    #     if self.is_cat:
    #         # 进行特征融合操作
    #         x = merged_features(x,fx)
    #         x = self.fc_2(x)
    def forward(self, x):
        feat, _ = self.gru1(x)
        x = F.relu(self.fc_1(feat[:, -1, :]))
        return x

class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.bn1_fc = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64) #64 64
        self.bn2_fc = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1) #64 1
        self.prob = prob

    def forward(self, x):
        # x = F.dropout(x, training=self.training, p=self.prob)
        # x = F.relu(self.bn1_fc(self.fc1(x)))
        # x = F.dropout(x, training=self.training, p=self.prob)
        # x = F.relu(self.bn2_fc(self.fc2(x)))
        # x = F.dropout(x, training=self.training, p=self.prob)
        # x = self.fc3(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x
# class Predictor(nn.Module):
#     def __init__(self, prob=0.5):
#         super(Predictor, self).__init__()
#         self.fc1 = nn.Linear(128, 32)
#         self.bn1_fc = nn.BatchNorm1d(64)
#         self.fc2 = nn.Linear(32, 16) #64 64
#         self.bn2_fc = nn.BatchNorm1d(64)
#         self.fc3 = nn.Linear(16, 1) #64 1
#         self.prob = prob

#     def forward(self, x):
#         # x = F.dropout(x, training=self.training, p=self.prob)
#         # x = F.relu(self.bn1_fc(self.fc1(x)))
#         # x = F.dropout(x, training=self.training, p=self.prob)
#         # x = F.relu(self.bn2_fc(self.fc2(x)))
#         # x = F.dropout(x, training=self.training, p=self.prob)
#         # x = self.fc3(x)

#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = x.squeeze(-1)
#         return x

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class Domain_classifier(nn.Module):
    
    def __init__(self):
        super(Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction='mean')

    def forward(self, f_s, f_t, constant):
        input = torch.cat((f_s, f_t), dim=0)
        input = GradReverse.grad_reverse(input, constant)
        logits = F.relu(self.fc1(input))
        logits = torch.sigmoid(self.fc2(logits)).double()

        label_src = torch.ones((f_s.size(0),1)).cuda()
        label_tgt = torch.zeros((f_t.size(0),1)).cuda()
        label_concat = torch.cat((label_src, label_tgt), 0).double()

        w = torch.ones_like(label_concat).double()

        loss = self.bce(logits, label_concat, w.view_as(logits))

        return loss

class AdversarialNetwork(nn.Module):
    def __init__(self):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(128, 1024)#128
        self.ad_layer2 = nn.Linear(1024,1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        x = self.sigmoid(x)
        return x

class CNN_SL_bn(nn.Module):
    def __init__(self):
        super(CNN_SL_bn, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(13, 32, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*50
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*50
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=1, dilation=1)) #128*32*45
        self.med_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*25, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU())
        # self.Classifier = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 128))        

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        # src = src.view(src.size(0), self.input_dim, -1)
        src = src.permute(0, 2, 1)
        # src = src.type(torch.FloatTensor)
        full_features = self.encoder(src)
        full_features = full_features.view(-1, 32*25)
        features = self.med_layer(full_features)
        return features

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*50
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*50
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=1, dilation=1)) #128*32*45
        self.med_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*25, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU())

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        # src = src.view(src.size(0), self.input_dim, -1)
        src = src.permute(0, 2, 1)
        # src = src.type(torch.FloatTensor)
        full_features = self.encoder(src)
        full_features = full_features.view(-1, 32*25)
        features = self.med_layer(full_features)
        return features

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, stride=1, padding=1, dilation=1), #128*32*50 tfd3-4 11 128
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1),# tfd3-4 3 32
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=1, dilation=1),# tfd3-4 3 32
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1),# tfd3-4 3 32
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=1, dilation=1), #128*32*45 tfd3-4 11
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.LSTM(20, 32, 2, batch_first=True, bidirectional=True))
        self.med_layer = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(32*22, 1024),
            # nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU())

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        # src = src.view(src.size(0), self.input_dim, -1)
        src = src.permute(0, 2, 1)
        # src = src.type(torch.FloatTensor)
        full_features,_ = self.encoder(src)
        full_features = full_features[:, -1, :]
        features = self.med_layer(full_features)
        return features

class BiLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, num_layers=3, output_size=128):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(self.num_directions * self.hidden_size, self.output_size)

    def forward(self, input_seq):
        output, _ = self.lstm(input_seq)
        output = output[:, -1, :]
        pred = F.relu(self.linear(output))  # pred()
        return pred


class Att(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 1)
        )

    def forward(self, x, mask=None):
        x = torch.unsqueeze(x, 0)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = torch.squeeze(out, 0)
        return out

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0., emb_dropout=0.):
        super().__init__()
        # num_patches = (seq_len // patch_size)  # ** 2
        patch_dim = channels * patch_size  # ** 2
        # assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        # self.patch_size = patch_size

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()



    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        # print(x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)
        x = self.transformer(x)
        c_t = self.to_cls_token(x[:, 0])

        return c_t


class Discriminator_AR(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self):
        """Init discriminator."""
        super(Discriminator_AR, self).__init__()

        self.AR_disc = nn.GRU(128, 64, 2, batch_first=True)
        self.DC = nn.Linear(64, 1)
    def forward(self, input):
        """Forward the discriminator."""
        # src_shape = [batch_size, seq_len, input_dim]
        input = input.view(input.size(0),-1, 128 )
        encoder_outputs, (encoder_hidden) = self.AR_disc(input)
        features = F.relu(encoder_outputs[:, -1, :])
        domain_output = self.DC(features)
        return domain_output
    def get_parameters(self):
        parameter_list = [{"params":self.AR_disc.parameters(), "lr_mult":0.01, 'decay_mult':1}, {"params":self.DC.parameters(), "lr_mult":0.01, 'decay_mult':1},]
        return parameter_list

class Discriminator_ATT(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self):
        """Init discriminator."""
        super(Discriminator_ATT, self).__init__()
        self.transformer= Seq_Transformer(patch_size=128, dim=64, depth=8, heads= 2 , mlp_dim=256)
        self.DC = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()
    def forward(self, input):
        """Forward the discriminator."""
        # src_shape = [batch_size, seq_len, input_dim]
        input = input.view(input.size(0),-1, 128 )
        features = self.transformer(input)
        domain_output = self.DC(features)
        return domain_output