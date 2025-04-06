import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1,bias=False))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()

        self.embed = ConvBlock(1, 4, 4, normalization='batchnorm')
        self.latent = ConvBlock(1, 4, 4, normalization='batchnorm')

        self.weights = nn.Parameter(torch.empty(4))  
        nn.init.normal_(self.weights, mean=0.0, std=0.1)  
        
    def align_pstarmix(self, input):
        input = self.embed(input)
        input = self.latent(input)
        t1 = input[:,1,:].unsqueeze(1)
        t2 = input[:,3,:].unsqueeze(1)
        t1ce = input[:,2,:].unsqueeze(1)
        flair = input[:,0,:].unsqueeze(1)

        w = F.softmax(self.weights, dim=0)
        w_sum = self.weights.sum()
        w = self.weights / w_sum
        
        p_mix = w[0] * t1.detach() + w[1] * t2.detach() + \
                w[2] * t1ce.detach() + w[3] * flair.detach()

        align_loss = F.mse_loss(t1, p_mix) + \
                     F.mse_loss(t2, p_mix) + \
                     F.mse_loss(t1ce, p_mix) + \
                     F.mse_loss(flair, p_mix)
        

        input = torch.cat((flair, t1, t1ce, t2), 1) 
        return input, align_loss/4

    def align_pNmix(self, input):
        align_loss = 0
        for f in range(input.size()[1]):
            # print(input[:,f,:].size())
            mean = self.mean_enc
            # log_var = self.log_var_enc
            var = self.log_var_enc
            
            vlb = (mean - input[:,f,:]).pow(2).div(var) + var
            align_loss += vlb.mean() / 2
        return input, align_loss/4

    
    def forward(self, x, mask):
        # x, medmap_loss = self.align_pstarmix(x)
        x, medmap_loss = self.align_pNmix(x)
        
        ########ORIGINAL MODEL#######
        # x -> fuse_pred

        if self.is_training:
            return fuse_pred, medmap_loss
        return fuse_pred
