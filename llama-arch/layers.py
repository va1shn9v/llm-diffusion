import torch
import torch.nn as nn
from einops import rearrange

class LlamaEmb(nn.Module):
    def __init__(self,embed_dim, vocab_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.voacb_len = vocab_len
        self.embedding = nn.Embedding(vocab_len,embed_dim)
    
    def forward(self,x):
        return self.embedding(x)

class RMSNorm(nn.Module):
    def __init__(self,input_dim,eps = None):
        super().__init__()
        self.input_dim = input_dim
        self.g_scale = nn.Parameter(torch.ones(1,input_dim))
        self.eps = eps
    
    def forward(self,x):
        if self.eps:
            rms = torch.sqrt(torch.sum(x**2,dim = -1)/x.shape[-1] + self.eps)
        else:
            rms = torch.sqrt(torch.sum(x**2,dim = -1)/x.shape[-1])
        # print("RMS shape : ")
        # print(rms.shape)
        out = (x/rms.view(x.shape[0],x.shape[1],1)) * self.g_scale
        return out

class Rope(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    def forward(self,x):
        pass

class SwiGLU(nn.Module):


    

if __name__ == "__main__":
    emb = LlamaEmb(100,32)
    test_tensor = torch.randint(low=0,high=20,size=(5,8))
    print("---- Test Tesnor -----\n")
    print(test_tensor)
    print(test_tensor.shape)
    out = emb(test_tensor)
    print("----Output SHape: \n ")
    print(out.shape)
    rms_layer = RMSNorm(100)
    rm_out = rms_layer(out)
    print(rm_out.shape)

