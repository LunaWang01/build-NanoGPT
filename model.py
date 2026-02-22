#导入
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn#引入Pytorch的神经网络模块
from torch.nn import functional as F#引入神经网络模块的函数库（F）

class LayerNorm(nn.Module):#定义lM的类，继承nn.Module，具备了神经网络层的所有基础功能
#层归一化
    def __init__(self,ndim,bias):#类（class）的初始化函数
    #初始化函数，创建一个对象，自动执行的函数。赋值属性
    #self：实例本身，ndim：输入特征的维度，bias：布尔值，用来决定是否要给这个层加上偏置
        super.__init__()#调用父类nn.Module的初始化函数

    #下面：全1权重和全0偏置（基础归一化）
        self.weight=nn.Parameter(torch.ones(ndim))#创建一个可学习的参数weight
        #这个张量是模型需要训练和更新的参数，创建一个长度为ndim的全1张量
        self.bias=nn.Parameter(torch.zeros(ndim)) if bias else None
        #如果bias是真的，就创建一个长度为ndim的全0张量作为偏置参数
        #如果False，就不加偏置


    def forward(self,input):#前向传播代码，执行层归一化的完整计算逻辑。
    #input：传入这一层的原始特征数据
        return F.layer_norm(input,self.weight.shape,self.weight,self.bias,1e-5)

#保证了模型在预测下一个词的时候，只能看到当前及之前的词，不能偷看后面的内容
class CausalSelfAttention(nn.Module):#因果自注意力层
    def __init__(self,config):
    #config:模型的配置文件
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #assert检查，验证特征维度可以被注意力头数整除。特征维度可以平均分给每个注意力头
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd,bias=config.bias)
        #创建了一个线性变换层，用来把输入的特征（n——embd维）一次性映射为三个向量
        #输出维度为3*n_embd
        self.c_proj = nn.Linear(config.n_embd,config.n_embd,bias=config.bias)
        #输出投影层：把注意力计算后的结果再映射回原来特征维度（n_embd）
        #让多头信息重新融合，维持维度不变，方便残差连接
        self.attn_dropout = nn.Dropout(config.dropout)
        #Dropout是一种正则化手段，训练时随机让一部分神经元输出为0，防止过拟合
        #作用在注意力权重，随机把一些注意力分数清零
        self.resid_dropout = nn.Dropout(config.dropout)
        #作用在在最终输出，增强鲁棒性
        #增加模型的泛化能力。利用所有维度的信息
        self.n_head = config.n_head
        #把配置文件中的注意力头数、特征维度和Dropout概率存到类的属性中
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional,"scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention.Flash Attention requires PyTorch>=2.0")

        #创建了一个下三角遮挡板（因果掩码）只能看见当前词和之前的词
        #创建一个全是1的正方形矩阵
        #只保留下三角的1，对角线以上全为0
        #改变矩阵形状，变成四维
        #存到模型中，叫bias
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size))
                                        .view(1,1,config.block_size,config.block_size))

    def forward(self,x):#多头自注意力模块
    #模型在看一段文字时，同时从多个角度关注文字里不同部分的关联
        B,T,C = x.size()#size拆分三个
        q,k,v = self.c_attn(x).split(self.n_embd,dim=2)
        #c_attn:BxTxC转换成BxTx3C
        #在第三个维度上切成三段，每个长度都是self.n_embd
        k = k.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
        #注意头的数量，每个注意头分到的维度
        #T和注意头数量交换位置
        q = q.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        #算注意力，两张方法一个快速函数一个手动一步步算
        if self.flash:#如果开启了flash attention
            y = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=self.dropout if self.training else 0, is_causal=True)
            #把qkv放进去，自动生成y
        else:
            att = (q @ k.transpose(-2,-1))*(1.0 / math.sqrt(k.size(-1)))
            #算匹配分数q乘k ，（转置是为了满足矩阵乘法）
            #缩放，防止数据爆炸
            att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
            #加掩码
            #self.bias 下三角全1矩阵。把0找出来可以布尔矩阵。0禁区变成负无穷
            att = F.softmax(att,dim=-1)
            #softmax：禁区变成0，其他分数，变成权重（每一行的和为1）
            #沿最后一个维度（T)做softmax
            #变成注意力权重矩阵
            att = self.attn_dropout(att)
            #随机把一部分权重变成0，避免死记硬背（随机丢弃）
            y = att @ v
            #加权求和
        y = y.transpose(1,2).comtiguous().view(B,T,C)
        #合并多头
        y = self.resid_dropout(self.c_proj(y))
        #输出投影
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd,bias=config.bias)
        #全连接层（升） 特征维度扩大了四倍
        self.gelu = nn.GELU()
        #激活函数，引入非线性能力
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd,bias=config.bias)
        #降维。还原原始维度，方便后面的残差连接
        self.dropout = nn.Dropout(config.dropout)
        #随机输出为0

    def forward(self,x):
        x=self.c_fc(x)#变成（B,T,4*n_embd)
        x=self.gelu(x)
        x=self.c_proj(x)
        x=self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self,config):
        self.ln_1 = LayerNorm(config.n_embd,bias=config.bias)#层归一化   pre ln
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd,bias=config.bias)#post ln
        self.mlp = MLP(config)


    def forward(self,x):
        x = x+self.attn(self.ln_1(x))#自注意力计算
        x = x+self.mlp(self.ln_2(x))#MLP
        return x

@dataclass
class GPTConfig:
    block_size: int=1024 #一次性能处理的最大序列长度
    vocab_size: int=50304#词表大小
    n_layer: int=12#堆叠12个block，层数越多，模型能力越强，参数量越大
    n_head: int=12#注意力头
    n_embd: int=768#词嵌入维度，维度越高，表达能力越强
    dropout: float = 0.0
    bias: bool=True#添加偏置


class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.vocab_size is not None#断言语句，确保config中包含词汇大小和序列最大长度这两个关键参数
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(#模块字典：管理模型各个子模块
            wte = nn.Embedding(config.vocab_size,config.n_embd),#词嵌入层(B,T,e_embd)
            wpe = nn.Embedding(config.block_size,config.n_embd),#位置嵌入层（T,n_embd)
            drop = nn.Dropout(config.dropout),#防止过拟合
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),# transformer block叠加
            ln_f = LayerNorm(config.n_embd,bias=config.bias),#最终层归一化
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)#语言模型头
        #把transformer 输出的e_embd维向量，映射回vocab——size维，用来预测下一个token的概率
        self.transformer.wte.weight = self.lm_head.weight#权重共享，减少模型的参数量

        self.apply(self.__init__weight)#遍历模型里面的所有子模块，对所有模块都调用weights初始化权重
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):#特殊化初始化，对以c_proj。weight结尾的参数
                #因果自注意力里面的最后一个映射
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layer))
                #使用正态分布，均值为0
        print("number of parameters:%.2fM"%(self.get_num_params()/1e6,))#模型可训练参数量，以一百万为单位


    def get_num_params(self,non_embedding=True):#会减去位置嵌入的参数
        n_params = sum(p.numel() for p in self.parameters())#模型总参数
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()#减去位置嵌入的参数
        return n_params


    def _init_weights(self,module):#初始化权重 module代表神经网络中的一个小零件
        if isinstance(module,nn.Linear):#检查module是不是nn.linear的类型
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)#如果是线性层，就用正态分布来初始化它的权重
            if module.bias is not None:#如果有偏置，初始化为0
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):#如果是嵌入层，也同样的正态分布来初始化它的权重
                torch.nn.init.normal_(module.weight, mean=0.0,std=0.02)


    def forward(self,idx,targets=None):#idx：输入数据，代表句子里的每个词。可选的目标值，用来计算损失
        device = idx.device#把输入数据所在的设备存起来，方便后面用
        b,t = idx.size#获取输入数据的形状，b:一次处理多少条数据，t：每条数据有多长
        assert t <= self.config.block_size, f"Cannot forward sequence of length{t}, block size is only {self.config.block_size} "
        #t不可以超过模型设定的最大长度
        pos= torch.arange(0,t,dtype=torch.long, device=device)
        #生成一个从0到t-1的整数序列，用来表示每个词在句子里的位置，位置编码
        tok_emb = self.transformer.wte(idx)#词索引变成词向量————词嵌入层（b，t，n_embd）
        pos_emb = self.transformer.wpe(pos)#位置嵌入层（t，n_embd）
        x =self.transformer.drop(tok_emb+pos_emb)#相加，dropout
        for block in self.transformer.h:#遍历transformer的每一个编码器块，让输入x依次经过每个编码器的处理
            x = block(x)
        x=self.transformer.ln_f(x)#最后层归一化得到最终输出


        if targets is not None:#如果提供了目标标签（targets）   训练模型
            logits = self.lm_head(x)#将模型的最终输出x通过语言模型头lm_head，得到每个词的预测分数
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1), ignore_index=-1)#计算损失函数
        else:#生成模式
            logits = self.lm_head(x[:,[-1],:])#只取序列最后一个位置的输出，通过lm得到预测分布
            loss = None#不计算损失

        return logits, loss



    def crop_block_size(self,block_size):
        assert block_size <= self.config.block_size#新长度不能比原来的最大长度还大
        self.config.block_size = block_size#更新配置
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])#裁剪位置编码
        #裁剪好的权重重新包装成模型可学习参数
        for block in self.transformer.h:#遍历模型所有transformer编码器块
            if hasattr(block.attn, 'bias'):#检查当前这一层的注意力模块，有没有bias的偏置矩阵
                block.attn.bias = block.attn.bias[:,:,block_size,:block_size]#裁剪注意力偏置，【1:1：

    #改变block_size： 会影响position embedding 以及attention中的bias（mask矩阵


    @classmethod
    def from_pretrained(cls,model_type,override_args=None):#加载预训练模型名称，一个可选的字典
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-x1'}#确保输入的是gpt2变体
        override_args = override_args or {}#没有提供override就设为空字典
        assert all(k == 'dropout' for k in override_args)#限制用户只能覆盖dropout这一个参数，保持模型结构稳定性
        from transformers import GPT2LMHeadModel
        print("loading weight from pretrained gpt:%s"%model_type)


        config_args = {#不同GPT2模型变体的核心配置
            'gtp2':dict(n_layer=12,n_head=12,n_embd=768),
            'gpt2-medium':dict(n_layer=24,n_head=16,n_embd=1024),
            'gpt2-large':dict(n_layer=36,n_head=20,n_embd=1280),
            'gpt2-x1':dict(n_layer=48,n_head=25,n_embd=1600),
        }[model_type]#根据传入的type选择对应的配置
        print("forcing vocab_size=50257,block_size=1024,bias=True")#强制覆盖词汇表大小和块大小
        config_args['vocab_size']=50257
        config_args['block_size']=1024
        config_args['bias']=True
        #确保和预训练权重兼容
        if 'dropout' in override_args:#覆盖dropout 如果在字典中指定了新的dropout率，就用新值覆盖配置
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout']=override_args['dropout']

        #创建一个从头初始化的模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


    def configure_optimizers(self,weight_decay,learning_rate,betas,device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items()if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim()>=2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim()<2]
        optim_groups = [
            {'params': decay_params,'weight_decay':weight_decay},
            {'params':nodecay_params,'weight_decay':0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_decay_params = sum(p.numel() for p in nodecay_params)
        print(f'num decayed parameter tensors：{len(decay_params)},with {num_decay_params:,}parameters')
        print(f'num non-decayed parameter tensors：{len(nodecay_params)},with {num_decay_params:,}parameters')
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups,lr=learning_rate,betas=betas,**extra_args)
        print(f"using fused AdamW:{use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx







