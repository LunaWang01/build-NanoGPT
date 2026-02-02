# build-NanoGPT
从零复刻NanoGPT
#We always start with a dataset to train on. let's download the tiny shakespeare dataset
打开input.txt文件，把所有的文本作为字符串读入。r表示只读，用utf-8编码来读取文件，存到text变量中
with open ('input.txt', 'r',encoding='utf-8') as f:
    text = f.read()
print("length of dataset in characters:",len(text))#统计有多少个字符

chars=sorted(list(set(text)))#set去重，变成列表，按顺序排列
vocab_size=len(chars)#训练字符级模型的核心参数：词汇表大小
print("".join(chars))#拼成一个字符串
print(vocab_size)

#原版stoi={ch:i for i,ch in enumerate(chars)}
stoi={}#映射
for i,ch in enumerate(chars)
    stoi[ch]=i
#字典推导式，创建字符到数字的映射字典。enumerate给字符分配一个索引数字
#字符ch作为字典的键，索引i作为字典的值
itos={}#映射
for i,ch in enumerate(chars)
    itos[i]=ch
#原版itos={i:ch for i,ch in enumerate(chars)}#数字到字符的映射字典
def encode(s):#编码函数,s是要编码的字符串s=“hii”
    num_list=[]#空列表
    for c in s:#遍历每一个字符
        num=stoi[c]
        num_list.append(num)
    return num_list

#原版encode=lambda s:[stoi[c] for c in s]
def decode(l):
    char_list=[]
    for i in l:
        char=itos[i]
        char_list.append(char)
    return "".join(char_list)
#原版decode=lambda l:".join([itos[i] for i in l])

print(encode("hii there"))#测试可删除
print(decode(encode("hii there")))

#现在让我们对整个文本数据集进行编码并将其存储到 torch.Tensor 中
import torch
data=torch.tensor(encode(text),dtype=torch.long)#将所有字符变成一个整数序列，再转换成张量，并指定数据类型为long（64位整数）
print(data.shape,data.dtype)#张量形状，张量数据类型
print(data[:1000])
#出现整数序列，是对字符的翻译
#想把数据集分成一列和一个验证分隔
#一部分作为transformer训练数据，一小部分作为验证数据。为了了解我们的模型在多大程度上过拟合
#评估模型的泛化能力
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

block_size=8#模型一次能看到多少个字符，来预测下一个字符
train_data[:block_size+1]

x=train_data[:block_size]#前八个
y=train_data[1:block_size+1]#第二个到第九个 模型预测的目标字符
#x[0:8]预测y[7]
for t in range(block_size):#用前面的字符作为上下文，预测下一个字符
    context=x[:t+1]
    target=y[t]
    print(f"when input is{context} the target:{target}")
#根据给定的上下文，自动生成下一个最可能的字符，从而实现文本生成

#因为我们将开始对数据集中的随机位置进行采样，从中提取块

torch.manual_seed(1337)#设置随机数种子，使每次运行代码生成的随机结果都一样
batch_size=4#一次并行处理4条独立的文本序列
block_size=8#预测时最大上下文长度

def get_batch(split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1]for i in ix])
    return x,y
xb,yb=get_batch("train")
print("inputs:")
#四排八列，每一个都是训练集的一部分
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)
#最后它们会进入transformer，来创建损耗函数
#这个变换器会同时处理所有这些例子，然后查找正确的整数来预测张量y中的每个位置
print("————")

for b in range(batch_size):#遍历所有的样本，文本序列4
    for t in range(block_size):#遍历序列中的每一个位置
        context=xb[b,:t+1]#xb二维张量，形状是[batch_size,block_size]。从第b个样本，从开头到t+1个位置的所有元素
        target=yb[b,t]#yb与xb基本相同，整体向右移动了一位
        print(f"when input is {context.tolist()} the target:{target}")


print(xb)#我们有了一批要输入到transformer中的输入，开始将其输入到神经网络中


#最简单的神经网络开始，在语言建模的情况下，bigram语言模型
#二元语言模型Bigram language Model
#根据当前这一个词，来预测下一个最可能出现的词
import torch#导入库
import torch.nn as nn#获得实验的再现性
from torch.nn import functional as F
torch.manual_seed(1337)#使实验结果可复现，每次运行代码得到的随机数相同

#定义了一个模型类，它继承自PyTorch的nn.Module。是所有神经网络的基类
class BigramlanguageModel(nn.Module):
    def __init__(self,vocab_size):#类初始化。构造函数，词汇表大小，总共多少个token
        super().__init__()#调用父类的构造函数，继承nn.Module时必须写的
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)
        #创建一个标记嵌入表，输入一个词的索引，它就返回一个长度为vocab_size的向量
        #如果词汇表很小，vocab_size=5,索引就是01234
        #每个token都可以从查找表中直接读取下一个token的原始得分
    def forward(self,idx,targets=None):#前向传播。idx和targets都是[batch_size,block_size]
        #模型用现有工具做预测，对比标准答案，算出预测的误差
        #targets向右移动一位
        logits=self.token_embedding_table(idx)#用初始化的查找表，根据idx生成预测原始得分(logits)
        #输入中的每个整数都将引用该嵌入表，并将从该嵌入表中提取与其索引对应的一行
        #将idx喂给嵌入表，得到输出logits
        #logits是[batch_size,block_size,vocab_size](B,T,C)
        #对每个位置[batch_size,block_size]，模型预测下一个词是词汇表中每个词的得分。
        #得分越高，模型认为这个词越可能是下一个词
        if targets is None:
            loss =None
        else:
            B,T,C=logits.shape#解包logits的形状，把三个维度的数值拆分赋给BTC
            logits=logits.view(B*T,C)#展平logits以一维顺序展开它们，并将通道保留为第二维
        #原因是因为pytorch的交叉熵损失函数只接受二维输入，无法处理三维
            targets=targets.view(B*T)#变成一维
            loss=F.cross_entropy(logits,targets)#交叉熵损失，模型错题分数。
        # 对比模型预测得分l和标准得分（t）预测错了多少
        #损失函数，是预测和目标的交叉熵

        return logits,loss#输出预测得分和损失值。
    #用loss优化模型(调整查找表的数值),预测时，主要用logits来判断下一个token

    def generate(self,idx,max_new_tikens):#生成文本。根据初始token自动生成新的token序列
        #idx(B,T) 要生成的新token数量
        for _ in range(max_new_tokens):#_循环变量用不到。生成几个token循环几次
            logits,loss=self(idx)#调用模型的前向传播。根据idx生成预测得分和损失
            logits=logits[:,-1,:]#把（BTC）变成（BC）只保留最后一个token的预测得分
            probs=F.softmax(logits,dim=-1)#把原始得分转换成概率，形状保持（BC）不变
            idx_next=torch.multinomial(probs,num_samples=1)#根据概率分布采样出下一个token，输出形状为（B,1)
            #采样函数，保留随机性
            idx=torch.cat((idx,idx_next),dim=1)
            #把新生成的token拼接到原来的序列之后，更新token序列
            #dim=1代表时间步t的维度拼接


        return idx

m=BigramlanguageModel(vocab_size)#创建了一个模型实例
logits,loss=m(xb,yb)#将输入数据xb和目标yb喂给模型，得到输出out（也就是logits）
print(logits.shape)
print(loss)

print(decode(m.generate(idx=torch.zeros((1,1),dtype=torch.long),max_new_tokens=100)[0].tolist()))


#创建优化器
optimizer=torch.optim.AdamW(m.parameters(),lr=le-3)

batch_size=32
for steps in range(100):
    xb,yb=get_batch("train")
    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)#清除上一轮迭代中计算出的梯度
    loss.backward()#执行反向传播，计算损失对模型中所有参数的梯度
    optimizer.step()#根据计算出来的梯度，更新模型的参数

print(loss.item())

print(decode(m.generate(idx=torch.zeros((1,1),dtype=torch.long),max_new_tokens=100)[0].tolist()))
