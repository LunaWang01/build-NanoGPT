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
