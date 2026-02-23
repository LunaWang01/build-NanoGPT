```markdown
# NanoGPT

## 层归一化 LayerNorm

### 意义
稳定神经网络的训练，避免训练中产生梯度消失或者梯度爆炸；使神经网络更快收敛，训练过程更加稳定

对神经网络某一层的所有神经元的输出进行归一化，经过层归一化后，输出结果均值为0、方差为1。

### 计算公式
\[
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \text{weight} + \text{bias}
\]

*   **x**：输入的特征向量
*   **μ**：该样本(句子)内所有特征的均值
*   **σ**：该样本内所有特征的标准差
*   **ϵ**：防止分母为0的小常数
*   **weight**：缩放参数
*   **bias**：平移参数

### 初始化函数
*   **ndim**：输入特征的维度
*   **bias**：布尔值，平移参数
*   **weight**：缩放参数

### 基础归一化操作：
1.  创建一个长度为`ndim`的全1张量作为权重
2.  创建一个长度为`ndim`的全0张量作为偏置

### 前向传播
执行层归一化计算，输入和输出的形状相同，均为`(batch_size, seq_len, n_embd)`。
*   **batch_size**：一批有多少个句子
*   **seq_len**：每个句子有多少个词
*   **n_embd**：每个词的特征维度（比如 512、768）

### 序列数据特点
在Transformer模型中，序列数据的尺寸为`B × Q × D`。
*   **B**：Batch size，批量大小
*   **Q**：Sequence Length，序列长度
*   **D**：Features Dim，特征维度

层归一化仅作用于特征维度D。

---

## 因果自注意力层 CausalSelfAttention

### 核心意义
实现Mask掩码机制，避免模型看到未来的token。

### 初始化函数
**入参**：
*   **n_embd**：特征维度
*   **n_head**：注意力头数

**初始化步骤**：
1.  检查特征维度能否被注意力头数整除
2.  定义线性投影层`c_attn`，将输出维度转换为 `3 * n_embd`
3.  定义输出投影层`c_proj`，将注意力输出还原为原始特征维度
4.  定义`attn`注意力分数的dropout层
5.  定义`resid`输出的dropout层

### forward 多头自注意力模块
**核心维度定义**：
*   **B**：batch的数量（一次处理多少条样本）
*   **T**：序列长度（每个样本的token数量）
*   **C**：每个token的嵌入维度（embedding size）

**核心计算形状变化**：
1.  特征张量转换为 `(B, nh, T, hs)`（`nh`为注意力头数，`hs`为单头特征维度）
2.  注意力分数计算：`(B, nh, T, hs) × (B, nh, hs, T) = (B, nh, T, T)`
3.  施加因果自注意力mask，屏蔽未来位置的注意力分数

### 自注意力基础计算流程
**输入**：
*   输入序列 `X ∈ R^{n×d}`，其中`n`是序列长度，`d`是每个token的维度。
*   可学习的权重矩阵 `W_Q, W_K, W_V ∈ R^{d×d}`

**计算步骤**：
1.  计算Query、Key和Value：`Q = XW_Q`, `K = XW_K`, `V = XW_V`
2.  计算注意力分数：`A = softmax(\frac{QK^T}{\sqrt{d_k}})V`
    *   `QK^T`：计算Query和Key之间的点积，表示token之间的相关性。
    *   `\frac{1}{\sqrt{d_k}}`：缩放因子，防止点积值过大导致梯度不稳定。
    *   `softmax()`：对每一行进行归一化，得到注意力权重。
3.  最终结果为对Value矩阵`V`的加权求和。

---

## MLP 多层感知机
由神经元组成，包含输入层、隐藏层、输出层，在Transformer中执行feed forward层的计算逻辑。

---

## 构建GPT的transformer block

### 配置：GPTconfig
```python
@dataclass
class GPTConfig:
    block_size: int = 1024  # 一次性能处理的最大序列长度
    vocab_size: int = 50304 # 词表大小
    n_layer: int = 12       # 堆叠12个block，层数越多，模型能力越强，参数量越大
    n_head: int = 12        # 注意力头数
    n_embd: int = 768       # 词嵌入维度，维度越高，表达能力越强
    dropout: float = 0.0    # dropout概率
    bias: bool = True       # 是否添加偏置

class GPT(nn.Module):
    pass
```

GPT模型核心模块

__init__ 初始化函数

1. 执行参数合法性断言检查。
2. 构建transformer核心网络结构。
3. 定义lm_head语言模型头。
4. 执行模型权重初始化。

get_num_params 方法

计算模型的可训练参数量（单位：百万，M），计算时需减去位置嵌入的参数。

init_weights 权重初始化方法

采用正态分布对模型的权重进行初始化。

前向传播

计算流程：输入词索引 → 词嵌入 + 位置嵌入 → Dropout → 编码器块处理 → 层归一化 → 输出

功能：接收一段文本（词索引序列idx），通过 Transformer 模型的计算，输出对“下一个词”的预测分布（logits）；如果提供了目标标签targets，则同时计算出预测与真实值之间的损失（loss）。

crop_block_size 方法

修改模型支持的最大序列长度（block_size），将其调小，使模型能够处理更短的句子。

from_pretrained 方法

核心作用：从预训练模型（如 GPT-2）加载权重和配置，并转换适配到当前代码的模型中。

实现逻辑：做权重映射，将 Hugging Face 官方 GPT-2 的权重，“翻译”并复制到自定义的minGPT模型里。

配置优化器

为GPT模型配置训练所需的优化器，实现模型参数的优化更新。

```
```
