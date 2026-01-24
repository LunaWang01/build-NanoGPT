# build-NanoGPT
从零复刻NanoGPT
#We always start with a dataset to train on. let's download the tiny shakespeare dataset
打开input.txt文件，把所有的文本作为字符串读入。r表示只读，用utf-8编码来读取文件，存到text变量中
with open ('input.txt', 'r',encoding='utf-8') as f:
    text = f.read()
