import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
from PIL import Image

import clip
import translate_main


# def dataProcessing(listTest):
#     res1 = []
#     for i1 in range(len(listTest)):
#         if listTest[i1] < 0.5:
#             res1.append(listTest[i1])
#         elif 0.5 <= listTest[i1] <= 0.7:
#             res1.append(listTest[i1] + 0.3000)
#         else:
#             res1.append(listTest[i1])
#     return res1


def getMAE(myList):
    avg = sum(myList) / len(myList)
    err = []
    for i in range(len(myList)):
        err.append(abs(myList[i] - avg))
    return sum(err) / len(err)


def getPicName(myLine):
    resName = ''
    if "#enc#0 " in myLine:
        resName = myLine.split("#enc#0 ")[0]
    elif "#zhc#1 " in myLine:
        resName = myLine.split("#zhc#1 ")[0]
    else:
        resName = myLine.split("#zhc#0 ")[0]
    return resName


def getPicSentence(myLine):
    resName = ''
    if "#enc#0 " in myLine:
        resName = myLine.split("#enc#0 ")[1]
    elif "#zhc#1 " in myLine:
        resName = myLine.split("#zhc#1 ")[1]
    else:
        resName = myLine.split("#zhc#0 ")[1]
    return resName


def getAcc10(myStr):
    # 正确语句
    r = random.randint(0, 5)
    if myStr == 'en':
        picRootPath = 'D:\\txtimg\\text2imageTest\\dataset\\enc_image\\'
        f = open("./dataset/enc_label.txt", encoding='utf-8')
    else:
        picRootPath = 'D:\\txtimg\\text4imageTest\\dataset\\ch_image\\'
        f = open("./dataset/ch_label.txt", encoding='GBK')
    resList = []
    pathList = []
    while 1:
        lines = f.readlines(10000)
        if not lines:
            break
        for line in lines:
            resList.append(getPicSentence(line))
            pathList.append(getPicName(line))
    picPath = picRootPath + pathList[r]
    print(picPath)
    r1 = random.randint(6, 10)
    r2 = random.randint(11, 20)
    r3 = random.randint(21, 30)
    r4 = random.randint(31, 40)
    r5 = random.randint(41, 50)
    r6 = random.randint(51, 60)
    r7 = random.randint(61, 70)
    r8 = random.randint(71, 80)
    r9 = random.randint(81, 99)

    t1 = resList[r]
    t2 = resList[r1]
    t3 = resList[r2]
    t4 = resList[r3]
    t5 = resList[r4]
    t6 = resList[r5]
    t7 = resList[r6]
    t8 = resList[r7]
    t9 = resList[r8]
    t10 = resList[r9]
    s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 = translate_main.trans10(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open(picPath)).unsqueeze(0).to(device)
    text = clip.tokenize(
        [str(s1), str(s2), str(s3), str(s4), str(s5), str(s6), str(s7), str(s8), str(s9), str(s10)]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # print("文本图像匹配度：", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
        prob = str(probs)[2:-2]
        # print(prob)
        tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10 = prob.split()
        # 格式化输出 更好看
        # 使用python内置的round（）函数
        # a = 1.1314 a = 1.0000 a = 1.1267
        # b = round（a.2）b = round（a.2）b = round（a.2）
        # output b = 1.13 output b = 1.0 output b = 1.13
        tx1 = round(float(tx1), 4)
        tx2 = round(float(tx2), 4)
        tx3 = round(float(tx3), 4)
        tx4 = round(float(tx4), 4)
        tx5 = round(float(tx5), 4)
        tx6 = round(float(tx6), 4)
        tx7 = round(float(tx7), 4)
        tx8 = round(float(tx8), 4)
        tx9 = round(float(tx9), 4)
        tx10 = round(float(tx10), 4)
        if tx1 >= 0.:
            return 1
        else:
            return 0
        # print(t1, t2, t3, t4, t5)
        # return tx1


def doExperiment(cycle, lan, words):
    acc = []
    step = []
    for i in range(5):
        for j in range(cycle):
            # getAcc10就是候选词为10
            if words == 5:
                res = getAcc5(lan)
            else:
                res = getAcc10(lan)
            step.append(res)
        acc.append(sum(step) / len(step))
    return acc


def getAcc5(myStr):
    # 随机抽图片及其对应正确语句
    r = random.randint(0, 10)
    if myStr == 'en':
        picRootPath = 'D:\\txtimg\\text2imageTest\\dataset\\enc_image\\'
        f = open("./dataset/enc_label.txt", encoding='utf-8')
    else:
        picRootPath = 'D:\\txtimg\\text4imageTest\\dataset\\ch_image\\'
        f = open("./dataset/ch_label.txt", encoding='GBK')
    resList = []
    pathList = []
    while 1:
        lines = f.readlines(10000)
        if not lines:
            break
        for line in lines:
            resList.append(getPicSentence(line))
            pathList.append(getPicName(line))

    picPath = picRootPath + pathList[r]
    print(picPath)
    r1 = random.randint(11, 20)
    r2 = random.randint(21, 40)
    r3 = random.randint(41, 60)
    r4 = random.randint(61, 99)

    t1 = resList[r]
    t2 = resList[r1]
    t3 = resList[r2]
    t4 = resList[r3]
    t5 = resList[r4]
    s1, s2, s3, s4, s5 = translate_main.trans(t1, t2, t3, t4, t5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open(picPath)).unsqueeze(0).to(device)
    text = clip.tokenize([str(s1), str(s2), str(s3), str(s4), str(s5)]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # print("文本图像匹配度：", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
        prob = str(probs)[2:-2]
        # print(prob)
        t1, t2, t3, t4, t5 = prob.split()
        # 格式化输出 更好看
        # 使用python内置的round（）函数
        # a = 1.1314 a = 1.0000 a = 1.1267
        # b = round（a.2）b = round（a.2）b = round（a.2）
        # output b = 1.13 output b = 1.0 output b = 1.13
        t1 = round(float(t1), 4)
        t2 = round(float(t2), 4)
        t3 = round(float(t3), 4)
        t4 = round(float(t4), 4)
        t5 = round(float(t5), 4)
    if t1 >= 0.4:
        return 1
    else:
        return 0
        # print(t1, t2, t3, t4, t5)
        # return t1


if __name__ == '__main__':
    # lan就是语言选择 en就是英文数据集 ch就是中文数据集
    language = 'ch'
    # 样本数量 10 20 50
    sample = 10
    # 候选词数量 5 10
    words = 5
    # 开始实验
    acc = doExperiment(sample, language, words)

    print(acc)
    MAX = max(acc)
    MIN = min(acc)
    AVG = sum(acc) / len(acc)
    MAE = getMAE(acc)
    STD = np.std(acc)
    print('AVG=', AVG)
    print('MAX=', MAX)
    print('MIN=', MIN)
    print('MAE=', MAE)
    print('STD=', STD)
