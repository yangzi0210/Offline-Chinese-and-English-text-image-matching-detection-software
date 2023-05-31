import torch
from PIL import Image

import clip
import translate_main


def matching(str1, str2, str3, str4, str5):
    t1 = str1
    t2 = str2
    t3 = str3
    t4 = str4
    t5 = str5
    s1, s2, s3, s4, s5 = translate_main.trans(t1, t2, t3, t4, t5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open("./pic/一架飞机的卫星图像_1.jpg")).unsqueeze(0).to(device)
    text = clip.tokenize([str(s1), str(s2), str(s3), str(s4), str(s5)]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("文本图像匹配度：", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
        prob = str(probs)[2:-2]
        print(prob)
        t1, t2, t3, t4, t5 = prob.split()
        print(t1, t2, t3, t4, t5)
        return t1, t2, t3, t4, t5
