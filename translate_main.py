from transformers import AutoModelWithLMHead, AutoTokenizer


def trans(s1, s2, s3, s4, s5):
    # 加载预训练模型
    mode_name = "trans_model"
    model = AutoModelWithLMHead.from_pretrained(mode_name)
    tokenizer = AutoTokenizer.from_pretrained(mode_name)
    # 开始翻译
    text1, text2, text3, text4, text5 = s1, s2, s3, s4, s5
    print(text1, text2, text3, text4, text5)
    # 步骤1：将文本变为token，返回pytorch的tensor
    tokenized_text1 = tokenizer.prepare_seq2seq_batch([text1], return_tensors='pt')
    tokenized_text2 = tokenizer.prepare_seq2seq_batch([text2], return_tensors='pt')
    tokenized_text3 = tokenizer.prepare_seq2seq_batch([text3], return_tensors='pt')
    tokenized_text4 = tokenizer.prepare_seq2seq_batch([text4], return_tensors='pt')
    tokenized_text5 = tokenizer.prepare_seq2seq_batch([text5], return_tensors='pt')
    # 也可以使用：
    # tokenized_text = tokenizer([text], return_tensors="pt")
    # 步骤2：通过模型，得到预测出的token
    translation1 = model.generate(**tokenized_text1)  # 执行翻译，返回翻译后的tensor
    translation2 = model.generate(**tokenized_text2)
    translation3 = model.generate(**tokenized_text3)
    translation4 = model.generate(**tokenized_text4)
    translation5 = model.generate(**tokenized_text5)
    # 步骤3：将预测出的token转为单词
    translated_text1 = tokenizer.batch_decode(translation1, skip_special_tokens=True)
    translated_text2 = tokenizer.batch_decode(translation2, skip_special_tokens=True)
    translated_text3 = tokenizer.batch_decode(translation3, skip_special_tokens=True)
    translated_text4 = tokenizer.batch_decode(translation4, skip_special_tokens=True)
    translated_text5 = tokenizer.batch_decode(translation5, skip_special_tokens=True)

    # 步骤4：去掉多余的符号
    finalText1 = str(translated_text1)[2:-2]
    finalText2 = str(translated_text2)[2:-2]
    finalText3 = str(translated_text3)[2:-2]
    finalText4 = str(translated_text4)[2:-2]
    finalText5 = str(translated_text5)[2:-2]
    # print(finalText1, finalText2, finalText3, finalText4, finalText5)
    return finalText1, finalText2, finalText3, finalText4, finalText5


def trans10(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10):
    mode_name = "trans_model"
    model = AutoModelWithLMHead.from_pretrained(mode_name)
    tokenizer = AutoTokenizer.from_pretrained(mode_name)
    # 开始翻译
    text1, text2, text3, text4, text5, text6, text7, text8, text9, text10 = s1, s2, s3, s4, s5, s6, s7, s8, s9, s10
    # 步骤1：将文本变为token，返回pytorch的tensor
    tokenized_text1 = tokenizer.prepare_seq2seq_batch([text1], return_tensors='pt')
    tokenized_text2 = tokenizer.prepare_seq2seq_batch([text2], return_tensors='pt')
    tokenized_text3 = tokenizer.prepare_seq2seq_batch([text3], return_tensors='pt')
    tokenized_text4 = tokenizer.prepare_seq2seq_batch([text4], return_tensors='pt')
    tokenized_text5 = tokenizer.prepare_seq2seq_batch([text5], return_tensors='pt')
    tokenized_text6 = tokenizer.prepare_seq2seq_batch([text6], return_tensors='pt')
    tokenized_text7 = tokenizer.prepare_seq2seq_batch([text7], return_tensors='pt')
    tokenized_text8 = tokenizer.prepare_seq2seq_batch([text8], return_tensors='pt')
    tokenized_text9 = tokenizer.prepare_seq2seq_batch([text9], return_tensors='pt')
    tokenized_text10 = tokenizer.prepare_seq2seq_batch([text10], return_tensors='pt')
    # 也可以使用：
    # tokenized_text = tokenizer([text], return_tensors="pt")
    # 步骤2：通过模型，得到预测出的token
    translation1 = model.generate(**tokenized_text1)  # 执行翻译，返回翻译后的tensor
    translation2 = model.generate(**tokenized_text2)
    translation3 = model.generate(**tokenized_text3)
    translation4 = model.generate(**tokenized_text4)
    translation5 = model.generate(**tokenized_text5)
    translation6 = model.generate(**tokenized_text6)
    translation7 = model.generate(**tokenized_text7)
    translation8 = model.generate(**tokenized_text8)
    translation9 = model.generate(**tokenized_text9)
    translation10 = model.generate(**tokenized_text10)
    # 步骤3：将预测出的token转为单词
    translated_text1 = tokenizer.batch_decode(translation1, skip_special_tokens=True)
    translated_text2 = tokenizer.batch_decode(translation2, skip_special_tokens=True)
    translated_text3 = tokenizer.batch_decode(translation3, skip_special_tokens=True)
    translated_text4 = tokenizer.batch_decode(translation4, skip_special_tokens=True)
    translated_text5 = tokenizer.batch_decode(translation5, skip_special_tokens=True)
    translated_text6 = tokenizer.batch_decode(translation6, skip_special_tokens=True)
    translated_text7 = tokenizer.batch_decode(translation7, skip_special_tokens=True)
    translated_text8 = tokenizer.batch_decode(translation8, skip_special_tokens=True)
    translated_text9 = tokenizer.batch_decode(translation9, skip_special_tokens=True)
    translated_text10 = tokenizer.batch_decode(translation10, skip_special_tokens=True)

    # 步骤4：去掉多余的符号
    finalText1 = str(translated_text1)[2:-2]
    finalText2 = str(translated_text2)[2:-2]
    finalText3 = str(translated_text3)[2:-2]
    finalText4 = str(translated_text4)[2:-2]
    finalText5 = str(translated_text5)[2:-2]
    finalText6 = str(translated_text6)[2:-2]
    finalText7 = str(translated_text7)[2:-2]
    finalText8 = str(translated_text8)[2:-2]
    finalText9 = str(translated_text9)[2:-2]
    finalText10 = str(translated_text10)[2:-2]

    # print(finalText1, finalText2, finalText3, finalText4, finalText5)
    return finalText1, finalText2, finalText3, finalText4, finalText5, finalText6, finalText7, finalText8, finalText9, finalText10
