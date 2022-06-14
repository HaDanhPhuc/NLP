from flask import Flask, render_template,request,url_for
from flask_bootstrap import Bootstrap 
from tensorflow import keras
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from underthesea import word_tokenize
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import string  
import re

ROOT_DIR = os.path.abspath(os.curdir)

print(ROOT_DIR)

# Doc du lieu
dataset = pd.read_excel(ROOT_DIR+"/dataset_28600.xlsx")

dataset = shuffle(dataset)


def clean_text(text): 
    text = text.translate(string.punctuation)
    # xóa các ký tự không cần thiết
    text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',str(text))
    text = text.lower().split()
    text = " ".join(text)
    return text
def word_processing(sentence):
    stopwords = pd.read_csv(ROOT_DIR+"/vietnamese-stopwords.txt")
    sentence = [word for word in word_tokenize(sentence.lower(), format="text").split() if word not in stopwords]
    return [word for word in sentence if word != ""]

def word2id_processing(input_sentences):
    word2id = dict()
    max_words = 0
    for sentence in input_sentences:
        for word in sentence:
        
            if word not in word2id:
                word2id[word] = len(word2id)
    
        if len(sentence) > max_words:
            max_words = len(sentence)


    id2word = {v: k for k, v in word2id.items()}
    return word2id,max_words, id2word

dataset_clean = dataset['Comment'].map(lambda x: clean_text(x))
input_sentences = [word_processing(str(sentence)) for sentence in dataset_clean.values.tolist()]
word2id,max_words, id2word = word2id_processing(input_sentences)

className ={
        1:'phục vụ tệ',
        2:'món ăn tệ',
        3:'không hợp vệ sinh',
        4:'hợp vệ sinh',
        5:'phục vụ tốt',
        6:'món ăn ngon',
        7:'khác'
      }

app = Flask(__name__,static_url_path = "/static", static_folder = "static")
Bootstrap(app)
model = keras.models.load_model(ROOT_DIR+'/my_model.h5')

@app.route('/',methods = ["GET","POST"])
def index():
    if request.method =="GET":
        return render_template('TrangChu.html')
    else:
        text = request.form.get("text")
       
        tokenized_sample = word_processing(text)
        encoded_samples = [[word2id[word.lower()] for word in tokenized_sample]]
        encoded_samples = keras.preprocessing.sequence.pad_sequences(encoded_samples, maxlen=max_words)
        label = model.predict(np.array(encoded_samples))
        pre = np.argmax(label)
        return render_template("pred.html",text = className[pre])

@app.route('/<string:text>',methods = ["GET","POST"])
def pred(text):
	return text


if __name__ == '__main__':
	app.debug=True
	app.run()