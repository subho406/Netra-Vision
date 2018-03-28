# Copyright 2018 Cognibit Solutions LLP
# Author: Subhojeet Pramanik
#
# ===================================================
"""
Demo script that uses the im2txt inference library 

"""


import time
import math
import jsonpickle
from im2txt import run_inference
from im2txt.inference_utils import vocabulary
from flask import Flask, request, Response


im2txt=None
vocab= None
app = Flask(__name__) 


@app.route('/describe_image', methods=['POST'])
def describe_image():
	r=request 
	raw_data=r.data
	captions=im2txt.get_raw_image_caption(raw_data)
	response={'data':[]}
	for i,c in enumerate(captions):
		sentence = [vocab.id_to_word(w) for w in c.sentence[1:-1]]
		sentence=" ".join(sentence)
		r={'sentence':sentence,'prob':math.exp(c.logprob)}
		response['data'].append(r)
		print("  %d) %s (p=%f)" % (i, sentence, math.exp(c.logprob)))
	response_pickled = jsonpickle.encode(response)
	return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__=='__main__':
    checkpoint_path='pretrained/model.ckpt-2000000'
    vocab_file='pretrained/word_counts.txt'
    sample_image='sample_images/Seattle_-_Cheasty_Blvd_street_sign.jpg'
    vocab=vocabulary.Vocabulary(vocab_file)
    im2txt=run_inference.Im2txtInference(vocab_file=vocab_file,checkpoint_path=checkpoint_path)
    app.run(host="0.0.0.0", port=5000)

