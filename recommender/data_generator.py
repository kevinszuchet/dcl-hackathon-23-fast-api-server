#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:28:29 2023

@author: nahuelpatino
"""


import snowflake.connector
import pandas as pd
import numpy as np
import copy


snowflake.connector.paramstyle='qmark'

ctx = snowflake.connector.connect(
    user= '' ,
    password= '' ,
    account=  'gra03234.us-east-1' ,
    warehouse= 'EXPLORE',
    role='DATA'
    )


##################################

#Marketplace items with sales since May 2022 (for V2 and higher thumbnail quality):
start_date = '2022-05-12'
end_date = '2023-05-30' 
 
##################################


# Query for marketplace data:
query = """
WITH mps AS (

SELECT 
item_id,
sale_type, 
--count(*) as n_sales,
sum(price_usd) as usd_spent

from "DCL"."MARKETPLACE"."FCT_MARKETPLACE_SALES_WEARABLES"

group by item_id, sale_type    

),

pivot AS(
    
    SELECT * 
  FROM mps
    PIVOT(SUM(usd_spent) FOR sale_type IN ('mint', 'bid', 'order'))
      AS p (item_id, mint_usd, bid_usd, order_usd)
    order by item_id
    ),

pivot2 AS(select
    item_id,
     count(distinct buyer_address) as buyers,
    count(*) as n_sales
    from "DCL"."MARKETPLACE"."FCT_MARKETPLACE_SALES_WEARABLES"
    group by 1    
),

cte AS( 
SELECT *
FROM "DCL"."PROD"."DIM_WEARABLES" AS w

WHERE date_trunc('day', w.created_at)  >=  '{date}' 
    AND date_trunc('day', w.created_at)  <= '{date2}'  
ORDER BY created_at 
),

col AS(
SELECT * FROM "DCL"."PROD"."DIM_COLLECTIONS"
WHERE date_trunc('day', created_at)  >=  '{date}' 
    AND date_trunc('day', created_at)  <= '{date2}'  
       )

SELECT  cte.item_id, cte.collection,cte.name,cte.description, cte.image, cte.category, pivot.mint_usd, pivot.bid_usd, pivot.order_usd, pivot2.buyers, pivot2.n_sales

from cte
inner join col
on cte.collection = col.collection_id
inner join pivot
on cte.item_id = pivot.item_id
inner join pivot2
on cte.item_id = pivot2.item_id

qualify 1=row_number() over(partition by cte.item_id order by cte.collection,cte.name,cte.description, cte.image, cte.category)
"""


query=query.format(date = start_date,date2 = end_date)
cur = ctx.cursor()
cur.execute(query)

w_data = cur.fetch_pandas_all()
w_data = w_data.loc[ w_data.CATEGORY.isin(['upper_body', 'lower_body' ,'feet','hat']),:] # Proof of concept with just these categories 
w_data=w_data.fillna(0) 




#####
##### Generate tags from Visual Question Answering model #####
#####

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import pickle

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large", device="mps")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")
mps_device = torch.device("mps")
model=model.to(mps_device)
    
locs=['what', 'style', 'theme', 'color', 'season', 'occasion','who','porn','nazi']

colmap={}
for i in locs:
    w_data[i] = ''
    colmap[i] = w_data.columns.get_loc(i)


questions = {'lower_body': {'what':"What is this garment?",
                           'style': "What style is this garment?",
                            'theme':"What theme is this garment?",
                            'color':"What color is this garment?"},
             'upper_body': {'what':"What is this garment?",
                                        'style': "What style is this garment?",
                                         'theme':"What theme is this garment?",
                                         'color':"What color is this garment?"},
             'feet': {'what':"What are these shoes?",
                           'style': "What style are these shoes?",
                            'theme':"What theme are these shoes?",
                            'color':"What color are these shoes?"},         
             'hat': {'what':"Is this a hat or a cap?",
                           'style': "What style is this hat?",
                            'theme':"What theme is this hat?",
                            'color':"What color is this hat?"},
             'all':{'season': 'Is this cloting item for summer or winter?',
                    'occasion': 'What occasion would these clothes be good for?',
                    'who':'What kind of person would wear this garment?',
                    'porn':'Is there pornographic or adult content in this picture?',
                    'nazi':'Is there a nazi symbol in this picture?' }
                             }

for i in range(len(w_data)):
    q1=None
    answer=None
    img_url = w_data.iloc[i,4]
    category = w_data.iloc[i,5]
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    print(i)
    for z in locs:
        try:
            q1=questions[category][z]
        except:
            q1=questions['all'][z]
            
        inputs = processor(raw_image, q1, return_tensors="pt")
        inputs=inputs.to(mps_device)
        out = model.generate(**inputs)
        answer=processor.decode(out[0], skip_special_tokens=True, device="mps")
        w_data.iloc[i, colmap[z]] = answer
        print(answer)


 #    if i%100 == 0:
        #w_data.to_pickle('w_data.pkl')

# Save data
#w_data.to_pickle('w_data.pkl')

w_data.insert(0, 'py_ix', w_data.reset_index().index)

w_data.to_csv('Salesforce_blip-vqa-base2.csv')






#####
##### Generate Simmilarity Matrix and co-occurrence matrix #####
#####

##### Co-occurrence matrix
query = """


SELECT DISTINCT
 BUYER_ADDRESS, ITEM_ID

from "DCL"."MARKETPLACE"."FCT_MARKETPLACE_SALES_WEARABLES"
WHERE date_trunc('day', sale_at)  >=  '{date}' 
    AND date_trunc('day', sale_at)  <= '{date2}'  


"""

query=query.format(date = start_date,date2 = end_date)
cur = ctx.cursor()
cur.execute(query)

purchases_df = cur.fetch_pandas_all()



pivot_table = purchases_df.pivot_table(index='BUYER_ADDRESS', columns='ITEM_ID', aggfunc='size', fill_value=0)

pivot_table = pivot_table.T.dot(pivot_table)
for i in range(len(pivot_table)):
    pivot_table.iloc[i,i]=0
pivot_table=pivot_table.reindex(w_data.set_index('ITEM_ID').index, fill_value=0)

pivot_table=pivot_table.reindex(w_data.set_index('ITEM_ID').index, fill_value=0, axis=1)
pivot_table.fillna(0, inplace=True)

ct=pivot_table.values

with open('occurrence_matrix.pkl','wb') as f:
     pickle.dump(ct, f)


##### Simmilarity matrix
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


#Compute embedding for both lists
sep=' '
l1 = w_data['what'] + sep +  w_data['style'] + sep +   w_data['theme']  + sep +   w_data['color']  + sep +   w_data['season'] + sep +   w_data['occasion'] +  w_data['who']
embedding_2 = model.encode( l1.values, convert_to_tensor=True)
embedding_1 = model.encode( l1.values, convert_to_tensor=True)

res=util.pytorch_cos_sim(embedding_1, embedding_2)
res=res.numpy()



with open('simmilarity_matrix.pkl','wb') as f:
     pickle.dump(res, f)




























    
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from transformers import pipeline
mps_device = "mps"
    
    


for i in range(len(w_data)):
    a1=None
    a2=None
    a3=None
    a4=None
    a5=None
    a6=None

    img_url = w_data.iloc[i,3]
    category = w_data.iloc[i,4]
    if (category == 'lower_body') or (category == 'upper_body'):
        q1 = "What is this garment?"
        q2 = "What style is this garment?"
        q3 = "What theme is this garment?"
        q4 = "What color is this garment?"
    elif category == 'feet':
        q1 = "What are these shoes?"
        q2 = "What style are these shoes?"
        q3 = "What theme are these shoes?"
        q4 = "What color are these shoes?"
    elif category == 'hat':
        q1 = "Is this a hat or a cap?"
        q2 = "What style is this hat?"
        q3 = "What theme is this hat?"
        q4 = "What color is this hat?"        
    else:
        pass
    q5= 'Is this cloting item for summer or winter?'
    q6= 'What kind of person would wear this garment?'

    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    inputs = processor(raw_image, q1, return_tensors="pt")
    inputs=inputs.to(device)
    out = model.generate(**inputs)
    a1=processor.decode(out[0], skip_special_tokens=True, device="mps")
    w_data.iloc[i,-6] = a1
    
    inputs = processor(raw_image, q2, return_tensors="pt")
    inputs=inputs.to(device)

    out = model.generate(**inputs)
    a2=processor.decode(out[0], skip_special_tokens=True, device="mps")
    w_data.iloc[i,-5] = a2
    print(a2)
    
    inputs = processor(raw_image, q3, return_tensors="pt")
    inputs=inputs.to(device)

    out = model.generate(**inputs)
    a3=processor.decode(out[0], skip_special_tokens=True, device="mps")
    w_data.iloc[i,-4] = a3    

    inputs = processor(raw_image, q4, return_tensors="pt")
    inputs=inputs.to(device)
    out = model.generate(**inputs)
    a4=processor.decode(out[0], skip_special_tokens=True, device="mps")
    w_data.iloc[i,-3] = a4    

    inputs = processor(raw_image, q5, return_tensors="pt")
    inputs=inputs.to(device)
    out = model.generate(**inputs)
    a5=processor.decode(out[0], skip_special_tokens=True, device="mps")
    w_data.iloc[i,-2] = a5

    inputs = processor(raw_image, q6, return_tensors="pt")
    inputs=inputs.to(device)
    out = model.generate(**inputs)
    a6=processor.decode(out[0], skip_special_tokens=True, device="mps")
    w_data.iloc[i,-1] = a6


for i in 








w_data.to_pickle('all_items_blipvqa.pkl')


labels=pd.read_pickle('all_items_blipvqa.pkl')

labels["RANK"] = labels.groupby("IMAGE")["IMAGE"].rank(method="first", ascending=True)
labels=labels.loc[labels.RANK==1 ,:]

w_data_2=	pd.merge(w_data,labels[['IMAGE','what','style', 'theme', 'color', 'class', 'class2']], on='IMAGE', how='inner')

w_data_2.to_pickle('items_with_sales_blipvqa.pkl')





from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image



processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")



img_url = w_data.iloc[3,3]
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')


# prepare inputs
encoding = processor(raw_image, q1, return_tensors="pt")
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
a1=model.config.id2label[idx]


w_data['what'] = ''
w_data['style'] = ''
w_data['theme'] = ''
w_data['color'] = ''

for i in range(len(w_data)):
    a1=None
    a2=None
    a3=None
    a4=None
    img_url = w_data.iloc[i,3]
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    category = w_data.iloc[i,4]
    if (category == 'lower_body') or (category == 'upper_body'):
        q1 = "What is this garment?"
        q2 = "What style is this garment?"
        q3 = "What theme is this garment?"
        q4 = "What color is this garment?"
    elif category == 'feet':
        q1 = "What are these shoes?"
        q2 = "What style are these shoes?"
        q3 = "What theme are these shoes?"
        q4 = "What color are these shoes?"
    elif category == 'hat':
        q1 = "Is this a hat or a cap?"
        q2 = "What style is this hat?"
        q3 = "What theme is this hat?"
        q4 = "What color is this hat?"        
    else:
        pass


    # prepare inputs
    encoding = processor(raw_image, q1, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    a1=model.config.id2label[idx]
    w_data.iloc[i,-4] = a1    

    # prepare inputs
    encoding = processor(raw_image, q2, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    a2=model.config.id2label[idx]
    w_data.iloc[i,-3] = a2    

    # prepare inputs
    encoding = processor(raw_image, q3, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])
    a3=model.config.id2label[idx]
    w_data.iloc[i,-2] = a3    

    # prepare inputs
    encoding = processor(raw_image, q4, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    a4=model.config.id2label[idx]
    w_data.iloc[i,-1] = a4    











from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from PIL import Image

processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
mps_device = torch.device("mps")
model=model.to(mps_device)


r = requests.get(str(w_data.iloc[1,3])+".png", stream=True)
raw_image = Image.open(io.BytesIO(r.content)) 

pixel_values = processor(images=raw_image, return_tensors="pt").pixel_values

question = "What style is this t-shirt?"

input_ids = processor(text=question, add_special_tokens=False ).input_ids
input_ids = [processor.tokenizer.cls_token_id] + input_ids
input_ids = torch.tensor(input_ids).unsqueeze(0)


generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=15)
print(processor.batch_decode(generated_ids[0], skip_special_tokens=True))



w_data['what'] = ''
w_data['style'] = ''
w_data['theme'] = ''
w_data['color'] = ''

for i in range(len(w_data)):
    a1=None
    a2=None
    a3=None
    a4=None
    img_url = w_data.iloc[i,3]
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    pixel_values = processor(images=raw_image, return_tensors="pt").pixel_values

    category = w_data.iloc[i,4]
    if (category == 'lower_body') or (category == 'upper_body'):
        q1 = "What is this garment?"
        q2 = "What style is this garment?"
        q3 = "What theme is this garment?"
        q4 = "What color is this garment?"
    elif category == 'feet':
        q1 = "What are these shoes?"
        q2 = "What style are these shoes?"
        q3 = "What theme are these shoes?"
        q4 = "What color are these shoes?"
    elif category == 'hat':
        q1 = "Is this a hat or a cap?"
        q2 = "What style is this hat?"
        q3 = "What theme is this hat?"
        q4 = "What color is this hat?"        
    else:
        pass


    # prepare inputs
    input_ids = processor(text=q1, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
    a1=processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(a1)
    w_data.iloc[i,-4] = a1    

    # prepare inputs
    input_ids = processor(text=q2, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
    a2=processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(a2)
    w_data.iloc[i,-3] = a2    

    # prepare inputs
    encoding = processor(raw_image, q3, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])
    a3=model.config.id2label[idx]
    w_data.iloc[i,-2] = a3    

    # prepare inputs
    encoding = processor(raw_image, q4, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    a4=model.config.id2label[idx]
    w_data.iloc[i,-1] = a4    
























import io
import torch
from promptcap import PromptCap_VQA
import transformers
import requests
# QA model support all UnifiedQA variants. e.g. "allenai/unifiedqa-v2-t5-large-1251000"
vqa_model = PromptCap_VQA(promptcap_model="vqascore/promptcap-coco-vqa", qa_model="allenai/unifiedqa-t5-base")

if torch.cuda.is_available():
  vqa_model.cuda()

vqa_model.mps()

img_url = w_data.iloc[0,3]
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

r = requests.get(str(w_data.iloc[0,3])+".png", stream=True)
raw_image = Image.open(io.BytesIO(r.content)) 

print(vqa_model.vqa( "what type of tshirt is this?" ,  io.BytesIO(r.content)))

print(vqa_model.vqa(q2, io.BytesIO(r.content)))
print(vqa_model.vqa(q3, io.BytesIO(r.content)))
print(vqa_model.vqa(q4,  io.BytesIO(r.content)))


from PIL import Image

Image.open(image) 


















































import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from PIL import Image

processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")

file_path = hf_hub_download(repo_id="nielsr/textvqa-sample", filename="bus.png", repo_type="dataset")
image = Image.open(file_path).convert("RGB")

pixel_values = processor(images=image, return_tensors="pt").pixel_values

question = "what does the front of the bus say at the top?"

input_ids = processor(text=question, add_special_tokens=False).input_ids
input_ids = [processor.tokenizer.cls_token_id] + input_ids
input_ids = torch.tensor(input_ids).unsqueeze(0)

generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
print(processor.batch_decode(generated_ids, skip_special_tokens=True))

w_data['style'] = ''
w_data['theme'] = ''
w_data['color'] = ''

for i in range(len(w_data)):

    img_url = w_data.iloc[i,-5]
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    
    question = "What clothes style is this?"
    pixel_values = processor(images=raw_image, return_tensors="pt").pixel_values
        
    input_ids = processor(text=question, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    
    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
    print(processor.batch_decode(generated_ids, skip_special_tokens=True))
            
    w_data.iloc[i,-3] = style
    
    
    question = "What clothes theme is this?"
    inputs = processor(raw_image, question, return_tensors="pt")
    
    out = model.generate(**inputs)
    theme=processor.decode(out[0], skip_special_tokens=True)
        
    w_data.iloc[i,-2] = theme    

    question = "What clothes color is this?"
    inputs = processor(raw_image, question, return_tensors="pt")
    
    out = model.generate(**inputs)
    color=processor.decode(out[0], skip_special_tokens=True)
        
    w_data.iloc[i,-1] = color    














from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cats are there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
w_data['style'] = ''
w_data['theme'] = ''
w_data['color'] = ''

for i in range(len(w_data)):

    url = w_data.iloc[i,-5]
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    
    text = "What clothes theme is this?"

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")
    
    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])    
    
    
    
    w_data.iloc[i,-3] = style
        
        
    question = "What clothes theme is this?"
    inputs = processor(raw_image, question, return_tensors="pt")
    
    out = model.generate(**inputs)
    theme=processor.decode(out[0], skip_special_tokens=True)
        
    w_data.iloc[i,-2] = theme    

    question = "What clothes color is this?"
    inputs = processor(raw_image, question, return_tensors="pt")
    
    out = model.generate(**inputs)
    color=processor.decode(out[0], skip_special_tokens=True)
        
    w_data.iloc[i,-1] = color    


    















    
from open_flamingo import create_model_and_transforms

from transformers import AutoModel, AutoTokenizer
modelw = AutoModel.from_pretrained('learnanything/llama-7b-huggingface')
tokenizer = AutoTokenizer.from_pretrained('learnanything/llama-7b-huggingface')

modelw = AutoModel.from_pretrained('learnanything/llama-7b-huggingface',
                                    load_in_8bit=True,
                                    device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('learnanything/llama-7b-huggingface')


modelw = AutoModel.from_pretrained('EleutherAI/gpt-neo-125M')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')


model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path= 'EleutherAI/gpt-neo-1.3B',#"<path to llama weights in HuggingFace format>"
    tokenizer_path= 'EleutherAI/gpt-neo-1.3B',#"<path to llama tokenizer in HuggingFace format>"
    cross_attn_every_n_layers=4
)

# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch


checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)
    
    
from PIL import Image
import requests

"""
Step 1: Load images
"""
demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)

demo_image_two = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
        stream=True
    ).raw
)

query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
        stream=True
    ).raw
)



demo_image_one = Image.open(
    requests.get(
        "https://peer.decentraland.org/lambdas/collections/contents/urn:decentraland:matic:collections-v2:0xfb1d9d5dbb92f2dccc841bd3085081bb1bbeb04d:16/thumbnail.png", 
        stream=True
    ).raw
)

demo_image_two = Image.open(
    requests.get(
        "https://peer.decentraland.org/lambdas/collections/contents/urn:decentraland:matic:collections-v2:0x1daaee6ff60df09849c050c06db5330c6171b0a7:2/thumbnail.png",
        stream=True
    ).raw
)

query_image = Image.open(
    requests.get(
        "https://peer.decentraland.org/lambdas/collections/contents/urn:decentraland:matic:collections-v2:0x849a55b6f1769e3cd6335eb2fa09e4171e6bfb23:0/thumbnail.png", 
        stream=True
    ).raw
)


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1 
 (this will always be one expect for video which we don't support yet), 
 channels = 3, height = 224, width = 224.
"""
vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>An image of a hat.<|endofchunk|><image>An image of a foot wearable.<|endofchunk|><image>An image of"],
    return_tensors="pt",
)


"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)

print("Generated text: ", tokenizer.decode(generated_text[0]))
    
    
    
    
    
    
    
    
python /Users/nahuelpatino/Downloads/convert_llama_weights_to_hf.py \
    --input_dir /Users/nahuelpatino/Documents/llama7b/llama-7b --model_size 7B --output_dir /Users/nahuelpatino/Documents/llama7b/llama7bhf
    
    
    
















'/Users/nahuelpatino/Desktop/Screenshot 2023-05-05 at 17.05.29.png'


# single image, question answering
AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_image', \
      'image_path': '/Users/nahuelpatino/Desktop/Screenshot 2023-05-05 at 17.05.29.png', \
      'model_name': 'GIT_BASE_VQAv2', \
      'prefix': 'what is it?', \
}"





torch.device()

# -*- coding: utf-8 -*-
import torch
import math

dtype = torch.float
device = torch.device("mps")

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

# Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')





# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

    # Create a Tensor directly on the mps device
    x = torch.ones(5, device=mps_device)
    # Or
    x = torch.ones(5, device="mps")

    # Any operation happens on the GPU
    y = x * 2

    # Move your model to mps just like any other device
    model = YourFavoriteNet()
    model.to(mps_device)

    # Now every call runs on the GPU
    pred = model(x)











import requests
import time


for i in range(len(w_data)):
    url=w_data.iloc[ i , -2 ] 
    name=url[-58:]
    name=name.replace("/", "_" )
    name=name+'.png'
    print(url)
    img_data = requests.get( url).content
    with open(name, 'wb') as handler:
        handler.write(img_data)
        
    time.sleep(0.5)    
        
        




    