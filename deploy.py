#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *
from fastai import *
from fastai.vision.widgets import * 


# In[2]:


path = Path()
learn_inf = load_learner(path/'export.pkl', cpu=True)

btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()


# In[3]:


def on_data_change(change):
    lbl_pred.value = ''
    
    #Load Image
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl:
        display(img.to_thumb(244, 244))
        
    #Predict image
    pred, pred_idx, probs = learn_inf.predict(img)
    
    lbl_pred.value = f'Prediction: {pred}; Probabilty: {probs[pred_idx]:0.4f}'


# In[4]:


btn_upload.observe(on_data_change, names=['data'])

#Display Predcictions
display(VBox([widgets.Label('Select photo'), btn_upload, out_pl, lbl_pred]))


# In[ ]:




