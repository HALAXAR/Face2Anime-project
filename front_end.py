import streamlit as st
import torch
import numpy as np 
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize
from utils import *


# Loading the models
person_to_anime=torch.load("person_generator.pth",map_location=torch.device('cpu'))
anime_to_person=torch.load("anime_generator.pth",map_location=torch.device('cpu'))



st.title("Welcome to an Image coverter 🗿")
st.markdown("This webapp convert real human image to Anime or vice versa using cycle GAN model")

with st.sidebar:
    option = st.radio("Select option",
                      ("Human","Anime")
                      )
    upload1 =  st.file_uploader("Select the image file",type=["png","jpg"])
    # upload2 = st.file_uploader("Select the Second image file", type=["png", "jpg"])


col1, col2 = st.columns(2)
if (upload1 is not None):
    with col1:
        st.image(upload1)
    with col2:
        if option == "Human":
            st.markdown("Converting Human Image to Anime")
            image=Image.open(upload1)
            # image=np.array(upload1)
            image=transforms.ToTensor()(image)
            output_image=person_to_anime(image)
            resized_img=Resize((256,256))(output_image)
            img = resized_img / 2 + 0.5
            npimg = img.detach().numpy()
            st.image(np.transpose(npimg,(1,2,0)))

        else:
            st.markdown("Converting Anime Image to Human")
            image=Image.open(upload1)
            # image=np.array(upload1)
            image=transforms.ToTensor()(image)
            output_image=anime_to_person(image)
            resized_img=Resize((256,256))(output_image)
            img = resized_img / 2 + 0.5
            npimg = img.detach().numpy()
            st.image(np.transpose(npimg,(1,2,0)))

    