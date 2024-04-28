import streamlit as st
import torch
import numpy as np 
from PIL import Image
from torchvision import transforms
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
   

def preprocess_image(input_image):
    test_img = Image.open(input_image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_img = transform(test_img).unsqueeze(0)
    return test_img

def style_transfer_one_img(Gen_A, Gen_B, loadedA):  
    fake_B = Gen_A(loadedA).cpu()
    fake_B = np.transpose(fake_B.squeeze(0).detach().numpy(), (1, 2, 0))  # Transpose dimensions
    st.image(fake_B, clamp=True, channels='RGB')  # Display the image with RGB channels

def style_transfer_one_img_(Gen_A, Gen_B, loadedA):  
    fake_B = Gen_B(loadedA).cpu()
    fake_B = np.transpose(fake_B.squeeze(0).detach().numpy(), (1, 2, 0))  # Transpose dimensions
    st.image(fake_B, clamp=True, channels='RGB')



col1, col2 = st.columns(2)
if (upload1 is not None):
    with col1:
        st.image(upload1)
    with col2:
        if option == "Human":
            st.markdown("Converting Human Image to Anime")
            input_image = preprocess_image(upload1)
            style_transfer_one_img(person_to_anime.eval(), anime_to_person.eval(), input_image)
        else:
            st.markdown("Converting Anime Image to Human")
            # Add conversion logic for Anime to Human
            
            input_image = preprocess_image(upload1)
            style_transfer_one_img(person_to_anime.eval(),anime_to_person.eval(), input_image)

    