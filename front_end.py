import streamlit as st

st.title("Welcome to an Image coverter 🗿")
st.markdown("This webapp convert real human image to Anime or vice versa using cycle GAN model")

with st.sidebar:
    option = st.radio("Select option",
                      ("Human","Anime")
                      )
    upload1 =  st.file_uploader("Select the image file",type=["png","jpg"])
    # upload2 = st.file_uploader("Select the Second image file", type=["png", "jpg"])

if option == "Human":
    st.markdown("Converting Human Image to Anime",)

else:
    st.markdown("Converting Anime Image to Human")

col1, col2 = st.columns(2)
if (upload1 is not None):
    with col1:
        st.image(upload1)
    