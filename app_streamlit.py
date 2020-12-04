import streamlit as st
st.set_page_config(
            page_title="GreenEye", # => Quick reference - Streamlit
            page_icon=":leaves:",
            #layout="centered", # wide
            initial_sidebar_state="auto") # collapsed
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import numpy as np
import base64
import cv2
import time
###################################################################################
###################################################################################
#                               FUNCTIONS

@st.cache

def load_image(path):
    with open(path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    return encoded

def background_image_style(path):
    encoded = load_image(path)
    style = f'''
    <style>
    #hero {{
        background-image: url("data:image/png;base64,{encoded}");
    }}
    </style>
    '''
    return style

image_path = "greeneye/streamlit_images/background.jpg"

st.write(f'<div id="hero"><h1>GreenEye</h1><h2>Helping fight deforestation with deep learning</h2></div>', unsafe_allow_html=True)

def fbeta():
    return 1
def f2():
    return 1
def fbeta_round():
    return 1
def load():
    model = load_model('GreenEyeResNet50V1/',
                   custom_objects={'fbeta': fbeta, 'f2':f2, 'fbeta_round': fbeta_round})
    return model

#model = load()

def generate_imagedata(image):
    ''' GENERATE THE DATA FROM THE LOADED IMAGE '''
    size = (128,128)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(128,128),    interpolation=cv2.INTER_CUBIC))/255.

    return  img_resize[np.newaxis,...]

def import_and_predict(image, model):
    '''MAKE THE PREDICTION '''
    return model.predict(image)


#REMEMBER TO DELETE CLEAR WHEN YOU UPDATE YOUR MODE
#''' PRINT THE CODE'''
def decoder(prediction):
    l=[]
    alltags = [  'agriculture', 'artisinal mine', 'bare ground', 'blooming',
'blow down', 'cloudy', 'conventional mine','cultivation', 'habitation', 'partly cloudy', 'primary rainforest', 'road', 'selective logging',
'slash burn','river']
    for i in range(prediction.shape[1]):  #change this later and remove the clear function
        if prediction[0,i] > 0.4:
            l.append(alltags[i])
    if 'cloudy' in l:
        l.remove('cloudy')
    #if 'partly cloudy' in l:
    #    l.remove('partly cloudy')
    classes = ", ".join(l)

    return classes


#                       PREPROCESS EVERYTHING

#image1 = Image.open('greeneye/streamlit_images/image_1.jpg')
#image2 = Image.open('greeneye/streamlit_images/image_2.jpg')
#image3 = Image.open('greeneye/streamlit_images/image_3.jpg')

#prediction1 = decoder(import_and_predict(generate_imagedata(image1),model))
#prediction2 = decoder(import_and_predict(generate_imagedata(image2),model))
#prediction3 = decoder(import_and_predict(generate_imagedata(image3),model))


######################################################################################################################################################################
######################################################################################################################################################################
#                               Sidebar

st.sidebar.markdown(f"""
    # Examples!
    """)
##################################

st.sidebar.markdown(""" ### Example number 1""")
imagepath = 'greeneye/streamlit_images/train_1.jpg'
st.sidebar.image(imagepath,use_column_width=False,output_format = 'JPEG')

st.sidebar.markdown('''
**Features:**
*primary rainforest, river*
''')

##################################

st.sidebar.markdown(""" ### Example number 2 """)
imagepath = 'greeneye/streamlit_images/train_2.jpg'
st.sidebar.image(imagepath,use_column_width=False,output_format = 'JPEG')

st.sidebar.markdown('''
**Features:**
*habitation, primary rainforest, road*
''')


###################################################################################
###################################################################################
#                               INTRO

st.write(background_image_style(image_path), unsafe_allow_html=True)

CSS = """
.block-container h1 {
    color: #FFFFFF;
    font-family: 'Alfa Slab One';
}
#hero {
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
    position: relative;
    height: 400px;
    text-align: center;
    color: white;
    margin-top: -100px;
    margin-left: -480px;
    margin-right: -480px;
}

#hero h1 {
    padding-top: 160px;
}

#hero h2 {
    padding-top: 8px;
}

body {
    background-color: F4F3EE;
}

.sidebar-content h1 {
    text-align: center;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #000000
}

button {
    color: white !important;
    border: 3px solid white;
}


"""



st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.markdown("""# GREEN EYE
## *Helping fight deforestation with deep learning*""")



###################################################################################
###################################################################################
#                               lOAD THE IMAGE AND GET THE RESULTS

st.markdown('### please select a satellite image for testing')

direction = st.radio('Please select an image to test!', ('No Image','image 1', 'image 2', 'image 3'))

st.write(direction)



if direction == 'No Image':
    pass

elif direction == 'image 1':

    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text('Loading and Analizing')
        bar.progress(i + 1)
        time.sleep(0.1)

    st.image(image1,use_column_width=False,output_format = 'JPEG')
    st.write(f'### Features:\n  *{prediction1}*')

############################################################
elif direction == 'image 2':



    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text('Loading and Analizing')
        bar.progress(i + 1)
        time.sleep(0.1)

    st.image(image2,use_column_width=False,output_format = 'JPEG')
    st.write(f'### Features:\n  *{prediction2}*')

############################################################
elif direction == 'image 3':

    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text('Loading and Analizing')
        bar.progress(i + 1)
        time.sleep(0.1)

    st.image(image3,use_column_width=False,output_format = 'JPEG')

    st.write(f'### Features:\n  *{prediction3}*')

############################################################



###############################################################################################################
###############################################################################################################

st.markdown('''## Or maybe upload a satellite image!''')

file = st.file_uploader('',type = 'jpg',channels = 'RGB',)

if file is None:
    st.text("")
else:
    image = Image.open(file)
    st.image(image,use_column_width=False,output_format = 'JPEG')

    prediction = import_and_predict(generate_imagedata(image),model)
    #st.write(prediction)
    #st.write(type(prediction))

    st.write(f'### Features:\n  *{decoder(prediction)}*')






