import streamlit as st
import cv2
import numpy as np
from fastai.vision.all import *
import pathlib


EX_PATH = 'example_images'

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def predict_breed(image):
    pred, pred_idx, probs = learn.predict(image)
    pred_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}

    return pred_dict

# Loading
learn = load_learner('breed_model.pkl')
labels = learn.dls.vocab

input_file = None
img = None

# App
st.title('Pet Classifier!')
st.subheader('Neural Net Classifier model for identifying pet breeds:')
st.text('Architecture: Resnet50\n'
        'Training Dataset: Oxford Pets, source - https://docs.fast.ai/tutorial.pets.html\n'
        'With Reference to - https://www.tanishq.ai/blog/posts/2021-11-16-gradio-huggingface.html')

with st.sidebar:
    st.header('Choose File Selection Option')
    st.markdown(' - local: from pre-provided examples')
    st.markdown(' - upload: provide your own image file')
    selection_option = st.radio('File Select Option:', ['local', 'upload'])

st.header('Select File:')
# Upload/select file
if selection_option == 'local':
    im_files = [file for file in os.listdir(EX_PATH)
                if os.path.splitext(file)[-1] in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]
    input_file = os.path.join(EX_PATH, st.selectbox('Choose Image:', im_files))
    img = Image.open(input_file)

elif selection_option == 'upload':
    file_loader = st.file_uploader('Upload Pet Image :)')
    try:
        if file_loader is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(file_loader.read()), dtype=np.uint8)
            cv_img = cv2.imdecode(file_bytes, 1)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

            img = Image.fromarray(cv_img)
            input_file = img
    except Exception as e:
        st.write('Image upload failed for selected image, please try again')

# display image  
if img is not None:
    # resize
    base_width= 300
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)
    
    # centre
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image(img)

st.header('Predict Breed')
st.write('Show the top 3 predictions, along with their probability:')
# run prediction
run_predict = st.button('Predict')
if run_predict and input_file is not None:
    
    with st.spinner('Predicting...'):
        predictions = predict_breed(input_file)

    pred_df = pd.DataFrame(
                            predictions.items(), columns=['Species', 'probability %']
                           ).sort_values('probability %', ascending=False)

    # select top 3, clean output etc
    output_df = pred_df.head(3).reset_index().drop(columns=['index'])
    output_df['probability %'] = round(100*output_df['probability %'], 1)
    output_df['Species'] = output_df['Species'].apply(lambda x: x.replace('_', ' ').title())

    st.write(output_df)
