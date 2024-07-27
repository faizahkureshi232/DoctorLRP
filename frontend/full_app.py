import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import copy
import pandas as pd
from PIL import Image
import io
from lrp_alzh import LRPVisualizer
from lrp_BT import BT_LRPVisualizer
from objdetector_ALZH import AL_ObjectDetector
from objdetector_BT import BT_ObjectDetector
from ALZH_model import ALZH_VGGModel
from BT_tumor import BT_VGGModel

device = torch.device("cuda:0" ) #if torch.cuda.is_available() else "cpu")
st.title('Medical Image Analysis')

# Dropdown for model selection
model_choice = st.selectbox('Choose the scan type', ('MRI (Brain Tumor)', 'CT (Alzheimer)'))

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    st.image(image1, caption='Uploaded Image', use_column_width=True)

    transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.ToTensor(),
    ])

    # Convert image to tensor
    image_tensor = transform(image1)
    image_tensor=image_tensor.unsqueeze(0).to(device)  # Define this function as per your model's requirements
    



    if model_choice == 'MRI (Brain Tumor)':
        st.write("Using Brain Tumor Model")
        model_bt=BT_VGGModel()
        model_path = "/home/frontend/best_model_BT.pth"
        model_bt.load_state_dict(torch.load(model_path, map_location=device))
        model_bt.to(device)
        model_bt.eval()
        prediction = model_bt(image_tensor).max(1).indices.detach().cpu().numpy()
        dict_bt={0:'g', 1:'m', 2:'nt',3:'pt'}
        st.write(f"Predicted Label: {dict_bt[prediction[0]]}")

        lrp_visualizer = BT_LRPVisualizer(model_bt, device)
        output_path = lrp_visualizer.save_and_display_image(image_tensor)
    
        st.image(output_path, caption="Processed Image with Relevances", use_column_width=True)

        detector = BT_ObjectDetector()
        output_pathforalzh=detector.process_image(image1)

        st.image(output_pathforalzh, caption="Processed Image with Bounding Box", use_column_width=True)
        





    elif model_choice == 'CT (Alzheimer)':
        model_alzh=ALZH_VGGModel()
        model_path = "/home/Alzhimers/best_model_alzhimers.pth"
        model_alzh.load_state_dict(torch.load(model_path, map_location=device))
        model_alzh.to(device)
        model_alzh.eval()
        prediction = model_alzh(image_tensor).max(1).indices.detach().cpu().numpy()
        dict_alzh={0:'Dementia', 1:'No Dementia'}
        st.write(f"Predicted Label: {dict_alzh[prediction[0]]}")

        lrp_visualizer = LRPVisualizer(model_alzh, device)
        output_path = lrp_visualizer.save_and_display_image(image_tensor)
    
        st.image(output_path, caption="Processed Image with Relevances", use_column_width=True)

        detector = AL_ObjectDetector()
        output_pathforalzh=detector.process_image(image1)

        st.image(output_pathforalzh, caption="Processed Image with Bounding Box", use_column_width=True)
        



