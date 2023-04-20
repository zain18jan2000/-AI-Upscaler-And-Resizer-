import math
import time
import streamlit as st
from PIL import Image
import cv2
import os
import urllib.request

hide_streamlit_style = """
<style>
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;
    padding-left: 1%;
    }
</style>

"""

st.set_page_config(layout="wide")

if os.path.isfile('models/ESPCN_x2.pb'):
    print(True)
else:
    print("Downloading model x2 ...")
    urllib.request.urlretrieve("https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb",
                               "models/ESPCN_x2.pb")
    print("Done!")

    print("Downloading model x3 ...")
    url2 = "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x3.pb"
    urllib.request.urlretrieve(url2, "models/ESPCN_x3.pb")
    print("Done!")

    print("Downloading model x4 ...")
    url3 = "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb"
    urllib.request.urlretrieve(url3, "models/ESPCN_x4.pb")
    print("Done!")


@st.cache_resource
def loadModel(n):
    super_res = cv2.dnn_superres.DnnSuperResImpl_create()
    super_res.readModel('models/ESPCN_x'+n+'.pb')
    return super_res

# on removing (show_spinner=False), it will show that fuction is running on web app
@st.cache_data(show_spinner=False)
def upscale(file,task,_progressBar = None):
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())
        print('No file found, so added in list files')
    if isinstance(task,str):
        super_res = loadModel(task)
        super_res.setModel('espcn', int(task))
        if file.type.split('/')[0] == 'image':
            img = cv2.imread(file.name)
            upscaled_image = super_res.upsample(img)
            print('I upscaled upto',task,'times')
            cv2.imwrite("processed_"+file.name,upscaled_image)
            with st.sidebar:
                st.success('Done!', icon="✅")
            return True
        elif file.type.split('/')[0] == 'video':
            cap = cv2.VideoCapture(file.name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(3))
            height = int(cap.get(4))
            writer = cv2.VideoWriter("processed_"+file.name,fourcc,fps,(width*int(task),height*int(task)))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step_size = 1.0/length
            progress = 0

            with st.sidebar:
                st.info("Operation in progress. Please wait.",icon="ℹ️")
                my_bar = st.progress(0,text="")

            for percent_complete in range(length):
                frame = cap.read()[1]
                frame = super_res.upsample(frame)
                writer.write(frame)

                progress += step_size
                my_bar.progress(progress-0.000000001)
            with st.sidebar:
                st.success('Done! Thankyou for your patience', icon="✅")
            return True
        return True

    # Second case where custom size is required
    else:
        req_width,req_height = int(task[0]),int(task[1])
        if file.type.split('/')[0] == 'image':
            img = cv2.imread(file.name)
            actual_width,actual_height = img.shape[1],img.shape[0]
            w_ratio,h_ratio = req_width/actual_width , req_height/actual_height
            if min([w_ratio,h_ratio]) <= 1.0:
                img = cv2.resize(img,(req_width,req_height))
                print("I did resizing only!")
                cv2.imwrite("processed_" + file.name, img)
                with st.sidebar:
                    st.success('Done!', icon="✅")
                return True
            # rounding off the ratios
            w_ratio,h_ratio = math.ceil(w_ratio),math.ceil(h_ratio)
            # find bigger number
            upscale_number = max(w_ratio,h_ratio)

            # task can be greater than 4 but we can upscale upto 4. So setting task to 4.
            if upscale_number >= 4:
                upscale_number = 4

            super_res = loadModel(str(upscale_number))
            super_res.setModel('espcn', int(upscale_number))
            upscaled_image = super_res.upsample(img)
            print("Before resizing ",(upscaled_image.shape[1], upscaled_image.shape[0]))
            upscaled_image = cv2.resize(upscaled_image,(task[0],task[1]))
            print("Final size got: ",(upscaled_image.shape[1],upscaled_image.shape[0]))

            print("I upscale upto", upscale_number , "times and then resize it.")

            cv2.imwrite("processed_" + file.name, upscaled_image)
            with st.sidebar:
                st.success('Done!', icon="✅")
            return True

        # If file is video
        elif file.type.split('/')[0] == 'video':
            cap = cv2.VideoCapture(file.name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(3))
            height = int(cap.get(4))
            if height > 2160 or width > 3840:
                with st.sidebar:
                    st.success("Sorry, I can't processed Video with resolution above 4k. Please select custom size option.!", icon="ℹ️")
            writer = cv2.VideoWriter("processed_" + file.name, fourcc, fps, (int(task[0]),int(task[1])))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step_size = 1.0 / length
            progress = 0
            progress_text = "Operation in progress. Please wait."
            with st.sidebar:
                st.info("Operation in progress. Please wait.", icon="ℹ️")
                my_bar = st.progress(0)
            for percent_complete in range(length):
                frame = cap.read()[1]
                frame = cv2.resize(frame, (task[0], task[1]))
                writer.write(frame)
                progress += step_size
                my_bar.progress(progress-0.000000001)
            with st.sidebar:
                st.success('Done! Thankyou for your patience', icon="✅")
            return True
        return "It's second"


if 'disable_opt2' not in st.session_state:
    st.session_state.disable_opt2 = True
if 'disable_opt1' not in st.session_state:
    st.session_state.disable_opt1 = False
if 'disable_download' not in st.session_state:
    st.session_state.disable_download = True
if 'disable_proceed' not in st.session_state:
    st.session_state.disable_proceed = False
if 'delete_all' not in st.session_state:
    st.session_state.delete_all = True
    media_files = os.listdir()
    for i in media_files:
        print('Deleting all previous files')
        if '.mp4' in i or '.mov' in i or '.jpg' in i or '.png' in i or 'jpe' in i or '.jpeg' in i or 'pgm' in i:
            os.remove(i)


st.markdown(hide_streamlit_style, unsafe_allow_html=True)

col1,_,col2 = st.columns([6,1,3],gap="small")

def toggle_state_opt1():

    if st.session_state.get("opt1") == True:
        st.session_state.opt2 = False
        st.session_state.disable_opt2 = True

    else:
        st.session_state.opt2 = True
        st.session_state.disable_opt2 = False

def toggle_state_opt2():
    if st.session_state.get("opt2") == True:
        st.session_state.opt1 = False
        st.session_state.disable_opt1 = True
    else:
        st.session_state.opt1 = True
        st.session_state.disable_opt1 = False

# Update the states based on user selection before drawing the widgets in the web page
toggle_state_opt2()
toggle_state_opt1()
options = ["2", "3","4"]
progressBar = None

with col1:
    file = st.file_uploader(" ",type=['png','jpeg','jpg','pgm','jpe','mp4','mov'])
    if file is not None:
        # writing file and saving its details in dict for further processing
        bytes_data = file.getvalue()
        file_size = len(bytes_data)
        print("File size: ",file_size)

        if file.type.split('/')[0] == "image" and file_size > 1650000:
            st.session_state.disable_proceed = True
            with st.sidebar:
                st.info('Sorry, maximum size of image is 1.6MB', icon="ℹ️")
        elif file.type.split('/')[0] == "image":
            image = Image.open(file)
            st.session_state.disable_proceed = False
            st.image(image,caption="Upload Image", use_column_width=True)
            st.session_state.disable_proceed = False
        elif file.type.split('/')[0] == 'video' and file_size > 250000000:
            with st.sidebar:
                options = ["2", "3"]
                st.info('Sorry, maximum size of video is 250MB', icon="ℹ️")
                st.session_state.disable_proceed = True
        elif file.type.split('/')[0] == 'video':
            video = st.video(file)
            print(type(video))
            options = ["2", "3"]
            st.session_state.disable_proceed = False
            with st.sidebar:
                st.info('For custom size, currently I can processed video without AI.', icon="ℹ️")



with col2:
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")

    st.subheader(" UPSCALE RESOLUTION UP TO")
    st.markdown("\n")
    st.markdown("\n")

    opt1 = st.checkbox("MULTIPLES OF",key="opt1",value=True,on_change=toggle_state_opt1)
    st.selectbox("SELECT", options,key="opt1_selBox",disabled=st.session_state.disable_opt1)

    st.markdown("\n")
    st.markdown("\n")
    opt2 = st.checkbox("CUSTOM SIZE",key="opt2",on_change=toggle_state_opt2)

    col1,col2 = st.columns(2)
    with col1:
        st.number_input("Width", step=1, min_value=150,max_value=3840, value=900, key="width",disabled=st.session_state.disable_opt2)

    with col2:
        st.number_input("Height", step=1, min_value=150,max_value=2160, value=900, key="height",disabled=st.session_state.disable_opt2)

    st.markdown("\n")
    st.markdown("\n")

    _, dcol, _ = st.columns([1,5,1],gap="small")


    with dcol:
        if st.button("PROCEED",disabled=st.session_state.disable_proceed, use_container_width=True) and file is not None:
            if st.session_state.get('opt1') == True:
                task = st.session_state.opt1_selBox
            else:
                task = [st.session_state.width, st.session_state.height]
            print(task)
            st.session_state.disable_download = not upscale(file, task, progressBar)

            # print(resulted_file.shape)

        st.markdown("\n")
        st.markdown("\n")

        if file is None:
            st.session_state.disable_download = True

        if st.session_state.disable_download == True:
            st.button("DOWNLOAD FILE", disabled=True,use_container_width=True)
        else:
            with open('processed_' + file.name, "rb") as download_file:
                st.download_button(label="DOWNLOAD FILE", data=download_file,
                                   file_name='processed_' + file.name, mime="image/png",
                                   disabled=st.session_state.disable_download, use_container_width=True)

st.markdown("\n")
st.markdown("\n")
st.info("DESCRIPTION :    This web app is a free tool designed to upscale or resize image resolution. While the app"+
        " is still undergoing development, we are delighted to offer you to use it for image resolution upscaling."+
        " We welcome your feedback and suggestions, and encourage you to contact us at zain.18j2000@gmail.com "+
        "to share your thoughts. Thank you for your interest in our web application, and we look forward "+
        "to hearing from you as we continue to work towards making this project a resounding success.")

