import streamlit as st
import huggingface_hub
import os
import time
import io
from PIL import Image, ImageDraw
import numpy as np
import cv2
import requests
import json
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path
from tqdm import tqdm

from backend.image_generation import DMD2_T2IAdapt
from backend.SAM import SAM, SAM_custom

# ------- CONFIGURATION -------
config = """
      <View>
        <View style="padding: 25px; box-shadow: 2px 2px 8px #AAA;">
          <Image name="img" value="$image" width="100%" maxWidth="100%" brightnessControl="true" contrastControl="true" zoomControl="true" rotateControl="true"></Image>
        </View>
        <RectangleLabels name="tag" toName="img">
          <Label value="Hello"></Label>
          <Label value="Moon"></Label>
        </RectangleLabels>
      </View>
    """

interfaces = [
  "panel",
  "update",
  "controls",
  "side-column",
  "completions:menu",
  "completions:add-new",
  "completions:delete",
  "predictions:menu",
],

user = {
  'pk': 1,
  'firstName': "James",
  'lastName': "Dean"
},

task = {
  'completions': [],
  'predictions': [],
  'id': 1,
  'data': {
    'image': "https://variety.com/wp-content/uploads/2016/05/pooh.jpg"
  }
}

# ------- HELPER FUNCTIONS -------
@st.cache_resource
def footer():
    ft = """
    <style>
    a:link , a:visited{
    color: #BFBFBF;  /* theme's text color hex code at 75 percent brightness*/
    background-color: transparent;
    text-decoration: none;
    }

    a:hover,  a:active {
    color: #0283C3; /* theme's primary color*/
    background-color: transparent;
    text-decoration: underline;
    }

    #page-container {
    position: relative;
    min-height: 10vh;
    }

    footer{
        visibility:hidden;
    }

    .footer {
    position: relative;
    left: 0;
    top:230px;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: #808080; /* theme's text color hex code at 50 percent brightness*/
    text-align: left; /* you can replace 'left' with 'center' or 'right' if you want*/
    }
    </style>

    <div id="page-container">

    <div class="footer">
    </div>

    </div>
    """
    return ft

# ----------------- SESSION STATES ---------------------
if "mask_canvas" not in st.session_state:
    st.session_state["mask_canvas"] = None

if "segmap_canvas" not in st.session_state:
    st.session_state["segmap_canvas"] = None

# -------------- LOADING FUNCTIONS ---------------------

@st.cache_resource
def load_dmd2_pipeline():
    pipe = DMD2_T2IAdapt()
    return pipe

@st.cache_resource
def load_custom_SAM_pipeline():
    ckpt_root = "backend\\ckpts\\segmentation\\"
    labels_definiton_file_path = "D:\\UI\\Streamlit\\AnimatedDrawingsPlusPlusTool\\backend\\repos\\segment_anything\\label_definitions\\sam_drawings_full.json"
    original_ckpt_path = os.path.join(ckpt_root, "sam_vit_b_01ec64.pth")
    finetuned_ckpt_path = os.path.join(ckpt_root, "model_best.pth")
    pipe = SAM_custom(original_ckpt_path=original_ckpt_path, custom_ckpt_path=finetuned_ckpt_path, num_classes=27, label_definitions_path=labels_definiton_file_path)
    return pipe


@st.cache_resource
def load_SAM_pipeline():
    pipe = SAM()
    return pipe
# ------------------------------------------------------------

# function for sketch-guided image generation
def imagine():
    pipe = load_dmd2_pipeline()

    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:",
        ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == "point":
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    # bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        # background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        width=512,
        height=512,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="imagine",
    )

    # Do something interesting with the image data and paths
    if st.button('Imagine!'):
        if canvas_result.image_data is not None:
            canvas_image = canvas_result.image_data
            with st.spinner(text='Imagining...'):
                gen_img_vis = pipe.generate_from_canny(canvas_image,"a drawing, colorful, line-art,m sketch")
                print("Image generated!")
                st.image(gen_img_vis)
            st.markdown("Congratulations! Your character is ready.")
            st.balloons()


# ------------  HELPER FUNCTIONS ------------
def name_to_id(name):
    mapping = {"All":0, "Face":6, "Hair":20}
    return mapping[name]

def filter_class(labels_np, class_id):
    if class_id>0:
        binmask = labels_np==class_id
        labels_np[binmask] = class_id
        labels_np[~binmask] = 0
    return labels_np
# ---------- HELPER FUNCTIONS END ----------

# function for segmentation using SAM
def segment_drawings():
    pipe = load_custom_SAM_pipeline()
    pipe_original_sam = load_SAM_pipeline()
    RANSAC_STEPS = 1
    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:",
        ("rect", "point", "polygon", "freedraw", "line", "circle", "transform"),
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == "point":
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    
    class_name = st.sidebar.selectbox("Class", ["All", "Face", "Hair"])
    class_id = name_to_id(class_name)
    
    stroke_color = "#000000"
    # bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

    bg_color = "#eee"
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg", "jpeg"])
    num_of_refine_pts = st.sidebar.slider("Number of refine points:", 0, 100, 5)
    darken_background = st.sidebar.checkbox("Darken background", True)
    st.sidebar.markdown("---")
    use_gt_bbox = False
    use_gt_mask = False
    use_gt_info = st.sidebar.checkbox("Use Ground Truth information:", value=False)
    if use_gt_info: 
        gt_options = st.sidebar.radio("", ["Bounding Box","Segmentation Map"])
        if gt_options == "Bounding Box":
            use_gt_bbox = True
            use_gt_mask = False
        else:
            use_gt_bbox = False
            use_gt_mask = True
            gt_mask = st.sidebar.file_uploader("GT Mask:", type=["png"])
    
    st.sidebar.markdown("---")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    if bg_image is not None:
        # Open image and create a canvas component
        input_image = Image.open(bg_image).convert("RGB").resize((1024,1024))            
        canvas_image = input_image.resize((512,512))
        w_inp,h_inp = input_image.size
        w_canv,h_canv = canvas_image.size
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=canvas_image if bg_image else None,
            update_streamlit=realtime_update,
            width=w_canv,
            height=w_canv,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == "point" else 0,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
            key="segment_drawings",
        )

        # Do something interesting with the image data and paths
        if st.button('Segment'):
            bbox_from_canvas = False
            if bg_image is not None:
                # extract bbox information from canvas
                bbox_info = [[int(point['left']),int(point['top']),int(point['width']),int(point['height'])] for point in canvas_result.json_data['objects'] if point['type']=='rect']
                if len(bbox_info)==0:
                    bbox_info = [0,0,w_inp-1,h_inp-1]
                else:
                    # latest bbox is the one we want
                    bbox_info = bbox_info[-1]
                    bbox_from_canvas = True

                x,y,w_box,h_box = bbox_info
                bbox_coords = np.array([x, y, x + w_box, y + h_box])
                if bbox_from_canvas:
                    bbox_coords = bbox_coords*(w_inp/w_canv) # scale the bbox taken from canvas to match the input image size
                print("Latest BBOX:", bbox_coords)
                with st.spinner(text='Segmentation in progress...'):
                    num_of_refine_pts = int(num_of_refine_pts)
                    input_image_np = np.array(input_image)
                    seg_out = None
                    if use_gt_bbox:
                        gt_bbox = [[np.array(bbox_coords).astype('float').tolist()]]
                        bbox_mask = pipe_original_sam.segment_with_bbox(input_image=input_image_np, input_bbox=gt_bbox)[0]
                        bbox_mask_np = bbox_mask.squeeze(0).numpy()
                        bbox_mask_np = np.transpose(bbox_mask_np, (1, 2, 0))[:,:,0]
                        seg_out = bbox_mask_np.astype(np.uint8)
                        seg_out[bbox_mask_np] = 26
                    elif use_gt_mask:
                        if gt_mask is not None:
                            gt_labels = Image.open(gt_mask).convert("RGB").resize((w_inp,h_inp))
                            gt_labels_np = np.array(gt_labels)[:,:,0]
                            seg_out = filter_class(gt_labels_np, class_id)
                        else:
                            st.session_state["segmap_canvas"] = None
                            st.warning("Please upload the GT Segmentation Map!")
                            
                    else:
                        labels, sem_mask = pipe.segment_with_bbox(input_image=input_image_np, input_bbox=bbox_coords)
                        labels_np = np.array(labels)
                        seg_out = labels_np
                        seg_out = filter_class(seg_out, class_id)
                        # refinement
                        if int(num_of_refine_pts)>0 and class_id>0:
                            refined_mask_ransac = np.ones_like(labels_np==class_id).astype(bool)
                            for _ in tqdm(range(RANSAC_STEPS)):
                                refine_pts = np.argwhere(labels_np==class_id)
                                refine_pts = np.flip(refine_pts, axis=1)
                                np.random.shuffle(refine_pts)
                                refine_pts = refine_pts[:int(num_of_refine_pts)]
                                refine_pts = refine_pts.tolist()
                                refined_mask_tensor = pipe_original_sam.segment_with_points(input_image, [refine_pts])[0]
                                refined_mask = refined_mask_tensor.squeeze(0).numpy()
                                refined_mask = np.transpose(refined_mask, (1, 2, 0))[:,:,0]
                                refined_mask_ransac  = refined_mask_ransac & refined_mask

                            seg_out = labels_np
                            seg_out[refined_mask_ransac] = class_id
                            seg_out[~refined_mask_ransac] = 0

                    if seg_out is not None:
                        composite_mask = seg_out>0
                        composite_factor = 0.8
                        input_image_np = np.array(input_image)
                        seg_out = pipe.labels_to_colors(seg_out).astype(np.uint8)
                        if darken_background:
                            input_image_np = input_image_np*(1-composite_factor) + seg_out*composite_factor
                        else:
                            input_image_np[composite_mask] = input_image_np[composite_mask]*(1-composite_factor) + seg_out[composite_mask]*composite_factor
                        input_image_np = input_image_np.astype(np.uint8)
                        result = Image.fromarray(input_image_np).convert("RGB").resize((w_canv,w_canv))
                        st.session_state["segmap_canvas"] = result

    if st.session_state["segmap_canvas"] is not None:
        canvas_mask = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=st.session_state["segmap_canvas"],
                update_streamlit=True,
                width=st.session_state["segmap_canvas"].size[0],
                height=st.session_state["segmap_canvas"].size[1],
                drawing_mode=drawing_mode,
                point_display_radius=point_display_radius if drawing_mode == "point" else 0,
                key="segmask_canvas",
            )



# function for segmentation using SAM
def segment():
    pipe = load_SAM_pipeline()

    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:",
        ("point", "rect", "polygon", "freedraw", "line", "circle", "transform"),
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == "point":
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg", "jpeg"])
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    if bg_image is not None:
        # Open image and create a canvas component
        input_image = Image.open(bg_image).convert("RGB").resize((512,512))
        w,h = input_image.size
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=input_image if bg_image else None,
            update_streamlit=realtime_update,
            width=w,
            height=h,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == "point" else 0,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
            key="segment",
        )

        # Do something interesting with the image data and paths
        if st.button('Segment'):
            if bg_image is not None:
                with st.spinner(text='Segmentation in progress...'):
                    circle_pts = [[int(point['left']),int(point['top'])] for point in canvas_result.json_data['objects'] if point['type']=='circle']
                    if len(circle_pts)==0:
                        circle_pts = [[w//2,h//2]]
                    print(circle_pts)
                    masks = pipe.segment_with_points(input_image, [circle_pts])
                    print("Masks returned!")
                    for i in range(len(masks)):
                        mask = masks[i].squeeze(0)
                        mask = mask.cpu().numpy()
                        mask = np.transpose(mask, (1, 2, 0))
                        mask = mask[:,:,0].astype(np.uint8)*255
                        mask = Image.fromarray(mask).convert("RGB")
                        st.markdown("Segmentation is ready!")
                        blended = Image.blend(input_image, mask, 0.5).resize((512,512))
                        st.session_state["mask_canvas"] = blended
                st.balloons()

    if st.session_state["mask_canvas"] is not None:
        canvas_mask = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=st.session_state["mask_canvas"],
                update_streamlit=True,
                width=st.session_state["mask_canvas"].size[0],
                height=st.session_state["mask_canvas"].size[1],
                drawing_mode=drawing_mode,
                point_display_radius=point_display_radius if drawing_mode == "point" else 0,
                key="segmask_canvas",
            )
        # replace = st.button("Fill-in!")
        # if replace:
        #     pipe_fill = load_dmd2_pipeline()
        #     with st.spinner(text='Filling in...'):
        #         gen_img_vis = pipe_fill.generate_from_canny(input_image,"a drawing, colorful, line-art, sketch")
        #         print("Image generated!")
        #         im1 = np.array(input_image)
        #         im2 = np.array(gen_img_vis)
        #         im1[mask] = im2[mask]
        #         final = Image.fromarray(im1).resize((512,512))
        #         st.image(final)
        #         replace = False
        #     st.balloons()

# ------------------------------------------------------

def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "Segment Drawings": segment_drawings,
        "SAM": segment,
        "Imagine": imagine,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()



# STREAMLIT LAYOUT
if __name__ == "__main__":
    # set page config
    st.set_page_config(page_title='AD++', page_icon='art')
    

    with st.sidebar:
        _, icon_col, _= st.columns(3)
        with icon_col:
            st.image('./static/icon.png')
        # main title
        st.markdown("<h1 style='text-align: center; color: black; font-family: 'Monospace';'>Animated Drawings ++</h1>", unsafe_allow_html=True)

    main()

    # ----------------- footer -----------------
    st.write(footer(), unsafe_allow_html=True)
