"""
BRISC2025 Brain Tumor — Streamlit Deployment App
=================================================
Pipeline:
  1) Classify the uploaded MRI scan (EfficientNetB3 classifier).
  2) If the predicted class is a tumor class (not "no_tumor"):
        a) Run the U-Net segmentation model to localize the tumor.
        b) Run Grad-CAM on the classifier to visualize what drove the decision.
     Otherwise: show the classification result only.

Run with:
    streamlit run app.py

Expected files in the same folder (adjust paths in the CONFIG section below):
    - brisc2025_efficientnetb3.keras   (classification model, from Phase 1/2 training)
    - brisc2025_unet_best.keras        (segmentation model)
"""

import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import matplotlib
import gdown

# ----------------------------------------------------------------------------
# CONFIG — edit these to match your trained models / class order
# ----------------------------------------------------------------------------
# Local paths where the models will be cached after downloading once.
CLS_MODEL_PATH = "brisc2025_efficientnetb3.keras"
SEG_MODEL_PATH = "brisc2025_unet_best.keras"

# Google Drive file IDs (from the shareable link, e.g.
# https://drive.google.com/file/d/<THIS_PART>/view -> that's the ID).
# Leave empty ("") if you'd rather keep the models local and skip downloading.
CLS_MODEL_DRIVE_ID = "193EZRN3LffUxjmud_4KgIx3y0Eebr3Wb"
SEG_MODEL_DRIVE_ID = "1IkylXktgWXA3MmWGBExi1s0TSnSGGNWQ"

# Must match the order Keras used when it built `class_names` during training
# (alphabetical order of the subfolders inside TRAIN_DIR). Confirmed via
# print(train_ds_raw.class_names) during training.
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
NO_TUMOR_LABEL = "no_tumor"   # the class name that means "healthy / no tumor"

CLS_IMG_SIZE = 300   # must match training (IMG_SIZE used for the classifier)
SEG_IMG_SIZE = 256   # must match training (IMG_SIZE used for the U-Net)

# Last conv layer inside the EfficientNetB3 backbone used for Grad-CAM.
# "top_conv" is the standard last conv layer name for EfficientNetB3.
GRADCAM_LAYER_NAME = "top_conv"


# ----------------------------------------------------------------------------
# Custom objects needed to reload the segmentation model (same as training)
# ----------------------------------------------------------------------------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1e-6):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_bin, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


CUSTOM_OBJECTS = {
    "bce_dice_loss": bce_dice_loss,
    "dice_coef": dice_coef,
    "iou_metric": iou_metric,
}


# ----------------------------------------------------------------------------
# Download models from Google Drive (only if not already downloaded)
# ----------------------------------------------------------------------------
def ensure_model_downloaded(local_path, drive_id):
    if os.path.exists(local_path):
        return local_path  # already cached on disk, skip download
    if not drive_id:
        raise FileNotFoundError(
            f"'{local_path}' not found locally and no Drive ID was provided. "
            f"Set the corresponding *_DRIVE_ID at the top of app.py."
        )
    url = f"https://drive.google.com/uc?id={drive_id}"
    with st.spinner(f"Downloading {local_path} from Google Drive (first run only)..."):
        gdown.download(url, local_path, quiet=False)
    return local_path


# ----------------------------------------------------------------------------
# Model loading (cached so it only happens once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    ensure_model_downloaded(CLS_MODEL_PATH, CLS_MODEL_DRIVE_ID)
    ensure_model_downloaded(SEG_MODEL_PATH, SEG_MODEL_DRIVE_ID)
    # compile=False: we only need these models for inference, not training,
    # so we skip reloading the optimizer/loss/metrics config entirely.
    # This avoids deserialization errors caused by TF/Keras version drift
    # between the training environment and the deployment environment.
    cls_model = tf.keras.models.load_model(CLS_MODEL_PATH, compile=False)
    seg_model = tf.keras.models.load_model(
        SEG_MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False
    )
    return cls_model, seg_model


# ----------------------------------------------------------------------------
# Preprocessing helpers
# ----------------------------------------------------------------------------
def preprocess_for_classifier(pil_img):
    img = pil_img.convert("RGB").resize((CLS_IMG_SIZE, CLS_IMG_SIZE))
    arr = tf.keras.utils.img_to_array(img)
    # match training: force grayscale look, then EfficientNet preprocessing
    arr = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(arr))
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return tf.expand_dims(arr, axis=0)  # (1, H, W, 3)


def preprocess_for_segmentation(pil_img):
    img = pil_img.convert("RGB").resize((SEG_IMG_SIZE, SEG_IMG_SIZE))
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return tf.expand_dims(arr, axis=0)  # (1, H, W, 3)


# ----------------------------------------------------------------------------
# Classification (with simple horizontal-flip TTA, as in the notebook)
# ----------------------------------------------------------------------------
def classify(cls_model, pil_img):
    batch = preprocess_for_classifier(pil_img)
    probs = cls_model.predict(batch, verbose=0)
    probs_flip = cls_model.predict(tf.image.flip_left_right(batch), verbose=0)
    avg_probs = (probs[0] + probs_flip[0]) / 2.0
    pred_idx = int(np.argmax(avg_probs))
    return CLASS_NAMES[pred_idx], avg_probs


# ----------------------------------------------------------------------------
# Segmentation (with horizontal-flip TTA, as in the notebook)
# ----------------------------------------------------------------------------
def segment(seg_model, pil_img):
    batch = preprocess_for_segmentation(pil_img)
    pred = seg_model(batch, training=False)
    pred_flip = seg_model(tf.image.flip_left_right(batch), training=False)
    pred_flip_unflipped = tf.image.flip_left_right(pred_flip)
    mask = ((pred + pred_flip_unflipped) / 2.0)[0, :, :, 0].numpy()
    return mask  # values in [0,1], shape (SEG_IMG_SIZE, SEG_IMG_SIZE)


def overlay_mask_on_image(pil_img, mask, threshold=0.5, color=(255, 0, 0), alpha=0.4):
    base = pil_img.convert("RGB").resize((SEG_IMG_SIZE, SEG_IMG_SIZE))
    base_arr = np.array(base).astype(np.float32)

    mask_bin = (mask > threshold).astype(np.float32)
    overlay = base_arr.copy()
    for c in range(3):
        overlay[..., c] = base_arr[..., c] * (1 - mask_bin * alpha) + color[c] * (mask_bin * alpha)

    tumor_pct = 100.0 * mask_bin.sum() / mask_bin.size
    return Image.fromarray(overlay.astype("uint8")), tumor_pct


# ----------------------------------------------------------------------------
# Grad-CAM for the classifier
# ----------------------------------------------------------------------------
@st.cache_resource
def build_gradcam_model(_cls_model):
    """Builds the pieces needed for Grad-CAM without constructing a single
    Functional Model that spans from the outer model's input into the
    nested EfficientNet backbone's internal layers (that triggers a
    "Graph disconnected" error in Keras 3 once the model has been
    reloaded from disk). Instead we:
      1) build a small submodel using the backbone's OWN input/output
         (always consistent, since it's a self-contained Functional model),
         producing [last_conv_activations, backbone_pooled_output].
      2) keep the remaining classifier-head layers (Dropout/Dense/...) as
         a plain list, and apply them imperatively inside GradientTape.
    """
    backbone = None
    for layer in _cls_model.layers:
        if "efficientnet" in layer.name.lower():
            backbone = layer
            break
    if backbone is None:
        raise ValueError("Could not find the EfficientNet backbone layer in the classifier.")

    backbone_submodel = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(GRADCAM_LAYER_NAME).output, backbone.output],
    )

    head_layers = [
        layer for layer in _cls_model.layers
        if layer is not backbone and not isinstance(layer, tf.keras.layers.InputLayer)
    ]

    return backbone_submodel, head_layers


def make_gradcam_heatmap(grad_parts, img_batch, pred_index):
    backbone_submodel, head_layers = grad_parts

    with tf.GradientTape() as tape:
        conv_output, pooled_output = backbone_submodel(img_batch, training=False)
        tape.watch(conv_output)

        x = pooled_output
        for layer in head_layers:
            x = layer(x, training=False)
        predictions = x
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.45):
    base = pil_img.convert("RGB").resize((CLS_IMG_SIZE, CLS_IMG_SIZE))
    base_arr = np.array(base).astype(np.float32)

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((CLS_IMG_SIZE, CLS_IMG_SIZE))
    heatmap_resized = np.array(heatmap_img)

    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]  # (H, W, 3) in [0,1]

    overlay = base_arr * (1 - alpha) + (jet_heatmap * 255) * alpha
    return Image.fromarray(np.uint8(overlay))


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="BRISC2025 Brain Tumor Pipeline", layout="wide")
    st.title("🧠 BRISC2025 — Brain Tumor Classification, Segmentation & Grad-CAM")
    st.caption(
        "Upload an MRI scan. The model classifies it first; if a tumor is detected, "
        "it also runs segmentation and Grad-CAM to show where and why."
    )

    try:
        cls_model, seg_model = load_models()
    except FileNotFoundError as e:
        st.error(str(e))
        return
    grad_model = build_gradcam_model(cls_model)

    uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Upload an image to run the pipeline.")
        return

    pil_img = Image.open(uploaded_file)

    with st.spinner("Classifying..."):
        pred_class, probs = classify(cls_model, pil_img)

    col_img, col_result = st.columns([1, 1])
    with col_img:
        st.image(pil_img, caption="Uploaded scan", use_container_width=True)

    with col_result:
        st.subheader("Classification result")
        st.markdown(f"**Predicted class:** `{pred_class}`  \n**Confidence:** {probs.max()*100:.2f}%")
        st.bar_chart({name: float(p) for name, p in zip(CLASS_NAMES, probs)})

    if pred_class == NO_TUMOR_LABEL:
        st.success("No tumor detected — segmentation and Grad-CAM are skipped.")
        return

    st.warning(f"Tumor detected ({pred_class}) — running segmentation and Grad-CAM...")

    col_seg, col_cam = st.columns(2)

    with col_seg:
        with st.spinner("Running segmentation..."):
            mask = segment(seg_model, pil_img)
            seg_overlay, tumor_pct = overlay_mask_on_image(pil_img, mask)
        st.subheader("Segmentation (tumor mask)")
        st.image(seg_overlay, use_container_width=True)
        st.caption(f"Estimated tumor area: {tumor_pct:.2f}% of the frame")

    with col_cam:
        with st.spinner("Running Grad-CAM..."):
            pred_index = CLASS_NAMES.index(pred_class)
            cls_batch = preprocess_for_classifier(pil_img)
            heatmap = make_gradcam_heatmap(grad_model, cls_batch, pred_index)
            cam_overlay = overlay_heatmap_on_image(pil_img, heatmap)
        st.subheader("Grad-CAM (model attention)")
        st.image(cam_overlay, use_container_width=True)
        st.caption("Warmer colors = regions that most influenced the classifier's decision")


if __name__ == "__main__":
    main()
