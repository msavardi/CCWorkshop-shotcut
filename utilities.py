"""
utilities.py

Utility functions and classes for the C&C Workshop.
Includes image preprocessing, video frame extraction, annotation UI, and a custom Keras data generator.
"""

import os
import random

import holoviews as hv
import imageio.v2 as imageio
import numpy as np
import panel as pn
import tensorflow as tf
from bokeh.io import output_notebook
from matplotlib.colors import ListedColormap
from PIL import Image
from PIL import Image as pil_image
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

# Custom colormap for visualizations
custom_cmap = ListedColormap(['#1f77b4', '#2ca02c', '#ff7f0e'])

hv.extension('bokeh')
output_notebook()

# Mapping from class index to class name
id2cls = ['CS', 'MS', 'LS']


def save_image(img_array, label, index, parent_folder='train'):
    """
    Save an image array to disk as a JPEG in a label-specific folder.

    Args:
        img_array (np.ndarray): Image data as a numpy array (float [0,1] or int [0,255]).
        label (str): Class label (e.g., 'CS', 'MS', 'LS').
        index (int): Index for the filename.
        parent_folder (str): Parent directory to save images in.
    """
    label_folder = os.path.join(parent_folder, label)
    os.makedirs(label_folder, exist_ok=True)
    img_8bit = (img_array * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_8bit)

    # Save file as: CS/0.jpg, MS/1.jpg, etc.
    img_pil.save(os.path.join(label_folder, f"{index}.jpg"), format='JPEG')


def shot_scale(image, model, dims=(125, 224)):
    """
    Predict the shot scale class for an image using a model.

    Args:
        image (np.ndarray): Input image array.
        model (tf.keras.Model): Trained model for prediction.
        dims (tuple): Target dimensions (height, width).

    Returns:
        tuple: (predicted class index, softmax probabilities)
    """
    width_height_tuple = (dims[1], dims[0])
    cval = 0
    try:
        raw_img = pil_image.fromarray(image)
        img = raw_img.copy()
        img.thumbnail(width_height_tuple, pil_image.NEAREST)
        final_img = pil_image.new(
            img.mode, width_height_tuple,
            (cval if img.mode == 'L' else (cval, cval, cval))
        )
        final_img.paste(
            img,
            ((width_height_tuple[0] - img.size[0]) // 2,
             (width_height_tuple[1] - img.size[1]) // 2)
        )
        image_c = np.asarray(final_img, dtype='float32') / 255.
        image_bn = np.asarray(final_img.convert('LA').convert('RGB'), dtype='float32') / 255.
        image = np.stack([image_c, image_bn], axis=0)
        pp = np.sum(model.predict(image, verbose=None), axis=0)
    except Exception as e:
        print(f"[{e}] A loading error occurred")
        return 0
    return np.argmax(pp), tf.nn.softmax(pp)


def extract_from_video(filename, time_step, model):
    """
    Extract frames from a video at regular intervals and predict shot scale.

    Args:
        filename (str): Path to video file.
        time_step (float): Time interval (in seconds) between frames.
        model (tf.keras.Model): Trained model for prediction.

    Returns:
        list: List of [movie, frame_num, pred_class, confidence, image].
    """
    vid = imageio.get_reader(filename, 'ffmpeg')
    movie = os.path.basename(filename)[:-4]
    print(movie)
    out = []
    meta = vid.get_meta_data()
    nframe = meta['duration'] * meta['fps']
    for num in tqdm(range(int(nframe // (meta['fps'] * time_step)))):
        try:
            image = vid.get_data(int(time_step * num * meta['fps']))
        except Exception:
            continue
        preds = shot_scale(image, model)
        out.append([
            movie, num, preds[0],
            tf.nn.softmax(preds[1])[preds[0]].numpy(), image
        ])
    return out


def view_predictions(out, downsample=5):
    """
    Visualize model predictions on a sequence of images.

    Args:
        out (list): Output from extract_from_video.
        downsample (int): Downsampling factor for display.

    Returns:
        hv.HoloMap: Interactive visualization of predictions.
    """
    v = {}
    box = hv.Box(0.4, 0.45, spec=0.04).opts(color='white', line_width=10)
    for (m, t, p, c, array) in out:
        img = np.asarray(array[::downsample, ::downsample, :], dtype='float32') / 255.
        v.update({
            t: (hv.RGB(img).opts(width=500) * box * hv.Text(0.4, 0.45, f"{id2cls[p]}"))
        })
    hv.extension('bokeh')
    return hv.HoloMap(v, kdims=['Time'])


def create_annotation_ui(samples):
    """
    Create a Panel UI for annotating a list of image samples.

    Args:
        samples (list): List of (image_array, label, extra) tuples.

    Returns:
        tuple: (Panel app, dict of selectors)
    """
    p_selectors = {}
    current_index = pn.state.cache.setdefault("current_index", 0)

    def next_callback(event):
        current_index = pn.state.cache.get("current_index", 0)
        if current_index + 1 < len(samples):
            current_index += 1
            pn.state.cache["current_index"] = current_index
            update_view(current_index)
        else:
            next_button.disabled = True

    def update_view(index):
        array, p, _ = samples[index]
        current_view[:] = [create_view(index, array, p)]
        label = p
        img = np.asarray(array, dtype='float32') / 255.0
        save_image(img, label, index)

    def create_view(index, array, initial_p):
        img = np.asarray(array[::2, ::2, :], dtype='float32') / 255.0
        rgb = hv.RGB(img).opts(width=500)
        p_selector = pn.widgets.RadioButtonGroup(name='Type', options=id2cls, value=initial_p)
        p_selectors[index] = p_selector

        @pn.depends(p_selector)
        def plot(p_value):
            return rgb * box * hv.Text(0.4, 0.45, p_value).opts(color='black', fontsize=12)
        return pn.Row(p_selector, plot)

    box = hv.Box(0.4, 0.45, spec=0.04).opts(color='white', line_width=10)
    current_view = pn.Column()
    next_button = pn.widgets.Button(name="Next", button_type="primary")
    next_button.on_click(next_callback)
    update_view(current_index)
    app = pn.Column(current_view, next_button)
    return app, p_selectors


def preprocessing(image, model=None, dims=(125, 224)):
    """
    Preprocess an image: resize, pad, and normalize.

    Args:
        image (np.ndarray): Input image array.
        model: (Unused, for compatibility).
        dims (tuple): Target dimensions (height, width).

    Returns:
        np.ndarray: Preprocessed image array (float32, [0,1]).
    """
    width_height_tuple = (dims[1], dims[0])
    cval = 0
    raw_img = pil_image.fromarray(image)
    img = raw_img.copy()
    img.thumbnail(width_height_tuple, pil_image.NEAREST)
    final_img = pil_image.new(
        img.mode, width_height_tuple,
        (cval if img.mode == 'L' else (cval, cval, cval))
    )
    final_img.paste(
        img,
        ((width_height_tuple[0] - img.size[0]) // 2,
         (width_height_tuple[1] - img.size[1]) // 2)
    )
    return np.asarray(final_img, dtype='float32') / 255.


class CustomImageGenerator(Sequence):
    """
    Custom Keras Sequence for loading and batching images from directories.

    Args:
        data_dir (str): Path to data directory with subfolders per class.
        batch_size (int): Number of images per batch.
        dims (tuple): Target image dimensions (height, width).
        shuffle (bool): Whether to shuffle data each epoch.
        val_split (float): Fraction of data to use for validation.
        is_val (bool): If True, use validation split; else use training split.
    """
    def __init__(self, data_dir, batch_size=8, dims=(125, 224), shuffle=True,
                 val_split=0.2, is_val=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dims = dims
        self.shuffle = shuffle
        self.is_val = is_val
        self.classes = sorted(os.listdir(data_dir))
        self.class_indices = {'CS': 0, 'MS': 1, 'LS': 2}
        self.filepaths = []
        for label in self.classes:
            for fname in os.listdir(os.path.join(data_dir, label)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    self.filepaths.append((os.path.join(data_dir, label, fname), self.class_indices[label]))
        random.shuffle(self.filepaths)
        split = int(len(self.filepaths) * (1 - val_split))
        self.filepaths = self.filepaths[split:] if is_val else self.filepaths[:split]
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data."""
        batch = self.filepaths[idx * self.batch_size:(idx + 1) * self.batch_size]
        x, y = [], []
        for path, label in batch:
            img = np.array(pil_image.open(path).convert('RGB'))
            processed = preprocessing(img, dims=self.dims)
            x.append(processed)
            y.append(label)
        x = np.array(x)
        y = tf.keras.utils.to_categorical(y, num_classes=len(self.classes))
        return x, y

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            random.shuffle(self.filepaths)
