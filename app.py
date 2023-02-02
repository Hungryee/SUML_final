from fastbook import *
from fastai.vision.widgets import *
import pathlib
from pathlib import Path
import os
import streamlit as sl


os.environ['KMP_DUPLICATE_LIB_OK']='True'

def init_learner():
    path = Path('dataset/images/Images')


    data = DataBlock(
        blocks=[ImageBlock, CategoryBlock],
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2,seed=42),
        get_y=parent_label,
        item_tfms=[Resize(256)]
    )

    data = data.new(
        item_tfms=[RandomResizedCrop(224, min_scale=0.5)],
        batch_tfms=aug_transforms(mult=0.0, do_flip=False, flip_vert=False, max_rotate=0.0, min_zoom=0.0, max_zoom=0.0, max_lighting=0.0, max_warp=0.0, p_affine=0.0, p_lighting=0.0, xtra_tfms=None, size=None, mode='bilinear', pad_mode='border', align_corners=True, batch=False, min_scale=1.0))

    dls = data.dataloaders(path, bs = 1)

    learn = cnn_learner(dls, resnet34, metrics=error_rate)

    # learn.fit(1)
    # learn.save('saved_model')
    learn.load('saved_model')

    return learn


def main():
    sl.header('Predict breed of a dog')
    sl.subheader('s20205')
    model = init_learner()

    uploaded_image = sl.file_uploader(
        'Upload your own image!',
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_image:
        sl.image(uploaded_image)
        prediction_result = model.predict(uploaded_image.getvalue())[0]

        prediction_result = prediction_result.split('-')[-1]
        sl.text(prediction_result)

if __name__ == '__main__':
    main()
