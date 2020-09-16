import bokeh
from bokeh.plotting import figure
import streamlit as st
import skimage.morphology
from skimage.color import rgb2gray
import skimage.filters
import skimage.io
import numpy as np

# disable fileuploader deprecation warn
st.set_option('deprecation.showfileUploaderEncoding', False)

# remove padding on left 
st.beta_set_page_config(layout='wide')

cmap_dict = {'Greys' : bokeh.palettes.grey(256),
             'Inferno'  : bokeh.palettes.inferno(256),
             'Magma' : bokeh.palettes.magma(256),
             'Plasma' : bokeh.palettes.plasma(256),
             'Cividis' : bokeh.palettes.cividis(256),
             'Viridis' : bokeh.palettes.viridis(256),
             'Turbo' : bokeh.palettes.turbo(256)}

def bokeh_imshow(img, height=800, cmap=None):

    # dims and color
    n, m = img.shape
    c = bokeh.models.LinearColorMapper(cmap_dict[cmap])

    # keep aspect
    width = int(m/n * height)
    p = figure(plot_height=height, plot_width=width,
               x_range=[0, m], y_range=[0, n])
    p.image([np.flipud(img)], x=0, y=0, dw=m, dh=n, color_mapper=c)

    return p

@st.cache
def filter_img(img, median, gauss, chambolle):

    '''apply selected filters to img'''

    # uniform illumination
    if median > 0:
        sq = skimage.morphology.square(median)
        img = skimage.filters.median(img, sq)

    # subtract background
    if gauss > 0:
        img_gauss = skimage.filters.gaussian(img, gauss)

        # convert img from int to float64 for subtraction
        img_float = skimage.img_as_float(img)
        img = img_float - img_gauss

    return img


# sidebar options
st.sidebar.markdown('## Image options')
cmap = st.sidebar.selectbox('Colormap', ['Greys', 'Inferno', 'Magma', 'Plasma', 'Viridis', 'Cividis', 'Turbo'], 4)

st.sidebar.markdown('## Filters')
median = st.sidebar.slider('Median filter', min_value=0, max_value=10, value=0, step=1)
gauss = st.sidebar.slider('Background subtraction (Gaussian)', min_value=0, max_value=100, value=0, step=5)
chambolle = st.sidebar.slider('Denoising variation filter (Chambolle)', min_value=0.0, max_value=0.1, step=0.01)

st.sidebar.markdown('## Thresholding')


st.sidebar.markdown('## Mask overlay')
mask = st.sidebar.radio('Segments', ['None', 'Unique', 'Full'])

# load image
img_f = st.file_uploader('Input image', type=['jpg', 'tif', 'png'])
if img_f is not None:

    # convert rgb to grey
    img = skimage.io.imread(img_f)
    if img.ndim >= 3:
        img = rgb2gray(img)

    # filter
    img = filter_img(img, median, gauss, chambolle)
    p = bokeh_imshow(img, cmap=cmap)
    st.bokeh_chart(p)


