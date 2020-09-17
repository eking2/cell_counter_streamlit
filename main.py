import bokeh
from bokeh.plotting import figure
from bokeh.models import WheelZoomTool, Span
import streamlit as st
import skimage.morphology
from skimage.color import rgb2gray
import skimage.filters
import skimage.io
import skimage.restoration
import skimage.segmentation
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

def bokeh_imshow(img, overlay=None, height=800, cmap=None, overlay_solid=False):

    '''plot image array with bokeh'''

    # dims
    n, m = img.shape

    # keep aspect
    width = int(m/n * height)
    p = figure(plot_height=height, plot_width=width,
               x_range=[0, m], y_range=[0, n])
    # wheel zoom on
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.xgrid.visible = False
    p.ygrid.visible = False

    if overlay is not None:
        # change 0 to na in overlay, then transparency
        overlay = overlay.astype('float')
        overlay[overlay == 0] = np.nan
        p.image([np.flipud(img)], x=0, y=0, dw=m, dh=n, color_mapper=bokeh.models.LinearColorMapper(bokeh.palettes.grey(256)),
                global_alpha=0.5)

        if overlay_solid:
            overlay[overlay > 0] = 1
            r = p.image([np.flipud(overlay)], x=0, y=0, dw=m, dh=n, palette=['#00ff00'])

        # unique labels
        else:
            r = p.image([np.flipud(overlay)], x=0, y=0, dw=m, dh=n, color_mapper=bokeh.models.LinearColorMapper(cmap_dict['Turbo']))

        # set 0 transparent
        r.glyph.color_mapper.nan_color = (0, 0, 0, 0)

    # no overlay
    else:
        c = bokeh.models.LinearColorMapper(cmap_dict[cmap])
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

    # total variation filter, denoise
    if chambolle > 0:
        img = skimage.restoration.denoise_tv_chambolle(img, weight=chambolle)

    # scale to [0, 255]
    img_min = img.min()
    img_max = img.max()

    x_std = (img - img_min) / (img_max - img_min)
    img = x_std * 255

    return img


def img_hist(img, height, thresh=None):

    '''plot histogram of image intensity values'''

    img_arr = img.flatten()

    # freedman-diaconis to select n bins
    #q75, q25 = np.percentile(img_arr, [75, 25])
    #denom = len(img_arr)**(1/3)
    #bin_width = 2 * (q75 - q25)/denom
    #bins = (img_arr.max() - img_arr.min()) / bin_width
    #bins = int(bints)

    hist, edges = np.histogram(img_arr, 'auto', density=True)

    # plot
    p = figure(plot_height=height-100)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.xaxis.axis_label = 'Pixel intensity'
    p.yaxis.axis_label = 'Density'

    if thresh is not None:
        thresh_line = Span(location=thresh, dimension='height', line_color='red', line_width=2)
        p.add_layout(thresh_line)

    return p

def segment(img, thresh, border, area_bounds, ecc_bounds):

    '''segment image by threshold and count regions'''

    img_thresh = img < thresh

    # label each segment, first round without border, size, area cutoffs
    img_labels = skimage.measure.label(img_thresh)

    # ignore cells that are on border
    img_thresh = skimage.segmentation.clear_border(img_labels, buffer_size=border)

    # cutoff on cell size
    # and eccentricity (how circular, 0 = perfect circle, 1 = ellipse)
    # avoids overlapping cells
    props = skimage.measure.regionprops(img_thresh)
    saved_segments = np.zeros_like(img_thresh)
    for prop in props:
        if prop.area >= area_bounds[0] \
        and prop.area <= area_bounds[1] \
        and prop.eccentricity >= ecc_bounds[0] \
        and prop.eccentricity <= ecc_bounds[1]:
            saved_segments += img_thresh == prop.label

    # # second round label with filtered segments
    img_thresh = skimage.measure.label(saved_segments > 0)
    saved_props = skimage.measure.regionprops(img_thresh)

    return img_thresh, saved_props



# sidebar options
st.sidebar.markdown('## Image options')
cmap = st.sidebar.selectbox('Colormap', ['Greys', 'Inferno', 'Magma', 'Plasma', 'Viridis', 'Cividis', 'Turbo'], 4)
height = st.sidebar.number_input('Height', min_value=200, value=800)

st.sidebar.markdown('## Filters')
median = st.sidebar.slider('Median filter', min_value=0, max_value=10, value=0, step=1)
gauss = st.sidebar.slider('Background subtraction (Gaussian)', min_value=0, max_value=100, value=0, step=5)
chambolle = st.sidebar.slider('Denoising variation filter (Chambolle)', min_value=0.0, max_value=0.5, step=0.05)

st.sidebar.markdown('## Thresholding')
thresh_method = st.sidebar.radio('Method', ['Manual', 'Otsu'])
if thresh_method == 'Manual':
    thresh = st.sidebar.slider('Pixel threshold', min_value=0, max_value=256, value=0, step=1)
border = st.sidebar.slider('Border pixel buffer', min_value=0, max_value=10, value=0, step=1)
area = st.sidebar.slider('Area bounds', min_value=0, max_value=2000, value=(0, 2000), step=50)
ecc = st.sidebar.slider('Eccentricity bounds', min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.1)


st.sidebar.markdown('## Mask overlay')
mask = st.sidebar.radio('Segments', ['None', 'Unique', 'Full'])

# load image
st.title('Cell counter')
img_f = st.file_uploader('Input image', type=['jpg', 'tif', 'png'])
if img_f is not None:

    # convert rgb to grey
    img = skimage.io.imread(img_f)
    if img.ndim >= 3:
        img = rgb2gray(img)

    # filter
    img = filter_img(img, median, gauss, chambolle)

    # otsu on filtered image
    if thresh_method == 'Otsu':
        thresh = skimage.filters.threshold_otsu(img)

    # segment
    img_thresh, props = segment(img, thresh, border, area, ecc)

    if mask == 'None':
        p = bokeh_imshow(img, height=height, cmap=cmap)
    elif mask == 'Unique':
        p = bokeh_imshow(img, img_thresh, height=height)
    elif mask == 'Full':
        p = bokeh_imshow(img, img_thresh, height=height, overlay_solid=True)
    st.bokeh_chart(p)
    st.text('')

    # plot intensity hist
    if thresh:
        p2 = img_hist(img, height, thresh)
        st.bokeh_chart(p2)

        count = len(props)
        avg_area = np.round(np.mean([prop.area for prop in props]), 2)
        avg_ecc = np.round(np.mean([prop.eccentricity for prop in props]), 2)
        st.write('Colonies:', count)
        st.write('Average area:', avg_area)
        st.write('Average eccentricity:', avg_ecc)





