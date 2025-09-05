import keras

# Paths
base = "/home/houyh/nova/nova"   
MODEL_DIR       = base+"/models/"
LOG_DIR         = base+"/logs/"
PLOT_DIR        = base+"/plots/"
TENSORBOARD_DIR = base+"/tensorboard/"
DATA_DIR        = base+"/data/"
NUE_DATA_DIR    = base+"/data/raw/august25/"

# Data Loading
MAX_EXAMPLES = 2000
PIXEL_SCALE = 100.0
VERTEX_SCALE = 0.01
CUT_THRESHOLD = 0.05

# Keras
DATA_FORMAT = keras.backend.image_data_format()
