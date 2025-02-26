import tensorflow as tf
import numpy as np
import argparse
import os
from datetime import datetime

import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def accuracy(model_path, models, nbStep):


    for model in models:
        model.evaluate_latest_model(model_path, nbStep)
