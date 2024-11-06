import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import whisper
import pandas as pd
import gtts  
import os
import playsound
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav