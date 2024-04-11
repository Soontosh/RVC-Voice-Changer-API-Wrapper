"""
Heavily cut down CLI file from Mangio-RVC-Fork repo. Converted to simple function for 3B1B use case.
Original File: https://github.com/Mangio621/Mangio-RVC-Fork/blob/main/infer-web.py
"""

# Import necessary libraries
import os
import shutil
import sys

# Get the directory of python script
now_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the system path
sys.path.append(now_dir)

import warnings
import numpy as np
import torch
import traceback

# Import necessary modules and functions
from config import Config
from fairseq import checkpoint_utils
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from my_utils import load_audio
from vc_infer_pipeline import VC
import scipy.io.wavfile as wavfile

# Define temporary directory and clean up if it exists
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
# Clean up other directories
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
# Create necessary directories
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "audios"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "datasets"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
# Set environment variable
os.environ["TEMP"] = tmp
# Ignore warnings
warnings.filterwarnings("ignore")
# Set seed for reproducibility
torch.manual_seed(114514)

# Define root directory for weights
weight_root = os.path.join(now_dir, "weights")

# Create a configuration object
config = Config()

# Import SQLite3 for database operations
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('TEMP/db:cachedb?mode=memory&cache=shared', check_same_thread=False)
cursor = conn.cursor()

# Create necessary tables in the database
cursor.execute("""
    CREATE TABLE IF NOT EXISTS formant_data (
        Quefrency FLOAT,
        Timbre FLOAT,
        DoFormant INTEGER
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS stop_train (
        stop BOOL
    )
""")

# Declare global variables
global DoFormant, Quefrency, Timbre

# Initialize Hubert model
hubert_model = None

def clear_sql():
    cursor.execute("DELETE FROM formant_data")
    cursor.execute("DELETE FROM stop_train")
    conn.commit()
    conn.close()
    print("Clearing SQL database...")

def load_hubert():
    """
    Function to load the Hubert model.
    """
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(now_dir, "hubert_base.pt")],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

def vc_single(
    sid,
    input_audio_path0,
    input_audio_path1,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
):  
    """
    Function to perform voice conversion on a single audio file.
    """

    # Add current directory to input_audio_path0 and input_audio_path1
    input_audio_path0 = os.path.join(now_dir, input_audio_path0)
    input_audio_path1 = os.path.join(now_dir, input_audio_path1)

    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path0 is None or input_audio_path0 is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        if input_audio_path0 == '':
            audio = load_audio(input_audio_path1, 16000, DoFormant, Quefrency, Timbre)
            
        else:
            audio = load_audio(input_audio_path0, 16000, DoFormant, Quefrency, Timbre)
            
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        ) 

        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path1,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0_file=f0_file,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)
    
def get_vc(sid, to_return_protect0, to_return_protect1):
    """
    Load a voice conversion model based on the provided speaker id (sid).
    
    Args:
        sid (str): The speaker id.
        to_return_protect0 (dict): A dictionary to be updated based on the model configuration.
        to_return_protect1 (dict): A dictionary to be updated based on the model configuration.
        
    Returns:
        tuple: A tuple of dictionaries with updated configurations.
    """
    # Global variables
    global n_spk, tgt_sr, net_g, vc, cpt, version

    # If sid is empty or a list, clean up the existing models
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Load the model configuration
            # ...
            # Clean up the model and configuration
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return ({"visible": False, "__type__": "update"}, {"visible": False, "__type__": "update"}, {"visible": False, "__type__": "update"})
    # If sid is not empty, load the model for the speaker
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 0:
        to_return_protect0 = to_return_protect1 = {
            "visible": False,
            "value": 0.5,
            "__type__": "update",
        }
    else:
        to_return_protect0 = {
            "visible": True,
            "value": to_return_protect0,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": True,
            "value": to_return_protect1,
            "__type__": "update",
        }
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]

    # Return the updated configurations
    return (
        {"visible": True, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )

def V2V_infer(com): # Inputs as list
    """
    Perform voice conversion based on the provided command-line arguments.
    
    Args:
        com (list): A list of command-line arguments. #need more info lol
    """

    # Global variables
    global DoFormant, Quefrency, Timbre

    # Get parameters for inference
    model_name = com[0]
    source_audio_path = com[1]
    output_file_name = com[2]
    feature_index_path = com[3]
    f0_file = None # Not Implemented Yet
    speaker_id = int(com[4])
    transposition = float(com[5])
    f0_method = com[6]
    crepe_hop_length = int(com[7])
    harvest_median_filter = int(com[8])
    resample = int(com[9])
    mix = float(com[10])
    feature_ratio = float(com[11])
    protection_amnt = float(com[12])
    protect1 = 0.5
    
    if com[13] == 'False' or com[13] == 'false' or com[13] == False:
        DoFormant = False
        Quefrency = 0.0
        Timbre = 0.0
        cursor.execute("DELETE FROM formant_data")
        cursor.execute("INSERT INTO formant_data (Quefrency, Timbre, DoFormant) VALUES (?, ?, ?)", (Quefrency, Timbre, 0))
        conn.commit()
    else:
        print("Com 13: ", com[13], "; Type: ", type(com[13]))
        DoFormant = True
        Quefrency = float(com[15])
        Timbre = float(com[16])
        cursor.execute("DELETE FROM formant_data")
        cursor.execute("INSERT INTO formant_data (Quefrency, Timbre, DoFormant) VALUES (?, ?, ?)", (Quefrency, Timbre, 1))
        conn.commit()
    
    print("Starting the inference...")
    vc_data = get_vc(model_name, protection_amnt, protect1)
    print(vc_data)
    print("Performing inference...")
    conversion_data = vc_single(
        speaker_id,
        source_audio_path,
        source_audio_path,
        transposition,
        f0_file,
        f0_method,
        feature_index_path,
        feature_index_path,
        feature_ratio,
        harvest_median_filter,
        resample,
        mix,
        protection_amnt,
        crepe_hop_length,        
    )

    audio_output_folder = os.path.join(now_dir, 'audio-outputs')

    if "Success." in conversion_data[0]:
        print("Inference succeeded. Writing to %s/%s..." % (audio_output_folder, output_file_name))
        wavfile.write('%s/%s' % (audio_output_folder, output_file_name), conversion_data[1][0], conversion_data[1][1])
        print("Finished! Saved output to %s/%s" % (audio_output_folder, output_file_name))
    else:
        print("Inference failed. Here's the traceback: ")
        print(conversion_data[0])

        return conversion_data
    
    clear_sql()

    return os.path.join(audio_output_folder, output_file_name)