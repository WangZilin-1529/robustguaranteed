import pandas as pd
import scipy.stats as st
import math

def calculate_sample_size(proportion, MoE, confi_level):
    p1 = proportion*(1-proportion)
    z = st.norm.ppf(confi_level)
    p2 = (z/MoE)**2
    return math.ceil(p1*p2)
