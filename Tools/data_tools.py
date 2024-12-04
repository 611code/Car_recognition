import sys
import os

def Data_DeBug(msg,data):
    try:
        shape = data.shape
    except:
        shape = None
    print(f"[Message:{msg}, data:{data}, shape:{shape}]")