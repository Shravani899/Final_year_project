import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Create widgets
prompt = ctk.CTkEntry(master=app, height=40, width=512)
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

# Load the model
modelid = "CompVis/stable-diffusion-v1-4"

device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    image.save('generatedimage.png')
    
    # Convert the image for Tkinter
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img  # Keep a reference to avoid garbage collection

trigger = ctk.CTkButton(master=app, height=40, width=120, text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
