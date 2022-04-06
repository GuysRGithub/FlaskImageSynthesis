import gradio as gr
import torch 

is_cuda = torch.cuda.is_available()
def greet(name):
    if is_cuda:
        return "Hello cuda" + name + "!!"
    else:
        return "Hello ooops" + name + "!!"
iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()