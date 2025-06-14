import os
import gradio as gr
from ipcc_colab_agent import IPCCLLMAgent, create_interface

# Initialize the agent
agent = IPCCLLMAgent()

# Create Gradio interface
interface = create_interface()

# For Netlify deployment
app = interface.app

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8080)
