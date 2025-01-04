import gradio as gr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def hello(name):
    return f"Hello {name}!"

try:
    logger.info("Starting test server...")
    demo = gr.Interface(fn=hello, inputs="text", outputs="text")
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        debug=True
    )
    logger.info("Test server started successfully")
except Exception as e:
    logger.error(f"Error: {str(e)}")