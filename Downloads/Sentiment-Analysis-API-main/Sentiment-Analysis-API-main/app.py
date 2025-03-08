import gradio as gr
from transformers import pipeline

# Load Sentiment Analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})"

# Create Gradio Interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="text",
    title="Sentiment Analysis API",
    description="Enter a sentence, and the model will predict if it's POSITIVE or NEGATIVE."
)

# Launch the Gradio app
iface.launch()
