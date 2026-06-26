import gradio as gr
from rag_pipeline import RAGPipeline

# Initialize RAG Pipeline with your API key
groq_api_key = '<GROQ API KEY>'
rag_pipeline = RAGPipeline(groq_api_key=groq_api_key)


# ==================== GRADIO FUNCTIONS ====================

def process_pdf(pdf_file):
    """Process uploaded PDF file"""
    if pdf_file is None:
        return "Please upload a PDF file."
    
    success, message = rag_pipeline.process_pdf(pdf_file.name)
    
    if success:
        return f"{message}"
    else:
        return f"{message}"


def answer_question(question, top_k):
    """Answer question using RAG"""
    if not question or question.strip() == "":
        return "Please enter a question."
    
    # Check if PDF has been processed
    if rag_pipeline.vector_store.get_collection_count() == 0:
        return "Please upload and process a PDF first before asking questions!"
    
    answer = rag_pipeline.query(question, top_k=int(top_k))
    return answer


# ==================== GRADIO INTERFACE ====================

with gr.Blocks(title="PDF Question Answering System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    #  PDF Question Answering System (RAG)
    
    Upload a PDF document and ask questions about its content. The system uses Retrieval-Augmented Generation (RAG) 
    to find relevant information and provide accurate answers.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Upload PDF")
            pdf_input = gr.File(
                label="Upload PDF Document",
                file_types=[".pdf"],
                type="filepath"
            )
            process_btn = gr.Button("Process PDF", variant="primary", size="lg")
            status_output = gr.Textbox(
                label="Processing Status",
                interactive=False,
                lines=3
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Step 2: Ask Questions")
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Enter your question about the PDF content...",
                lines=3
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Number of Context Chunks to Retrieve"
            )
            answer_btn = gr.Button("Get Answer", variant="primary", size="lg")
            answer_output = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=10
            )
    
    gr.Markdown("""
    ---
    ### How to Use:
    1. **First**: Upload a PDF document in Step 1 and click "Process PDF"
    2. **Then**: Wait for the success message
    3. **Finally**: Ask questions about the PDF content in Step 2
    
    ### Tips:
    - Upload a clear, text-based PDF for best results
    - Ask specific questions related to the content
    - Increase the number of context chunks for more comprehensive answers
    - The system works best with well-structured documents
    """)
    
    # Example questions
    gr.Markdown("### Example Questions (Use after uploading PDF):")
    example_questions = [
        ["What is the main topic of this document?"],
        ["Summarize the key findings."],
        ["What methodology was used?"],
        ["What are the conclusions?"]
    ]
    
    gr.Examples(
        examples=example_questions,
        inputs=question_input,
        label="Click to use example questions"
    )
    
    # Connect buttons to functions
    process_btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=status_output
    )
    
    answer_btn.click(
        fn=answer_question,
        inputs=[question_input, top_k_slider],
        outputs=answer_output
    )


# ==================== LAUNCH ====================

if __name__ == "__main__":
    demo.launch(share=True)