import os
import tempfile
import json
from typing import TypedDict, List, Dict
import PyPDF2
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import MultiQueryRetriever
from langgraph.graph import END, StateGraph

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "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")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "quiz-generator")
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize LangSmith client for tracing
client = Client(api_key=LANGCHAIN_API_KEY)
tracer = LangChainTracer(project_name=LANGCHAIN_PROJECT)

# Initialize LLM
llm = ChatGroq(
    temperature=0.2,
    model_name="llama3-8b-8192",
    groq_api_key=GROQ_API_KEY,
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Define state types for LangGraph
class GraphState(TypedDict):
    retriever: MultiQueryRetriever
    content: str
    difficulty: str
    num_questions: int
    questions: List[Dict]

# Document processing function
def process_document(file_path, file_type):
    """Loads the document, splits it into chunks, and creates a retriever."""
    if file_type == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_type == 'docx':
        loader = Docx2txtLoader(file_path)
    else:  # Default to text
        loader = TextLoader(file_path)
    
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Use MultiQueryRetriever to generate multiple queries
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )
    
    return retriever

# LangGraph nodes
def retrieve_content(state: GraphState) -> GraphState:
    """Retrieve relevant content based on difficulty level."""
    retriever = state.get("retriever")
    difficulty = state.get("difficulty", "medium")
    
    if retriever is None:
        raise ValueError("Retriever object is missing")

    # Provide a valid query
    query = f"Information for {difficulty} difficulty quiz"
    
    # Get relevant documents
    docs = retriever.get_relevant_documents(query)
    content = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant content found."

    return {
        "retriever": retriever,
        "content": content,
        "difficulty": difficulty,
        "num_questions": state["num_questions"]
    }

def generate_questions(state: GraphState) -> GraphState:
    """Generate quiz questions based on content."""
    content = state["content"]
    difficulty = state["difficulty"]
    num_questions = state["num_questions"]
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an expert quiz creator. Create {num_questions} quiz questions with the following parameters:
    
    1. Difficulty level: {difficulty}
    2. Each question should have four possible answers (A, B, C, D)
    3. One answer should be correct
    4. Only use information found in the provided content
    
    Content:
    {content}
    
    Return the quiz in the following JSON format:
    json
    [
        {{"question": "Question text",
          "options": [
              "A. Option A",
              "B. Option B", 
              "C. Option C",
              "D. Option D"
          ],
          "correct_answer": "A. Option A",
          "explanation": "Brief explanation of why this is correct"
        }}
    ]
    
    Only return the JSON without any additional explanation or text.
    """)
    
    # Initialize parser
    parser = JsonOutputParser()
    
    # Create chain
    chain = prompt | llm | parser
    
    # Generate questions
    questions = chain.invoke({
        "content": content,
        "difficulty": difficulty,
        "num_questions": num_questions
    })
    
    return {"questions": questions}

# Create LangGraph
def create_quiz_graph():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve_content", retrieve_content)
    workflow.add_node("generate_questions", generate_questions)
    
    # Add edges
    workflow.add_edge("retrieve_content", "generate_questions")
    workflow.add_edge("generate_questions", END)
    
    # Set entry point
    workflow.set_entry_point("retrieve_content")
    
    return workflow.compile()

# API routes
@app.route('/api/generate-quiz', methods=['POST'])
def generate_quiz():
    """Handles file upload and generates a quiz."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Get parameters
    difficulty = request.form.get('difficulty', 'medium')
    num_questions = int(request.form.get('num_questions', 5))
    
    # Get file extension
    file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'txt'
    
    # Determine file type
    file_type = 'pdf' if file_extension == 'pdf' else 'docx' if file_extension in ['doc', 'docx'] else 'text'
    
    # Save file temporarily
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    try:
        # Process document
        retriever = process_document(file_path, file_type)
        
        # Create and run graph
        quiz_graph = create_quiz_graph()
        result = quiz_graph.invoke({
            "retriever": retriever,
            "difficulty": difficulty,
            "num_questions": num_questions
        })
        
        return jsonify({
            "quiz": result["questions"],
            "metadata": {
                "difficulty": difficulty,
                "num_questions": len(result["questions"]),
                "source": file.filename
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if the API is running."""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
