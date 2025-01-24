import os
from typing import List
from dotenv import load_dotenv, find_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import get_response_synthesizer
from custom_query_engines import SummaryQueryEngine, VectorQueryEngine
from custom_retriever import CustomRetriever
import shutil

class RAGPipeline:
    def __init__(self, gemini_key_env_var='GEMINI_API_KEY'):
        # Load environment variables
        load_dotenv(find_dotenv())
        self.GEMINI_API_KEY = os.getenv(gemini_key_env_var)

        # Set default models and embeddings
        Settings.llm = Gemini(api_key=self.GEMINI_API_KEY)
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.documents = None
        self.single_document = None
        self.nodes = None
        self.nodes_single = None
        self.summary_index = None
        self.vector_index = None
        self.keyword_index = None
        self.summary_query_engine = None
        self.vector_query_engine = None
        self.summary_tool = None
        self.vector_tool = None
        self.case_outcome_tool = None
        self.query_engine = None

    def load_documents(self, input_dir=None, input_files=None):
        """Load documents from a directory or specific files."""
        if input_dir:
            self.documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
        if input_files:
            self.single_document = SimpleDirectoryReader(input_files=input_files).load_data()

    def split_documents(self, chunk_size=8192):
        """Split documents into chunks for processing."""
        splitter = SentenceSplitter(chunk_size=chunk_size)
        if self.documents:
            self.nodes = splitter.get_nodes_from_documents(self.documents)
        if self.single_document:
            self.nodes_single = splitter.get_nodes_from_documents(self.single_document)

    def create_indices(self):
        """Create the summary index and vector store index."""
        if self.nodes_single:
            self.summary_index = SummaryIndex(self.nodes_single)
        if self.nodes:
            self.vector_index = VectorStoreIndex(self.nodes)
            self.keyword_index = SimpleKeywordTableIndex(self.nodes)

    def create_query_engines(self):
        """Create query engines for summary and vector-based retrieval."""
        if self.summary_index:
            self.summary_query_engine = SummaryQueryEngine(
                retriever=self.summary_index.as_retriever(similarity_top_k=10),
                synthesizer=get_response_synthesizer(),
                llm = Settings.llm
            )
        if self.vector_index and self.keyword_index:

            custom_retriever = CustomRetriever(
            vector_retriever=self.vector_index.as_retriever(),
            keyword_retriever=self.keyword_index.as_retriever(),
            bert_model="onnx/",
            mode="AND"
        )

            self.vector_query_engine = VectorQueryEngine(
                retriever=custom_retriever,
                synthesizer=get_response_synthesizer(),
                llm = Settings.llm
            )

    def create_tools(self):
        """Create query tools for summarization and vector retrieval."""
        if self.summary_query_engine:
            self.summary_tool = QueryEngineTool.from_defaults(
                query_engine=self.summary_query_engine,
                description="Useful for summarization questions related to legal Case Files."
            )
        if self.vector_query_engine:
            self.vector_tool = QueryEngineTool.from_defaults(
                query_engine=self.vector_query_engine,
                description="Useful for retrieving specific context from the legal cases provided. Provides all the relevant Acts, Sections, and other legal information related to the query in the response."
            )
        if self.vector_query_engine:
            self.case_outcome_tool = QueryEngineTool.from_defaults(
                query_engine=self.vector_query_engine,
                description="Useful for predicting outcome of a given case or scenario in the query using the legal cases provided as context when explicityly asked for outcome or prediction."
                
            )

    def create_router_engine(self, verbose=True):
        """Create the router engine to handle query routing."""
        # Check if all the tools are valid
        query_engine_tools = [self.summary_tool, self.vector_tool, self.case_outcome_tool]
        
        # Remove any None tools
        query_engine_tools = [tool for tool in query_engine_tools if tool is not None]
        
        # Check if there are valid tools left
        if not query_engine_tools:
            raise ValueError("No valid query engine tools found. Cannot create router engine.")

        self.query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=query_engine_tools,
            verbose=verbose
        )

    def query(self, query):
        """Query the router engine."""
        response = self.query_engine.query(query)
        return str(response)

# Your FastAPI app and endpoint definitions...

UPLOAD_DIR = "uploads"
if os.path.exists(UPLOAD_DIR):
    shutil.rmtree(UPLOAD_DIR)
os.makedirs(UPLOAD_DIR)

file_uploaded = False

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    question: str

@app.post("/upload-files")
async def upload_files(files: List[UploadFile] = File(...)):
    """Handle multiple file uploads, save them, and update the pipeline."""
    global file_uploaded
    try:
        uploaded_file_paths = []

        # Save all uploaded files to the local directory
        for file in files:
            if not file.filename.endswith(('.txt', '.pdf', '.docx')):
                raise HTTPException(status_code=400, detail="Unsupported file type. Please upload .txt, .pdf, or .docx files.")
            
            upload_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(upload_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_file_paths.append(upload_path)
        
        # Reload the pipeline with the uploaded files
        rag_pipeline.load_documents(input_dir=UPLOAD_DIR, input_files=uploaded_file_paths)
        rag_pipeline.split_documents()
        rag_pipeline.create_indices()
        rag_pipeline.create_query_engines()
        rag_pipeline.create_tools()
        rag_pipeline.create_router_engine()

        file_uploaded = True
        return {"message": "Files uploaded successfully", "file_paths": uploaded_file_paths}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Handle incoming queries and return the model's response."""
    global file_uploaded
    if not file_uploaded:
        print("No files uploaded. Loading default documents...")
        # Load default documents if no files are uploaded
        rag_pipeline.load_documents(input_dir='summaries')
        rag_pipeline.split_documents()
        rag_pipeline.create_indices()
        rag_pipeline.create_query_engines()
        rag_pipeline.create_tools()
        rag_pipeline.create_router_engine()

    query = request.question
    try:
        # Use the RAG pipeline to query the uploaded or default documents
        response = rag_pipeline.query(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)