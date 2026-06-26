from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

import os
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv

#Loading PDF files from directory
dir_loader = DirectoryLoader(
    r'',
    glob = '**/*.pdf' #Pattern to match PDF files
    loader_cls= = PyMuPDFLoader,
    use
)


#Text splitting into chunks

def split_documents(documents, chunk_size = 1000, chunk_overlap = 200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_funciton = len,
        separators=['\n\n','\n','',' ']
    )

    split_docs = text_splitter.split_documents(documents)

    return split_docs

chunks = split_documents(pdf_documents)


#Embedding Manager

class EmbeddingManager:
    def __init__(self, model_name:str = 'all-MiniLM-L6-v2'):
        #Initialize the embedding manager
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try :
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print('Error loading embedding manager model')
            raise
    
    def generate_embeddings(self, texts:List[str]) -> np.darray :
        #Generate embeddings for a lit of texts

        if not self.model :
            raise ValueError('Model not loaded')
        
        embeddings = self.model.encode(texts)

        return embeddings

#Initialize the embedding manager    
embedding_manager = EmbeddingManager()


#Creating Vector Store to store the vectors

class VectorStore:
    def __init__(self, collection_name : str = 'pdf_documents', persist_directory : str = '/tmp/vector_store'):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection_name = None
        self._initialize_store()

    def _intialize_store(self):
        #Initialize chromadb and client
        try :
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path = self.persist_directory)

            #Get or create collection
            self.collection = self.client.get_or_create_collection(
                self.collection_name,
                metadata={'description' : 'PDF document embeddings for RAG'}
            )

        except Exception as e:
            print('Error initializing vector store')
            raise

    def add_documents(self, documents : List[Any], embeddings:np.darray):
        #Add documents and their embeddings to the vector store

        if len(documents) != len(documents):
            raise ValueError('Number of documents must match the number of embeddings')
        
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f'doc_{uuid.uuid4().hex[:8]}_{i}'
            ids.append(doc_id)

            #Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

        try : 
            self.collection.add(
                ids = ids,
                embeddings = embeddings_list,
                metadatas = metadatas,
                documents = documents_text
            )
        
        except Exception as e:
            print(f'Error adding documents to vector store : {e}')
            raise

vectorstore = VectorStore()


#Convert the text to embeddings
texts=[doc.page_content for doc in chunks]

#Generate the embeddings
embeddings = embedding_manager.generate_embeddings(texts)

#Store in vector DB
vectorstore.add_documents(chunks, embeddings)


class RAGRetriever:
  #Handles query based retrieval from vector store
  def __init__(self, vector_store : VectorStore, embedding_manager : EmbeddingManager):
    #Initialize the retriever
    self.vector_store = vector_store
    self.embedding_manager = embedding_manager

  def retrieve(self, query : str, top_k: int = 5, score_threshold : float = 0.0) -> List[Dict[str,Any]]:
    #Retrieve relevant documents for a query
    print(f'Retrieving documents from query: {query}' )
    print(f'Top {top_k}, score threshold : {score_threshold}')

    #Generate query embedding
    query_embedding = self.embedding_manager.generate_embeddings([query])[0]

    #Search in vector store
    try :
      results = self.vector_store.collection.query(
          query_embeddings = [query_embedding.tolist()],
          n_results = top_k
      )

      #process results
      retrieved_docs = []

      if results['documents'] and results['documents'][0]:
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]

        for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
          similarity_score = 1- distance

          if similarity_score >= score_threshold:
            retrieved_docs.append({
                'id' : doc_id,
                'content' : document,
                'metadata' : metadata,
                'similarity_score' : similarity_score,
                'distance' : distance,
                'rank' : i+1
            })

        print(f'Retrieved {len(retrieved_docs)} documents after filtering')
      else :
        print('No documents found')

      return retrieved_docs

    except Exception as e:
      print(f'Error during retrieval : {e}')
      return []

rag_retriever = RAGRetriever(vectorstore, embedding_manager)



#Using a simple LLM to answer queries

load_dotenv()

# Initialize the Groq LLM (set GROQ_API_KEY  in environment)
groq_api_key = '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'

llm = ChatGroq(
    groq_api_key = groq_api_key, 
    model_name = 'llama-3.1-8b-instant', 
    temperature=0.1, 
    max_tokens = 1024)

#Simple RAG function : retrieve context + generate response

def rag_simple(query, retriever, llm, top_k =3):
    #Retrieve the context

    results =  retriever.retrieve(query, top_k = top_k)
    context = '\n\n'.join([doc['content'] for doc in results]) if results else ''
    if not context:
        return 'No relevant context found to answer the question'

    #Generate the answer using GroqLLM
    prompt = f"""Use the following context to answer the question precisely
    Context :{context}
    Question : {query}
    Answer :
    """

    #response = llm.invoke([prompt.format(context=context, query = query)])
    response = llm.invoke([prompt])
    return response
