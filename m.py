
import os
from PIL import Image
import asyncio

from supabase import create_client
import base64
from streamlit_mic_recorder import mic_recorder

import tempfile

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.retrieval import create_retrieval_chain

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains.combine_documents import create_stuff_documents_chain


from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

import tempfile
from langchain_community.document_loaders import TextLoader,CSVLoader


try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

from fuzzywuzzy import fuzz
import re

from datetime import datetime

import boto3
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time 
import hashlib


st.set_page_config(page_title="Ask Your Assistant", layout="wide")
load_dotenv(dotenv_path=".env")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
#supabase_url2=os.getenv("SUPABASE_URL2")

supabase = create_client(supabase_url, supabase_key)


BUCKET_NAME = "collegequestionpapers"
 # change if needed
AWS_REGION="ap-south-1"
AWS_ACCESS_KEY_ID=os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY=os.getenv("AWS_SECRET_ACCESS_KEY")

try:
    s3 = boto3.client(
    "s3",
    region_name="ap-south-1",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_KEY

)
except Exception as e:
    st.error("S3 initialization erro {e}")
    s3=None

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
asyncio.set_event_loop(asyncio.new_event_loop())
BASE_PATH = r"C:/Users/praja/Desktop/Sreamlit/myproject/data1/question_paper"


# Load .env (if present)
load_dotenv()

# Decorator

executor = ThreadPoolExecutor(max_workers=4)

# -----------------------
# Custom Cache Decorator
# -----------------------
def timed_cache(ttl_seconds=300):
    """Custom cache decorator with TTL"""
    cache = {}
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key
            key_parts = [str(arg) for arg in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
            key = hashlib.md5("".join(key_parts).encode()).hexdigest()
            
            now = time.time()
            
            # Check if cached and not expired
            if key in cache and now - cache[key]['timestamp'] < ttl_seconds:
                return cache[key]['value']
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = {'value': result, 'timestamp': now}
            
            # Clean old cache entries periodically
            if len(cache) > 100:  # Limit cache size
                for k in list(cache.keys()):
                    if now - cache[k]['timestamp'] > ttl_seconds:
                        del cache[k]
            
            return result
        return wrapper
    return decorator






# -----------------------
# Utilities
# -----------------------
pdf_keywords = [
        "question paper", "previous year", "last year",
        "sample paper", "model paper", "pyq",
        "prev paper", "exam paper", "old paper",
        "previous paper", "last sem paper", "pdf"
    ]

@st.cache_resource(ttl=3600)
def get_vectorstore():
    """Load vectorstore from Supabase with proper configuration."""
    try:
        # Embedding model
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        supabase_client = supabase

        # Create vector store
        vectorstore = SupabaseVectorStore(
            client=supabase_client,
            embedding=embedding,
            table_name="college_vectors",
            query_name="match_documents"
        )

        return vectorstore

    except Exception as e:
        print("Vectorstore error:", e)
        return None

@timed_cache(ttl_seconds=600)
def detect_intent(text: str) -> str:
    """Return 'pdf', 'college', or 'general'."""
    if not text:
        return "general"
    q = text.lower()

    pdf_keywords = [
        "question paper", "previous year", "last year",
        "sample paper", "model paper", "pyq",
        "prev paper", "exam paper", "old paper",
        "previous paper", "last sem paper", "pdf"
    ]
    
    college_keywords =[
    # Basic info
    "college", "institute", "campus", "location", "address",

    # Fees & money
    "fees", "fee", "scholarship", "refund", "admission fee",
    "hostel fee", "exam fee", "transport fee",

    # Admission
    "admission", "apply", "application", "eligibility",
    "documents", "counselling", "registration",
    "entrance", "cutoff", "seat", "seat allotment",

    # Courses & academics
    "course", "courses", "program", "degree", "stream",
    "department", "faculty", "teacher", "hod",
    "syllabus", "curriculum", "subjects", "module",
    "semester", "exam", "examination", "paper pattern",

    # Student life
    "hostel", "mess", "food", "canteen", "wifi", "library",
    "sports", "lab", "computer lab", "classroom",
    "attendance", "uniform", "rules", "timing",

    # Infrastructure
    "building", "labs", "transport", "bus", "parking",
    "medical", "security",

    # Placement & internship
    "placement", "placements", "package", "internship",
    "companies", "recruiters", "average package",

    # Intake & seats
    "intake", "capacity", "strength",

    # Contact details
    "phone", "email", "contact", "helpline",
    "office", "principal", "director",

    # Events & activities
    "fest", "events", "seminar", "workshop", "orientation"
     ]                                                                                                                    
    generator_keywords = [
    # Definitions
    "define", "definition of", "what is", "explain", "describe","introduction"

    # Exam
    "important questions", "exam questions", "short answer", "long answer",
    "2 marks", "5 marks", "10 marks", "previous year questions",

    # Comparison
    "difference between", "compare", "vs", "advantages and disadvantages",

    # Lists
    "list", "list out", "types of", "features of", "applications of",

    # Examples & code
    "example", "with example", "write a program", "python program", "syntax of",

    # Study intent
    "easy explanation", "simple words", "notes on", "summary of"
]


    if any(kw in q for kw in pdf_keywords):
        return "pdf"
    if any(kw in q for kw in college_keywords):
        return "college"
    #if any(kw in q for kw in generator_keywords):
    return "generator"
   


# return None          
import re


@timed_cache(ttl_seconds=600)
def extract_course(q):
    if "bca" in q:
        return "bca"
    elif "bba" in q:
        return "bba"
    elif "tally" in q:
        return "tally"
    elif "o level" in q or "olevel" in q:
        return "o level"
    return None

    

@timed_cache(ttl_seconds=600)
def find_question_paper(prompt: str):
    """
    Search for question papers in Supabase based on user query
    Returns matching paper or None
    """
    if not prompt or not prompt.strip():
        return None

    q = prompt.lower()
    
    # Extract course
    course = None
    if "bca" in q:
        course = "bca"
    elif "bba" in q:
        course = "bba"
    elif "tally" in q:
        course = "tally"
    elif "o level" in q or "olevel" in q:
        course = "o level"
    
    # Extract semester - look for patterns like "2 semester", "2nd sem", etc.
    sem_patterns = [
        r'(\d+)(?:st|nd|rd|th)?\s*(?:sem|semester)',
        r'sem(?:ester)?\s*(\d+)',
        r'(\d)\s*(?:sem|semester)'
    ]
    
    semester = None
    for pattern in sem_patterns:
        match = re.search(pattern, q)
        if match:
            semester = match.group(1)
            break
    
    # If no pattern matched, try simple digit extraction
    if not semester:
        sem_match = re.search(r'\b([1-8])\b', q)
        semester = sem_match.group(1) if sem_match else None
    
    # Extract year
    year_match = re.search(r'\b(20\d{2})\b', q)
    year = year_match.group(1) if year_match else None
    
    # Extract subject - remove common words
    subject = q
    common_words = ["pdf", "question", "paper", "previous", "year", "sem", "semester", 
                   "give", "show", "find", "me", "need", "want", "download", "get", "bca", "bba"]
    
    # Remove course name if found
    if course:
        subject = subject.replace(course, "")
    
    # Remove common words
    for word in common_words:
        subject = subject.replace(word, "")
    
    # Clean up extra spaces
    subject = re.sub(r'\b\d\b', '', subject)
    subject = re.sub(r'\s+', ' ', subject).strip()
    
    # Print debug info
    #print(f"Searching for - Course: {course}, Semester: {semester}, Year: {year}, Subject: {subject}")
    
    try:
        # Query Supabase
        query = supabase.table("question_papers").select("*")
        
        # Apply filters if available
        if course:
            query = query.eq("course", course)
        if semester:
            query = query.eq("semester", int(semester))
            
        # Don't filter by year initially to get all years for this course/semester
        response = query.execute()
        papers = response.data
        
        if not papers:
            # Try a more flexible search without filters
            papers = supabase.table("question_papers").select("*").execute().data
        
        if not papers:
            return None
            
        # Score and find best match
        best_match = None
        best_score = 0
        exact_year_match = None
        year_matches = []
        
        for paper in papers:
            score = 0
            
            # Course match (highest weight)
            if course and paper.get("course", "").lower() == course:
                score += 40
                
            # Semester match
            if semester and str(paper.get("semester", "")) == semester:
                score += 30
                
            # Year match - give higher score for exact year match
            paper_year = str(paper.get("year", ""))
            if year and paper_year == year:
                score += 40  # Increased weight for exact year match
                exact_year_match = paper
            elif year and paper_year:
                # If not exact match but year exists
                score += 5
                
            # Subject fuzzy match
            paper_subject = paper.get("subject", "").lower()
            if subject and paper_subject:
                subject_score = fuzz.partial_ratio(subject, paper_subject)
                score += subject_score * 0.7  # Increased weight to 70%
            
            # Document type boost for question papers
            if paper.get("doc_type", "").lower() in ["question paper", "question_paper"]:
                score += 20
            
            # Print debug info for each paper
            print(f"Paper: {paper.get('subject')} ({paper.get('year')}) - Score: {score}")
            
            # If we have an exact year match with good subject match, prioritize it
            if exact_year_match and subject:
                paper_subject = exact_year_match.get("subject", "").lower()
                if fuzz.partial_ratio(subject, paper_subject) > 60:
                    return exact_year_match
            
            # Track year matches separately for same subject
            if semester and course and paper_subject:
                if fuzz.partial_ratio(subject, paper_subject) > 70:
                    year_matches.append({
                        'paper': paper,
                        'year': paper_year,
                        'subject_score': fuzz.partial_ratio(subject, paper_subject)
                    })
            
            
            if score > best_score and score > 20:  # Reduced threshold from 30 to 20
                best_score = score
                best_match = paper
        
  
        if len(year_matches) > 1:
 
            if year:
                for match in year_matches:
                    if match['year'] == year:
                        return match['paper']
        
        return best_match
        
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None



    

@timed_cache(ttl_seconds=600)
def creat_chunks(extract_chunks): 
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=50)
    text_chunks=text_spliter.split_documents(extract_chunks)
    return text_chunks


embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
#@timed_cache(ttl_seconds=600)
def Creat_Embedding(chunks): 

    vectors=[]

    for doc in chunks:
        embedding = embedding_model.embed_query(doc.page_content)

        vectors.append({"content": doc.page_content,
            "metadata": {"source": "upload"},
            "embedding": embedding})
    return vectors

# Store vectore into the vectore data base 

@timed_cache(ttl_seconds=600)
def vectore_database(verctors): 

    supabase.table("college_vectors").insert(verctors).execute()

@timed_cache(ttl_seconds=600)
def search_vectors(query_embedding):

    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.4,
            "match_count": 1
        }
    ).execute()

    return response.data 
                                                                                         

@timed_cache(ttl_seconds=600)
def build_rag_chain(llm):
    """Return a retrieval-augmented chain (or None) if vectorstore exists."""
    db = get_vectorstore()
    if db is None:
        return None

    # simple prompt for combining retrieved docs
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([                # "You are an expert tutor. Use retrieved context to answer clearly.""Give a short and direct answer only"
        ("system",  "You are a college academic assistant."
        "You must answer primarily using the provided CONTEXT extracted from PDFs. "
        "Always look into the context first and base your answer on it. "
        "Do NOT ignore the context. "
        "If the answer is clearly found in the context, use only that information. "
        "If the context has partial information or not available, expand slightly but keep it aligned with the context. "
        "Only if the context does NOT contain the answer at all, then you may use general acadmic knowledge. "
        "If the user asks for a list or questions, generate the COMPLETE list as requested. "
        "Never stop early. Number each point clearly. "
        "If the answer is not available anywhere, clearly say: 'This information is not available in the provided documents.'"),
        ("human", "Question: {input}\n\nContext:\n{context}")
    ])
    combiner_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k": 3}), combiner_docs_chain)
    return rag_chain

@st.cache_resource(ttl=3600)
def get_llm():
    """
    Return a configured LLM instance or None.
    Reads GROQ_API_KEY from environment (.env recommended).
    """
    
    api_key = os.getenv('GROQ_API_KEY')

    model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
    if not api_key or ChatGroq is None:
        return None
    try:
        llm = ChatGroq(model=model_name, temperature=0.5, api_key=api_key, max_tokens=800)
        return llm
    except Exception as e:
        st.warning(f"Could not initialize LLM: {e}")
 


def main():
  
    if "staff_mode" not in st.session_state:
        st.session_state.staff_mode = False
    if "upload_done" not in st.session_state:
        st.session_state.upload_done = False
    if "staff_auth" not in st.session_state:
        st.session_state.staff_auth = False
    

    GENERATOR_PROMPT = """
You are an expert BCA teacher.

Rules:
- You may use general knowledge
- Explain clearly for exams
- Use structured points
- Give examples where helpful
- Complete the answer fully

Question:
{question}

""" 
    if "all_chats" not in st.session_state:
        st.session_state.all_chats = {}
    if "all_chat" not in st.session_state:
        st.session_state.all_chat={}
    if "current_id" not in st.session_state:
        st.session_state.current_id=str(uuid.uuid4())
    
    chat_id = st.session_state.current_id

    if chat_id not in st.session_state.all_chat:
        st.session_state.all_chat[chat_id] = []

    if "messages" not in st.session_state:
        st.session_state.messages=[]

    # st.set_page_config(page_title="Ask Your Assistant", layout="wide")
    st.markdown("""
<style>

 /* Make ALL assistant message text black */
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] div {
    color: #000000 !important;
}

/* Make ALL user messages black */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {
    color: #000000 !important;
}

/* Force Markdown text everywhere to black */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li {
    color: #000000 !important;
}

/* Optional: headings also black */
h1, h2, h3, h4, h5, h6 {
    color: #000 !important;
}

</style>
""", unsafe_allow_html=True)
    
    st.markdown("""
<style>

.bottom-black-strip {
    position: fixed;
    bottom: 0;
    left: 1;
    width: 77%;
    height: 0.8cm;
    background:0;
    z-index: 999999;

    display: flex;
    justify-content: center;
    align-items: center;
}

.bottom-black-strip p {
    color: white;
    font-size: 14px;
    margin: 0;
}

</style>

<div class="bottom-black-strip">
    <p>This Chatboat can make mistake please !Chek Important Information </p>
</div>
""", unsafe_allow_html=True)



    st.markdown("""
<style>

/* Make assistant message wider */
[data-testid="stChatMessage"] {
    max-width: 100% !important;
}

/* Increase font size inside assistant messages */
.assistant-text {
    font-size: 20px !important;
    line-height: 1.6;
    color: black !important;
}

</style>
""", unsafe_allow_html=True)



  

    def get_base64(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    img = get_base64(r"C:\Users\praja\Desktop\MYCOLLEGECHATBOAT\data\adrian-infernus-GLf7bAwCdYg-unsplash.jpg")
    
    st.markdown(
    f"""
    <style>

    /*---------------------------------------------------------
        REMOVE WHITE SPACE EVERYWHERE
    ----------------------------------------------------------*/

    /* Remove Streamlit header */
    header, [data-testid="stHeader"] {{
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
    }}

    /* Remove top decoration */
    [data-testid="stDecoration"] {{
        display: none !important;
    }}

    /* Remove main padding */
    .block-container {{
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* Remove bottom block completely */
    [data-testid="stBottomBlock"] {{
        display: none !important;
        height: 0 !important;
        background: transparent !important;
    }}

    /* Remove Streamlit footer (this is your remaining white bar) */
    footer {{
        visibility: hidden;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* Force the background on entire app area */
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/jpg;base64,{img}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    html, body {{
        height: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }}

    /*---------------------------------------------------------
        CHAT INPUT FIXED AT BOTTOM
    ----------------------------------------------------------*/

    [data-testid="stChatInput"] {{
        position: fixed !important;
        bottom: 25px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;

        width: 90% !important;
        max-width: 1200px;

        background: rgba(255,255,255,0.30) !important;
        backdrop-filter: blur(12px);
        border-radius: 15px !important;
        padding: 20px !important;
        border: 1px solid rgba(255,255,255,0.5);
        z-index: 99999 !important;
    }}

    [data-testid="stChatInputTextarea"] textarea {{
        color: white !important;
        background: transparent !important;
        font-size: 18px !important;
    }}

    /* Prevent messages from going under chat input */
    .block-container {{
        padding-bottom: 270px !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

#    def main():
 



   
    def load_image_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    header_img = load_image_base64(r"C:\Users\praja\Desktop\MYCOLLEGECHATBOAT\data\Screenshot 2025-11-26 231110.png")
    st.markdown(
    f"""
    <style>

    .top-banner {{
        position: fixed;
        top: 0;
        left: 1;
        right: 1;
        display: flex;
        justify-content: center;
        align-items: top;
        padding: 10px 0;
        background: rgba(0,0,0,0);
        backdrop-filter: blur(1px);
        z-index: 99999;
        border-bottom-left-radius: 10px;
        border-bottom-right-radius: 10px;
        box-shadow: 0 4px 14px rgba(0,0,0,0);
    }}

    .top-banner img {{
        width: 100%;              /* wider */
        max-width: 1000px;        /* more max width */
        height: 50px;           /* longer */
        border-radius: 25px;
        object-fit: cover;
        border: 2px solid rgba(255,255,255,0.5);
    }}

    .block-container {{
        padding-top: 200px !important;   /* increase if image gets taller */
    }}

    </style>

    <div class="top-banner">
        <img src="data:image/png;base64,{header_img}">
    </div>
    """,
    unsafe_allow_html=True
) 
    #Fixed chat input box
    st.markdown("""
<style>

/* Fixed chat input bar at bottom but centered */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 20px !important;
    left: 58% !important;
    transform: translateX(-50%) !important;

    width: 70% !important;        /* full width but with side space */
    max-width: 900px !important;  /* keeps it neat on big screens */

    z-index: 999999 !important;

    padding: 12px 20px !important;
    background: rgba(255,255,255,0.10) !important;
    backdrop-filter: blur(15px);
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.4);
}

/* Avoid messages hiding behind the input */
[data-testid="stAppViewContainer"] > .main {
    padding-bottom: 200px !important;
}

</style>
""", unsafe_allow_html=True)
    

    
 # image in side the sidebar
    
    

# CSS for gradient sidebar
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"] > div:first-child {
        padding: 0 !important;
        margin: 0 !important;
        /* Gradient background: sky blue to pink */
        background: linear-gradient(to bottom, #87CEEB, #F8C8F8) !important;
        height: 100vh;  /* full height */
    }

    [data-testid="stSidebar"] {
        color: black;  /* text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)
     
    st.markdown("""
<style>
.sidebar-title {
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    color: #4A90E2;      /* Beautiful blue */
    padding-top: 10px;
    padding-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)
    
    
    def instruction():
        with st.expander("How to Use this Assitant",expanded=False):
            st.markdown("""**Start a Conversation**  
Type your question in the chat box. You don’t need special formatting. Just write naturally and the assistant will understand and respond.

**What You Can Ask**  
You can ask about programming, machine learning, databases, networking, security, or general concepts. You can also request summaries, notes, definitions, or step-by-step help.

**Working With Code**  
If you’re stuck, paste your code and ask what’s wrong or how to improve it. The assistant can debug, explain errors, and suggest fixes.

**Using Your Documents**  
If your setup uses RAG or uploaded files, the assistant can answer using your content. Try to keep your questions focused for the best results.

**Follow-Up Questions**  
The chat remembers the conversation during the session. You can say “explain more” or “give an example” without repeating the original question.

**Try Different Prompts**  
Ask for simple or advanced explanations, examples, comparisons, or code samples. The assistant adjusts based on what you need.

**If the Answer Isn’t Right**  
Reply with what you want changed. The assistant improves the answer based on your feedback.

**Stay Organized**  
Your messages and replies appear clearly in the chat. Scroll to view older answers. Refreshing resets the session, so save anything important.""")
            

    def uploadfile():
        vectors=[]
        (tab1,tab2)=st.sidebar.tabs(["Admin_Upload","Internal Info"])
        with tab1:
            st.sidebar.header("Upload PDF File")
            uploaded_file=st.sidebar.file_uploader("Upoad PDF",type=["pdf","png","jpg"])
            col1, col2, col3, col4,col5 = st.sidebar.columns(5)
            with col1:
                course = st.sidebar.selectbox("Course", ["BCA", "BBA", "O Level","CCC","Tally","BSc","BA"])
            with col2:
                semester=st.sidebar.selectbox("Semester",[1,2,3,4,5,6])
            with col3: 
                subject=st.sidebar.text_input("Subject")
            with col4:
                year = st.sidebar.selectbox("Year", ["2021", "2022", "2023", "2024", "2025","2026"])
            with col5:
                doc_type= st.sidebar.selectbox("Course2", ["Question Paper","Notice","Syllabus","Circular"])
        
        if st.button("Upload"):
            if uploaded_file and subject:
                unique_id = str(uuid.uuid4())
                file_name = f"{year}/{semester}/{unique_id}.pdf"
                s3.upload_fileobj(uploaded_file,BUCKET_NAME,file_name)

                 # ✅ Step 3: Generate S3 file URL
                file_url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{file_name}"

                # ✅ Step 4: Insert metadata into PostgreSQL (Supabase)
                supabase.table("question_papers").insert({
            "subject": subject,
            "semester": semester,
            "course":course,
            "year": year,
            "doc_type": doc_type,
            "file_url": file_url,
            "s3_key": file_name,
            "uploaded_at": datetime.utcnow().isoformat() }).execute()
                

                st.success("Uploaded and saved successfully")
            else:
                st.error("Please upload file and fill all fields")
                if not subject:
                    st.error("Please Enter Subject")
        

        with tab2:
            st.sidebar.header("Internal Information ")
            uploaded_file=st.sidebar.file_uploader("Upload file here",type=["pdf","txt","csv"])
            if uploaded_file is not None:


               with tempfile.NamedTemporaryFile(delete=False) as tmp:
                  tmp.write(uploaded_file.read())
                  file_path = tmp.name
                  if uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(file_path)

                  elif uploaded_file.type == "text/plain":
                    loader = TextLoader(file_path)
                  else:
                    loader = CSVLoader(file_path)
                  documents = loader.load()
                  text_spliter=RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=50)
                  text_chunks=text_spliter.split_documents(documents)

                  embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                  

                  for doc in text_chunks:
                     embedding = embedding_model.embed_query(doc.page_content)
                     vectors.append({"content": doc.page_content,
            "metadata": {"source": "upload"},
            "embedding": embedding})

             
             
        if st.button("Uploadvector"):
            supabase.table("college_vectors").insert(vectors).execute()
            st.success("Data stored in Supabase vector database")


            

        

        
        

            


    passwordfile=r"C:\Users\praja\Desktop\MYCOLLEGECHATBOAT\data\PASSWORD_FILE.txt"

    def load_password():

        if not os.path.exists(passwordfile): 
            with open(passwordfile,"w") as f:
                f.write("staff@staff")
        
        with open(passwordfile,'r') as f: 

            return f.read().strip()
            
    def Uplaod_password(newpasswrod):
        with open(passwordfile,"w") as f:
            return f.write(newpasswrod)
        

    
    # Login PAGE


    user = supabase.auth.get_user()

    if user:
      st.session_state.logged_in = True
      st.session_state.user_id = user.user.id
      st.session_state.user_email = user.user.email

    def login_user(email, password):
        res = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })
        return res

    if "logged_in" not in st.session_state:
      st.session_state.logged_in = False

    if "user_id" not in st.session_state:
      st.session_state.user_id = None

    if "user_email" not in st.session_state:
       st.session_state.user_email = None

    if "show_login" not in st.session_state: 
        st.session_state.show_login=None

    st.sidebar.write('.')
    
    st.sidebar.write('.')


    # ---------------- AUTO LOGIN (PERSIST SESSION) ---------------- #
    user = supabase.auth.get_user()

    if user and user.user:
       st.session_state.logged_in = True
       st.session_state.user_id = user.user.id
       st.session_state.user_email = user.user.email
    

    # Register user

    def register_user(email, password, name, ID, course):
        auth_res = supabase.auth.sign_up({"email": email,"password": password})

        if auth_res.user:
            supabase.table("profiles").insert({
            "id": auth_res.user.id,
            "name": name,
            "roll_id": ID,
            "course": course}).execute()
            return True
        return False
    
    def login_user(email, password):
        return supabase.auth.sign_in_with_password({
        "email": email,
        "password": password})
    
    def logout_user():
        supabase.auth.sign_out()
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.user_email = None
        st.rerun()






    if st.sidebar.button("### 👤 Admin ",key="log2", type="tertiary"): 
           
           st.session_state.show_login=True

    if st.session_state.show_login:


        with st.sidebar:

            if not st.session_state.logged_in:
             c1, c2,c3= st.tabs(["Register","Login","Logout"])

             with c1: 

               with st.form("register_form"):

                 name=st.text_input("Enter your Name ")

                 email=st.text_input("Enter Email ")

                 password=st.text_input("Enter Password",type="password")

                 ID=st.text_input("Enter your roll ID",placeholder="1AB2C3")

                 course=st.text_input("Enter your roll Course",placeholder="Ex.. BCA,BBA Olevel")


                 regbutton=st.form_submit_button("Register")

                 if regbutton:
                   if not all([name,email,password,ID,course]): 
                     st.error("Please Fill all the field")

                   else:
                       try: 
                          if register_user(email,password,name,ID,course):

                              st.success("✅ Registration successful")
                          else:
                            st.success("✅ Registration successful")
                     
                       except Exception as e:
                          st.error (e)

                    
                     
                     
             with c2 : 

                 with st.form("login_form"):
                    email = st.text_input("Enter Email")
                    password = st.text_input("Enter Password", type="password")

                    login_btn = st.form_submit_button("Login")

                    if login_btn:

                        if not email or not password:

                            st.error("Please Enter Email and Password")

                        try:
                            res=login_user(email,password)

                            if res.user:
                                st.session_state.logged_in = True
                                st.session_state.user_id = res.user.id
                                st.session_state.user_email = res.user.email

                                st.success(f" Welcome {res.user.email} 👋")
                                st.rerun()
                         
                        except Exception as e: 
                             st.error("❌ Not registered or wrong credentials")
             with c3:
                 st.success(f"✅ Logged in as {st.session_state.user_email}")
                 st.button("Logout", on_click=logout_user)


    # Load to the chat history


    def load_chat_history(user_id):
        res = (
        supabase
        .table("chat_history")
        .select("id,role, message, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=False)
        .execute()
    )
        return res.data
    

    def load_conversation_from_question(user_id, question_id):
        """Load a conversation (question + answer) by question ID"""
        try:
        # Get all messages for this user
           result = (
            supabase
            .table("chat_history")
            .select("id, role, message, created_at")
            .eq("user_id", user_id)
            .order("created_at", asc=True)
            .execute()
        )
        
           if not result.data:
            return []
        
        # Find the question and its answer
           conversation = []
           found_question = False
        
           for msg in result.data:
             if msg["id"] == question_id:
                conversation.append(msg)
                found_question = True
             elif found_question and len(conversation) == 1:
                # This should be the assistant's reply
                conversation.append(msg)
                break
        
             return conversation
        
        except Exception as e:
           st.error(f"Error loading conversation: {str(e)}")
           return []






                    

   
        

        
    with st.sidebar:
       if "show_history"not in st.session_state: 
         st.session_state.show_history=None
       if "selected_question_id" not in st.session_state:
               
               st.session_state.selected_question_id=None
       

       


       if st.sidebar.button("### 🆕 New Chat", type="tertiary"):
           st.session_state.messages=[]
           st.rerun()
           main()
       st.sidebar.button("### 🔍  Search Chat",key="search", type="tertiary")
       btn=st.sidebar.button("###   📘 How to use it ",key="Tem", type="tertiary")
       if st.sidebar.button("### ℹ️   Your Chat", key="hist", type="tertiary"):
           st.session_state.show_history = True
               
           if st.session_state.logged_in and st.session_state.user_id:
                hist=load_chat_history(st.session_state.user_id)

                if not hist:
                    st.info("Still There is no history")

                else: 
                    for i in hist: 
                        title = i["message"][:40] + "..." if len(i["message"]) > 40 else i["message"]
                        if st.sidebar.button(title, key=i["id"]):
                               st.session_state.selected_question_id = i["id"]
                               
                    
             
       


       
       #st.sidebar.button("### 📚  Library",key="lib", type="tertiary")
       #STAFF_PASSWORD = "msitm@staff"
       staff_btn=st.sidebar.button("### 🎓 College Staff ",key="Staff", type="tertiary")
       #password_change=st.sidebar.button("### 🎓 ChangePassword",key="password", type="tertiary")
       if staff_btn:
           st.session_state.show_staff_login = True
           st.session_state.staff_mode = True
           st.session_state.upload_done = False
       if st.session_state.get("show_staff_login") and not st.session_state.staff_auth:
           st.sidebar.subheader("🔐 Staff Login")
           #st.sidebar.subheader("🔐 Change Password")

           #staff_password = st.sidebar.text_input("Enter staff password",type="password" )

           with st.sidebar: 
               stored_password=load_password()
            #    staff_password = st.sidebar.text_input("Enter staff password",type="password" )
               col1, col2 = st.tabs(["Login", "Change Password"])

               with col1: 
                 
                 staff_password = st.sidebar.text_input("Enter staff password",type="password" )
                     
                 if st.button("Login"):
                    
                   
                    if staff_password == stored_password:
                      st.session_state.staff_auth = True
                      st.sidebar.success("Access granted")
                    else:
                         st.sidebar.error("Wrong password")
               with col2:
             
                 old=st.text_input("Enter your old Password !",type='password')
                 newpassword=st.text_input("Enter your New Passwrod !",type='password')
                 confirm_password=st.text_input("Enter confirm Password ",type='password')
                 
                 if st.button("Update"):
                     if old!=stored_password:
                       st.error("Old Password is wrong")

                     elif newpassword!=confirm_password:
                       st.error("Passwords do not match")
                     elif len(newpassword)<6:
                       st.error("Password must be at least 4 characters")

                     else: 
                       Uplaod_password(newpassword)
                       st.success("Password updated successfully")


                             
       if btn:
           instruction()
       if st.session_state.staff_auth:
        if st.session_state.staff_mode:
           uploadfile()
    


       # animated video

       # Your video URL
   # Msitm logo

       def load_image_base64(path):
         with open(path, "rb") as f:
           return base64.b64encode(f.read()).decode()

       icon_img = load_image_base64(r"C:\Users\praja\Desktop\MYCOLLEGECHATBOAT\data\msitmlogo2-removebg-preview.png")

       st.markdown(f"""
            <style>
    .top-icon {{
        position: fixed;
        top: 13px;
        left:4%;
        transform: translateX(-50%);
        
        width: 80px;     
        height: 80px;    
        border-radius: 50%;
        overflow: hidden;
        
        border: 3px solid rgba(255,255,255,0.7);
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
        z-index: 9999;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
    }}

    .top-icon img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}
    </style>

    <div class="top-icon">
        <img src="data:image/png;base64,{icon_img}">
    </div>
    """,
    unsafe_allow_html=True)
       
# Cboat logo

       def load_image_base64(path):
           with open(path, "rb") as f:
               return base64.b64encode(f.read()).decode()

       msitmlogo= load_image_base64(r"C:\Users\praja\Desktop\MYCOLLEGECHATBOAT\data\MSITM-Cboat_logo_design2-removebg-preview.png")
       
       st.markdown(f"""
<style>
/* HARD RESET for logo */
.top-icon2 {{
    position: fixed;
    top: 40px;
    left: 17%;
    transform: translateX(-50%);

    width: 140px;
    height: auto;

    border-radius: 0 !important;
    overflow: visible !important;

    background: transparent !important;
    box-shadow: none !important;
    z-index: 9999;
}}

.top-icon2 img {{
    width: 100%;
    height: auto;

    border-radius: 0 !important;   /* 🔴 THIS FIXES ELLIPSE */
    object-fit: contain !important;
    background: transparent !important;
}}
</style>

<div class="top-icon2">
    <img src="data:image/png;base64,{msitmlogo}">
</div>
""", unsafe_allow_html=True)

       
       st.markdown("""
<style>
/* Fixed sidebar header */
.sidebar-title {
    position: fixed;
    top: 0;
    left: 0;
    width: 21rem;   /* default Streamlit sidebar width */
    background: #121212;
    color: black;
    font-size: 22px;
    font-weight: 600;
    padding: 16px;
    z-index: 1000;
    border-bottom: 1px solid rgba(255,255,255,0.2);
    text-align: center;
}

/* Push sidebar content below the fixed title */
section[data-testid="stSidebar"] > div {
    padding-top: 70px;
}
</style>
""", unsafe_allow_html=True)
       

      
      

    def save_chat(user_id, role, message):
        supabase.table("chat_history").insert({
        "user_id": user_id,
        "role": role,
        "message": message
    }).execute()
       


    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # render previous conversation
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
    if "voice_text" not in st.session_state:
        st.session_state.voice_text = None
    col1, col2 = st.columns([8, 1])

    with col1:
      prompt = st.chat_input("Pass your question here")
      if not prompt:
        return

    # echo user message
      st.chat_message("user").markdown(prompt)


      save_chat(
        user_id=st.session_state.user_id,
        role="user",
        message=prompt)
    
      st.session_state.messages.append({"role": "user", "content": prompt})
      st.session_state.all_chat[chat_id].append({"role": "*", "text": prompt})

    if (st.session_state.selected_question_id and st.session_state.logged_in):
           conversation = load_conversation_from_question(
                                             st.session_state.user_id,
                                             st.session_state.selected_question_id
                                             )
           if conversation:
                for msg in conversation:
                    st.chat_message(msg["role"]).markdown(msg["message"])






    
    try:
        intent = detect_intent(prompt)

        

        # --- handle PDF intent ---
        if intent in ["pdf", "question_paper", "document","previous year","past year"]:
            with st.spinner("Searching for question paper"):
                paper = find_question_paper(prompt)
                if paper:
                #url=paper['file_url']
                   url=paper.get('file_url')
                   if url: 
                       if 's3.us-east-1.amazonaws.com' in url:
                           url = url.replace('s3.us-east-1.amazonaws.com', 's3.ap-south-1.amazonaws.com')
                       st.success(f"✅ Found: {paper.get('subject', 'Question Paper')} - {paper.get('year', '')}")
                       st.markdown("### 📄 Question Paper Preview")

                       viewer_url = f"https://docs.google.com/viewer?url={url}&embedded=true"

                        
                       viewer_url = f"https://docs.google.com/viewer?url={url}&embedded=true"
                    
                    
                       viewer_html = f'''
                        <iframe src="{viewer_url}" 
                                width="70%" 
                                height="500px" 
                                style="border: none;"
                                allowfullscreen>
                        </iframe>
                    '''
                     
                      
                       st.components.v1.iframe(viewer_url, width=560, height=360, scrolling=True)

                       
                       st.markdown("""
<style>
.download-btn{
    display:inline-block;
    padding:12px 24px;
    font-size:18px;
    color:white !important;
    background:linear-gradient(90deg,#4CAF50,#2ecc71);
    border-radius:10px;
    text-decoration:none;
    font-weight:600;
    transition:0.3s;
}
.download-btn:hover{
    transform:scale(1.05);
    background:linear-gradient(90deg,#2ecc71,#27ae60);
}
</style>
""", unsafe_allow_html=True)

                       st.markdown(
    f'<a href="{url}" download class="download-btn">⬇ Download PDF</a>',
    unsafe_allow_html=True
)
                
                
                
                
                else:
                  st.error("PDF not found.")
                return
         
        # --- handle knowledge/LLM intents ---
        llm = get_llm()
        
        if llm is None:
            st.info("LLM not configured. Set GROQ_API_KEY in your environment to enable QA. Returning fallback message.")
            # fallback behavior: if no LLM, try to show a helpful local response
            st.chat_message("assistant").markdown("I can't run the LLM because the GROQ API key is not configured. Please set GROQ_API_KEY in your .env and restart.")
            st.session_state.messages.append({"role": "assistant", "content": "LLM not configured."})
            return

        # Build RAG chain once
        rag_chain = build_rag_chain(llm)
        if rag_chain is None:
            st.warning("Vectorstore not available — answers may be generic ")
        # Use college-specific prompt if intent==college
        if intent == "college" and rag_chain:
            response = rag_chain.invoke({"input": prompt})

            if isinstance(response, dict):
                answer = response.get("answer") or response.get("result") or response.get("output")
                #answer = answer.split("\n")[0]
            else:
               answer = str(response)

            if not answer:
               answer = "I couldn't find a specific answer."
            
           
            save_chat(user_id=st.session_state.user_id,role="assistant", message=answer)

            st.chat_message("assistant").markdown(f'<span style="color:black;">{answer}</span>',unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            return

        # Default general RAG
        if intent=="generator":
            #response = rag_chain.invoke({"input": prompt})\\

            answer = llm.invoke(GENERATOR_PROMPT.format(question=prompt)).content
            st.chat_message("assistant" ).markdown(f'<span style="color:black;">{answer}</span>',unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            save_chat(user_id=st.session_state.user_id,role="assistant",message=answer)
            
        else:
            # If no rag_chain but LLM exists, run simple LLM-only response (optional)
            st.chat_message("assistant").markdown("No retriever available. Please ensure vectorstore exists.")
            st.session_state.messages.append({"role": "assistant", "content": "No retriever answer available."})
            st.session_state.all_chats[chat_id].append({"role": "assistant", "text": bot_reply})


    except Exception as exc:
        st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()


