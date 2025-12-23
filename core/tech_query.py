def process_tech_query(query, state, law_type, chat_history):

    from datetime import datetime
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_openai import AzureChatOpenAI
    from langchain.prompts import PromptTemplate

    from dotenv import load_dotenv
    from pathlib import Path
    import os

    ROOT_DIR = Path(__file__).resolve().parents[2]  # Labour Laws Query API
    load_dotenv(ROOT_DIR / ".env")


    API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    # API_KEY = "⚠️ MOVE_TO_ENV"
    # AZURE_ENDPOINT = "https://tml-az-dev-ca-aac-openai.openai.azure.com/"
    
    os.environ["AZURE_OPENAI_API_KEY"] = API_KEY
    os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_ENDPOINT
    os.environ["AZURE_OPENAI_API_VERSION"] = API_VERSION

    INDEX_FOLDER_DIC = {
        "Central" : "unified_central_index",
        "Maharashtra" : "unified_marathi_index",
        "Karnataka" : "unified_Kannada_index",
        "Uttar Pradesh" : "unified_up_index",
        "Uttrakhand" : "unified_uk_index",
        "Gujarat" : "unified_gujarat_index",
        "Jharkhand" : "unified_jha_index"
    }


    # 1. Load Index (LOCAL embeddings)
    INDEX_FOLDER = "core/unified_index_state/"+INDEX_FOLDER_DIC[state]
    embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
 
    vector_store = FAISS.load_local(
        INDEX_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = AzureChatOpenAI(
        azure_deployment=CHAT_DEPLOYMENT,
        openai_api_version=API_VERSION,
        temperature=0
    )
 
   
    def check_headers_ascii(llm):
        try:
            # OpenAI 1.x client usually stores httpx client here
            headers = getattr(getattr(llm, "client", None), "_client", None)
            if headers is not None:
                headers = headers.headers  # httpx.Headers
                for k, v in headers.items():
                    try:
                        v.encode("ascii")
                    except UnicodeEncodeError:
                        print("🚫 Non-ASCII header detected:", k, repr(v))
            else:
                print("Could not access underlying httpx client headers.")
        except Exception as e:
            print("Header check failed:", e)
    check_headers_ascii(llm)
 
 
    def format_docs_with_citation(docs):
        formatted_text = ""
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown File')
            page = doc.metadata.get('page', '?')
            content = doc.page_content.replace("\n", " ")
            formatted_text += (
                f"\n[SOURCE: {source} | PAGE: {page}]\n"
                f"CONTENT: {content}\n\n"
            )
        return formatted_text
 
    prompt_template =  """You are a Senior Legal Analyst specializing in Indian Labour Reforms, including:
- Code on Wages
- Occupational Safety, Health and Working Conditions Code (OSHWC)
- Code on Social Security
- Industrial Relations Code

Your analysis must be legally precise, citation-driven, and jurisdiction-aware.

------------------------------------------------------------
ANALYTICAL PERSPECTIVE:
------------------------------------------------------------
You must analyze the query strictly from the following legal perspective:
{perspective}

- The perspective defines the PRIMARY labour code or legal lens to apply.
- Do NOT introduce other labour codes unless they are legally necessary for comparison.
- If the query falls outside this perspective, clearly state so.

------------------------------------------------------------
CHAT HISTORY (CONTEXT ONLY – DO NOT CITE):
------------------------------------------------------------
The following is the prior conversation for contextual understanding ONLY.
- Use it to understand intent, continuity, and follow-up nature.
- DO NOT treat chat history as a legal source.
- DO NOT cite chat history.
- ALL legal conclusions must come from CONTEXT or Central Code references.

{chat_history}

------------------------------------------------------------
INSTRUCTIONS (STRICT):
------------------------------------------------------------

1. **Source Material Constraint**
   - You MUST primarily rely on the provided CONTEXT.
   - The CONTEXT is extracted from State Draft Rules and/or Central Labour Codes.
   - The CONTEXT contains explicit [SOURCE] and [PAGE] tags.

2. **Citation Rule (MANDATORY)**
   - EVERY factual or legal statement MUST end with a citation in the format:
     (File Name, Page No)
   - Example:
     "The employer must maintain electronic registers (OSHWC_Rules.pdf, Page 42)."

3. **Use of Internal Knowledge (Controlled)**
   - If the CONTEXT does NOT define a required legal term or background:
     - You MAY use internal knowledge of the relevant Central Labour Code.
     - You MUST explicitly state:
       "As per the Central Code (Internal Legal Reference)..."
     - Mention the exact Section number.
     - Clearly distinguish internal legal reference from contextual facts.

4. **Language**
   - Answer strictly in **ENGLISH**.
   - Do NOT translate statutory text unless necessary for explanation.

5. **No Hallucination Rule**
   - If the answer is NOT found in the CONTEXT and cannot be reasonably supplemented by Central Code knowledge:
     - Clearly state:
       "This information is not found in the provided documents."

------------------------------------------------------------
REQUIRED RESPONSE STRUCTURE:
------------------------------------------------------------

### 1. The Rule (From Context)
- Explain what the provided State Draft Rules or Central Code say about the issue.
- Some content in the CONTEXT may be irrelevant; you must identify and use only what is legally relevant.
- Mention the specific Rule number, Section number, Form number, or procedural reference where available.
- EACH sentence MUST include a citation (File Name, Page No).

### 2. Legal Definition (If Required)
- If the CONTEXT does not define a key legal term:
  - Provide the definition using Central Labour Code knowledge.
  - Explicitly label it as:
    "As per the Central Code (Internal Legal Reference)"
  - Mention the applicable Section number.

### 3. Old vs New Analysis (CRITICAL)
- Compare the provision with corresponding older legislation such as:
  - Factories Act, 1948
  - Contract Labour (Regulation and Abolition) Act, 1970
  - Inter-State Migrant Workmen Act, 1979
- Clearly state:
  - What has changed
  - What is newly introduced
  - What has been removed, merged, or consolidated
- If there is no substantive change, explicitly state:
  "This provision remains largely similar to the previous Act."

------------------------------------------------------------
CONTEXT:
------------------------------------------------------------
{context}

------------------------------------------------------------
QUESTION:
------------------------------------------------------------
{question}

------------------------------------------------------------
DETAILED ANALYST RESPONSE (IN ENGLISH):
------------------------------------------------------------

"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
   
    def expand_query(law_type: str, query: str) -> str:
        prompt = PromptTemplate(
            input_variables=["query", "law_type"],
            template="""
    You are assisting in legal information retrieval.

    Given:
    Query: {query}
    Labour law subsection: {law_type}

    Generate exactly 10 short related legal phrases or keywords relevant to this query.
    Return them as a comma-separated list only.
    """
        )

        chain = prompt | llm

        response = chain.invoke({
            "query": query,
            "law_type": law_type
        })

        expanded_terms = [
            term.strip()
            for term in response.content.split(",")
            if term.strip()
        ]

        expanded_query = f"{query} {law_type} " + " ".join(expanded_terms)
        return expanded_query

    query = expand_query(law_type, query)
    print(query)

    retriever = vector_store.as_retriever(search_kwargs={"k": 12})
    docs = retriever.invoke(query)
    context = format_docs_with_citation(docs)

    print(context)
    
 
    chain = PROMPT | llm
 
    response = chain.invoke({
        "context": context,
        "question": query,
        "perspective" : law_type,
        "chat_history" : chat_history
    })

 
    # print("AI ANSWER:")
 
    # print(response.content)
    return response.content