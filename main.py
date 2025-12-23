from fastapi import FastAPI, UploadFile, HTTPException
from core.query_pipeline import process_query
from utility.manage_sessions import authenticate_user_service, get_user_sessions, get_chat_session, update_or_create_session_service, star_user_session, delete_user_session, rename_user_session
# from utility.feedback import submit_feedback_logic, remove_feedback_logic
from fastapi import BackgroundTasks, Query
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timezone
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
# from langsmith import traceable
import os
from dotenv import load_dotenv
# from langsmith import Client
# from langsmith.run_helpers import get_current_run_tree
# from core.speech_to_text import convert_speech_to_text
# from core.text_to_speech import generate_audio_response
# from core.text_translate import translate_text


load_dotenv()

# LANGCHAIN_TRACING_V2= os.getenv("LANGCHAIN_TRACING_V2")
# LANGCHAIN_ENDPOINT= os.getenv("LANGCHAIN_ENDPOINT")
# LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
# LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT")

app = FastAPI(
    title='Labour Laws ChatBot API',
    description='APIs for question answering on labour laws',
    version='0.0.1' 
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the specific domain(s) for security, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # You can also specify ['GET', 'POST'] for tighter control
    allow_headers=["*"],  # You can specify specific headers if necessary
)

# client = Client()

# Input format
class AuthInput(BaseModel):
    user_id: str


class QueryRequest(BaseModel):
    user_id: str
    session_id: str
    session_title: str
    is_starred: bool
    message_id: str
    state_id: Optional[str] = None  # Optional if you aren't using it now
    state_name: str
    law_type: str
    query: str
    language_code: str
    timestamp: str  # ISO formatted timestamp from frontend


class StarSessionRequest(BaseModel):
    user_id: str  # can be email
    session_id: str
    starred: bool


class DeleteSessionRequest(BaseModel):
    user_id: str  # can be email
    session_id: str


# Define the feedback model
class FeedbackRequest(BaseModel):
    user_id: str
    session_id: str
    run_id: str
    score: int  # 0 for thumbs down, 1 for thumbs up, and 2 for double thumbs up
    value: str # poor for thumbs down, satisfactory for thumbs up, and good for double thumbs up
    comment: str


class RemoveFeedbackRequest(BaseModel):
    user_id: str
    session_id: str
    run_id: str


class AudioRequest(BaseModel):
    response: str
    language: str

class RenameSessionRequest(BaseModel):
    user_id: str
    session_id: str
    new_title: str

# @app.post("/query", tags=["Query"])
# async def handle_query(request: QueryRequest, background_tasks: BackgroundTasks):
#     try:
#         user_id = request.user_id.split("@")[0]
#         message_id = request.message_id
#         query = request.query

#         response = process_query(
#             query,
#             request.state_name,
#             request.law_type
#         )

#         # Placeholder until LangSmith is enabled
#         # run = get_current_run_tree()
#         # run_id = str(run.id)

#         import uuid
#         run_id = str(uuid.uuid4())

#         bot_timestamp = datetime.now(timezone.utc).isoformat()

#         result = {
#             "response": response,
#             "run_id": run_id,
#             "state_id": request.state_id,
#             "state_name": request.state_name
#         }

#         background_tasks.add_task(
#             update_or_create_session_service,
#             user_id=user_id,
#             session_id=request.session_id,
#             messages=[
#                 {
#                     "role": "user",
#                     "message_id": message_id,
#                     "state_id": request.state_id,
#                     "state_name": request.state_name,
#                     "message": query,
#                     "timestamp": request.timestamp,
#                     "feedback_status": "not given"
#                 },
#                 {
#                     "role": "bot",
#                     "message_id": run_id,
#                     "state_id": request.state_id,
#                     "state_name": request.state_name,
#                     "message": response,
#                     "timestamp": bot_timestamp,
#                     "feedback_status": "not given"
#                 }
#             ],
#             title=request.session_title
#         )

#         return result

#     except Exception as e:
#         return {
#             "response": "Error processing the query",
#             "error": str(e)
#         }

@app.post("/query", tags=["Query"])
async def handle_query(request: QueryRequest, background_tasks: BackgroundTasks):
    try:
        user_id = request.user_id.split("@")[0]
        session_id = request.session_id
        message_id = request.message_id
        query = request.query

        # ✅ 1. Fetch existing chat history
        session_data = get_chat_session(user_id, session_id) or {}
        chat_history = session_data.get("messages", [])

        # ✅ 2. Pass history into LLM pipeline
        response = process_query(
            query=query,
            state=request.state_name,
            law_type=request.law_type,
            chat_history=chat_history,
            session_id=session_id,
        )

        import uuid
        run_id = str(uuid.uuid4())
        bot_timestamp = datetime.now(timezone.utc).isoformat()

        result = {
            "response": response,
            "run_id": run_id,
            "state_id": request.state_id,
            "state_name": request.state_name
        }

        # ✅ 3. Persist updated history
        background_tasks.add_task(
            update_or_create_session_service,
            user_id=user_id,
            session_id=session_id,
            messages=[
                {
                    "role": "user",
                    "message_id": message_id,
                    "state_id": request.state_id,
                    "state_name": request.state_name,
                    "message": query,
                    "timestamp": request.timestamp,
                    "feedback_status": "not given"
                },
                {
                    "role": "bot",
                    "message_id": run_id,
                    "state_id": request.state_id,
                    "state_name": request.state_name,
                    "message": response,
                    "timestamp": bot_timestamp,
                    "feedback_status": "not given"
                }
            ],
            title=request.session_title
        )

        return result

    except Exception as e:
        return {
            "response": "Error processing the query",
            "error": str(e)
        }


@app.get("/sessions", tags=["Session"])
async def fetch_user_sessions(user_id: str = Query(..., description="User email or ID")):
    try:
        core_user_id = user_id.split("@")[0]
        sessions = get_user_sessions(core_user_id)
        return {"status": "success", "sessions": sessions}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/session_history", tags=["Session"])
async def get_session_chat_history(
    user_id: str = Query(..., description="User email or ID"),
    session_id: str = Query(..., description="Session ID to fetch history for")
):
    try:
        core_user_id = user_id.split("@")[0]
        chat_history = get_chat_session(core_user_id, session_id)
        return {"status": "success", "chat_history": chat_history}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/update_star_status", tags=["Session"])   
async def star_session(request: StarSessionRequest):
    try:
        user_id = request.user_id.split("@")[0]
        result = star_user_session(user_id, request.session_id, request.starred)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/delete_session", tags=["Session"])   
async def delete_session(request: DeleteSessionRequest):
    try:
        user_id = request.user_id.split("@")[0]
        result = delete_user_session(user_id, request.session_id)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/rename_session", tags=["Session"])
async def rename_session(request: RenameSessionRequest):
    try:
        user_id = request.user_id.split("@")[0]        
        result = rename_user_session(user_id, request.session_id, request.new_title)
        return {"status": "success", "message": "Session renamed successfully", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# # Endpoint for speech-to-text functionality
# @app.post("/speech_to_text", tags=['Speech'])
# @traceable  # This decorator will enable LangSmith tracing for this function
# async def speech_to_text(file: UploadFile):
#     try:
#         user_text, user_language = convert_speech_to_text(file.file)
#         # user_text = convert_speech_to_text(file.file)
#         run = get_current_run_tree()
#         # Check if the conversion returned valid data
        
#         return {"text": user_text, "language": user_language, "run_id": run.id}
#     except Exception as e:
#         return {"response": "Error processing the audio", "error": str(e)}

    
# @app.post("/translated_response", tags=['Speech'])
# @traceable  # This decorator will enable LangSmith tracing for this function
# async def get_translated_response(query: AudioRequest):
#     translated_response = translate_text(query.response, query.language)
#     return translated_response


# @app.post("/speech_response", tags=['Speech'])
# @traceable  # This decorator will enable LangSmith tracing for this function
# async def get_speech_response(query: AudioRequest):
#     try:

#         # Generate audio response in the regional language (e.g., Marathi 'mr-IN')
#         audio_stream = generate_audio_response(query.response, query.language)
#         # print(audio_file)
#         # If audio_stream is empty, return an error message instead of streaming
#         if audio_stream is None or audio_stream.getbuffer().nbytes == 0:
#             raise HTTPException(status_code=500, detail="Audio generation failed or returned empty content")
#         # run = get_current_run_tree()
#          # Debug final size
#         print(f"Final audio size: {audio_stream.getbuffer().nbytes} bytes")
#         # Create a StreamingResponse to send audio data to the frontend
#         # audio_stream.seek(0)
#         return StreamingResponse(audio_stream, media_type="audio/wav")
        
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e), "message": "Failed to generate speech response"})


# Feedback endpoint
# @app.post("/submit-feedback", tags=["Feedback"])
# async def submit_feedback(feedback: FeedbackRequest):
#     """
#     Submit feedback for a bot message and update LangSmith + chat history.
#     """
#     try:
#         user_id = feedback.user_id.split("@")[0]
#         return submit_feedback_logic(user_id, feedback.session_id, feedback.run_id,
#                                      feedback.score, feedback.value, feedback.comment)  # Feedback stored + LangSmith logging happens here
#     except Exception as e:
#         return {"status": "error", "message": str(e)}


# @app.post("/remove-feedback", tags=["Feedback"])
# async def remove_feedback(request: RemoveFeedbackRequest):
#     """
#     Remove feedback for a bot message (reset status and archive feedback).
#     """
#     try:
#         user_id = request.user_id.split("@")[0]
#         return remove_feedback_logic(
#             user_id=user_id,
#             session_id=request.session_id,
#             run_id=request.run_id
#         )
#     except Exception as e:
#         return {"status": "error", "message": str(e)}


# @app.post("/feedback", tags=['Feedback'])
# # @traceable  # This decorator will enable LangSmith tracing for this function
# async def handle_feedback(feedback: FeedbackRequest):
#     try:
#         # Log the feedback to LangSmith
#         client.create_feedback(
#             run_id=feedback.run_id,
#             key="",
#             score=feedback.score,
#             value=feedback.value,
#             comment=feedback.comment
#         )
#         return {"message": "Feedback logged successfully"}
#     except Exception as e:
#         return {"error": str(e), "message": "Failed to log feedback"}

if __name__ =='__main__':
    uvicorn.run("main:app",host = '0.0.0.0',port=8007,reload=True)
