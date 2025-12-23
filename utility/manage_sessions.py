from .blob_utils import read_json_from_blob, write_json_to_blob, copy_blob, delete_blob
from datetime import datetime, timezone
from pprint import pprint
import os
from dotenv import load_dotenv

load_dotenv()  # Load env variables from .env

# ---- Authentication Service ----
def authenticate_user_service(user_id: str):
    """Authenticates user and checks first-time login."""
    auth_blob_path = "auth/users.json"
    users = read_json_from_blob(auth_blob_path)

    if user_id in users:
        is_first_time = users[user_id].get("first_time", True)
        if is_first_time:
            users[user_id]["first_time"] = False
            write_json_to_blob(auth_blob_path, users)
        return {"authenticated": True, "first_time": is_first_time}
    else:
        return {"authenticated": False, "first_time": False}


# ---- Session Services ----
def get_user_sessions(user_id: str):
    """Fetches all session metadata for a user."""
    blob_path = f"sessions/{user_id}/active/sessions.json"
    return read_json_from_blob(blob_path) or []


def get_chat_session(user_id: str, session_id: str):
    """Fetches chat history for a specific session."""
    blob_path = f"chat_history/{user_id}/active/{session_id}.json"
    return read_json_from_blob(blob_path)


def update_or_create_session_service(
    user_id: str,
    session_id: str,
    messages: list,
    title: str = "New Session"
):
    """
    Creates or updates a chat session and appends multiple messages atomically.
    """

    sessions_blob_path = f"sessions/{user_id}/active/sessions.json"
    chat_history_blob_path = f"chat_history/{user_id}/active/{session_id}.json"

    # 1️⃣ Fetch session metadata
    sessions_data = read_json_from_blob(sessions_blob_path) or []
    session_found = False

    # Use last message timestamp for updates
    last_timestamp = messages[-1]["timestamp"]

    for session in sessions_data:
        if session.get("session_id") == session_id:
            session_found = True
            session["last_updated"] = last_timestamp
            break

    if not session_found:
        new_session = {
            "session_id": session_id,
            "title": title,
            "starred": False,
            "created_on": last_timestamp,
            "last_updated": last_timestamp
        }
        sessions_data.append(new_session)

    write_json_to_blob(sessions_blob_path, sessions_data)

    # 2️⃣ Fetch chat history
    chat_history = read_json_from_blob(chat_history_blob_path) or {
        "session_id": session_id,
        "messages": []
    }

    # Append all messages
    chat_history["messages"].extend(messages)

    write_json_to_blob(chat_history_blob_path, chat_history)

    return {
        "session_metadata": sessions_data,
        "chat_history": chat_history
    }


def star_user_session(user_id: str, session_id: str, is_starred: bool):
    """
    Update the 'starred' field for a session in sessions metadata.
    """
    sessions_blob_path = f"sessions/{user_id}/active/sessions.json"
    sessions_data = read_json_from_blob(sessions_blob_path) or []

    for session in sessions_data:
        if session.get("session_id") == session_id:
            session["starred"] = is_starred
            break
    else:
        raise ValueError("Session not found")

    write_json_to_blob(sessions_blob_path, sessions_data)
    return {"status": "success", "starred": is_starred}


def delete_user_session(user_id: str, session_id: str):
    """
    Move the session's chat history to the 'deleted' folder and update sessions metadata.
    """
    # Paths
    original_path = f"chat_history/{user_id}/active/{session_id}.json"
    deleted_path = f"chat_history/{user_id}/deleted/{session_id}.json"
    sessions_blob_path = f"sessions/{user_id}/active/sessions.json"
    sessions_deleted_blob_path = f"sessions/{user_id}/deleted/sessions.json"

    # Step 1: Move chat history file
    copy_blob(original_path, deleted_path)
    delete_blob(original_path)

    # Step 2: Move session metadata from active to deleted
    sessions_data = read_json_from_blob(sessions_blob_path) or []
    deleted_sessions_data = read_json_from_blob(sessions_deleted_blob_path) or []

    session_to_delete = None
    updated_sessions = []
    
    for session in sessions_data:
        if session.get("session_id") == session_id:
            session_to_delete = session
        else:
            updated_sessions.append(session)

    # Write updated active session list
    write_json_to_blob(sessions_blob_path, updated_sessions)

    # Add to deleted sessions
    if session_to_delete:
        deleted_sessions_data.append(session_to_delete)
        write_json_to_blob(sessions_deleted_blob_path, deleted_sessions_data)
    else:
        raise ValueError("Session metadata not found")

    return {"status": "deleted", "session_id": session_id}

def rename_user_session(user_id: str, session_id: str, new_title: str):
    """
    Rename a chat session title in sessions.json stored as a list in Azure Blob.
    """
    try:
        blob_path = f"sessions/{user_id}/active/sessions.json"

        # Read the existing session metadata list
        sessions_data = read_json_from_blob(blob_path) or []

        session_found = False

        for session in sessions_data:
            if session.get("session_id") == session_id:
                session["title"] = new_title
                session["last_updated"] = datetime.now(timezone.utc).isoformat()
                session_found = True
                break

        if not session_found:
            raise ValueError(f"Session ID '{session_id}' not found for user '{user_id}'")

        # Save the updated session list
        write_json_to_blob(blob_path, sessions_data)

        return {"session_id": session_id, "new_title": new_title}

    except Exception as e:
        raise e


if __name__ == "__main__":
    
    # Sample test data
    user_id = "MBS527180@tatamotors.com"
    session_id = "sess001"
    session_title = "Test Chat Session"
    is_starred = False
    state_name = "TestState"
    user_query = "Hello, what can you do?"
    language_code = "en"
    timestamp = datetime.now(timezone.utc).isoformat()

    print("\n--- Test: Authenticate User ---")
    result = authenticate_user_service(user_id)
    pprint(result)

    print("\n--- Test: Get User Sessions ---")
    sessions = get_user_sessions(user_id)
    pprint(sessions)

    print("\n--- Test: Get Specific Session ---")
    session_data = get_chat_session(user_id, session_id)
    pprint(session_data)