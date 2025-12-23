from datetime import datetime
from langsmith import Client
from .blob_utils import read_json_from_blob, write_json_to_blob

client = Client()

def submit_feedback_logic(user_id: str, session_id: str, run_id: str, score: str, value: str, comment: str):
    # Step 1: Update feedback status in chat history
    history_path = f"chat_history/{user_id}/active/{session_id}.json"
    history = read_json_from_blob(history_path)

    found = False
    for msg in history.get("messages", []):
        if msg.get("message_id") == run_id and msg["role"] == "bot":
            msg["feedback_status"] = value
            found = True
            break

    if not found:
        raise ValueError("Message ID not found in chat history.")


    write_json_to_blob(history_path, history)

    # Step 2: Log feedback locally in blob
    feedback_path = f"feedback/{user_id}/active/feedback.json"
    feedback_entry = {
        "message_id": run_id,
        "feedback": value,
        "comment": comment,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat()
    }
    existing = read_json_from_blob(feedback_path) or []
    existing.append(feedback_entry)
    write_json_to_blob(feedback_path, existing)

    score_map = {
    "bad": 0,
    "good": 1,
    "awesome": 2
    }
    score = score_map.get(value.lower())

    # Step 3: Log feedback to LangSmith
    try:
        client.create_feedback(
            run_id=run_id,
            key="" #langsmith key here for Labour Law,
            score=score,  # Optional: can map feedback to score if you want
            value=value,
            comment=comment
        )
    except Exception as e:
        return {"message": "Feedback saved but failed to log to LangSmith", "error": str(e)}

    return {"status": "success", "message": "Feedback saved and logged to LangSmith"}


def remove_feedback_logic(user_id: str, session_id: str, run_id: str):
    # Step 1: Update feedback_status in chat history
    history_path = f"chat_history/{user_id}/active/{session_id}.json"
    history = read_json_from_blob(history_path)

    found = False
    for msg in history.get("messages", []):
        if msg.get("message_id") == run_id:
            msg["feedback_status"] = "not given"
            found = True
            break

    if not found:
        raise ValueError("Message ID not found in chat history")

    write_json_to_blob(history_path, history)

    # Step 2: Move feedback to removed folder
    original_feedback_path = f"feedback/{user_id}/active/feedback.json"
    removed_feedback_path = f"feedback/{user_id}/removed/feedback.json"

    feedback_data = read_json_from_blob(original_feedback_path) or []
    removed_data = read_json_from_blob(removed_feedback_path) or []

    # Find and remove feedback from original list
    updated_feedback = []
    removed_entry = None
    for fb in feedback_data:
        if fb.get("message_id") == run_id:
            removed_entry = fb
        else:
            updated_feedback.append(fb)

    if removed_entry:
        removed_entry["removed_on"] = datetime.now().isoformat()
        removed_data.append(removed_entry)
        write_json_to_blob(removed_feedback_path, removed_data)
        write_json_to_blob(original_feedback_path, updated_feedback)
    else:
        raise ValueError("Feedback entry not found for removal.")

    return {"status": "removed", "message_id": run_id}
