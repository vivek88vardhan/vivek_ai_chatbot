class SessionManager:
    def __init__(self):
        self.sessions = {}

    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {"history": []}
        return self.sessions[session_id]

    def update_session(self, session_id, user_input, bot_response):
        session = self.get_session(session_id)
        session["history"].append({"user": user_input, "bot": bot_response})
