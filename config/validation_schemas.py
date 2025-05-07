"""Configuration validation schemas"""

NLP_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string", "enum": ["en", "nl"]},
        "model": {"type": "string"},
        "max_length": {"type": "integer"},
        "models": {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string"},
                "intent": {"type": "string"},
                "embedding": {"type": "string"}
            }
        }
    },
    "required": ["language"]
}

LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "oneOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "temperature": {"type": "number"},
                                "top_p": {"type": "number"}
                            }
                        }
                    },
                    "required": ["name"]
                }
            ]
        }
    }
}

UI_SCHEMA = {
    "type": "object",
    "properties": {
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "title": {"type": "string"},
        "theme": {"type": "string", "enum": ["light", "dark"]}
    }
}

DATABASE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "type": {"type": "string", "enum": ["sqlite", "mongodb"]},
        "connection": {"type": "string"}
    }
}
