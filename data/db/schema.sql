CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    last_seen TIMESTAMP,
    preferences JSON
);

CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    timestamp TIMESTAMP,
    message TEXT,
    response TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE metrics (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP,
    metric_type TEXT,
    value REAL
);
