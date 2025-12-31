import sqlite3
from datetime import datetime
from pathlib import Path

# ================================
# CONFIG
# ================================

DB_PATH = Path("ai_detector.db")


# ================================
# DATABASE CONNECTION
# ================================

def get_connection():
    """Create a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)


# ================================
# DATABASE INITIALIZATION
# ================================

def init_db():
    """
    Initialize database and create table only if it does not exist.
    Prints a message indicating the result.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        # Check if table already exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='predictions'
        """)
        table_exists = cursor.fetchone()

        if table_exists:
            print("✔️ Tabella 'predictions' già esistente.")
        else:
            cursor.execute("""
                CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    lexical_diversity REAL,
                    burstiness REAL,
                    avg_sentence_length REAL,
                    model_version TEXT
                )
            """)
            conn.commit()
            print("✅ Tabella 'predictions' creata correttamente.")


# ================================
# INSERT OPERATIONS
# ================================

def save_prediction(
    prediction: str,
    confidence: float,
    lexical_diversity: float = None,
    burstiness: float = None,
    avg_sentence_length: float = None,
    model_version: str = "v1.0"
):
    """Save a model prediction into the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                timestamp,
                prediction,
                confidence,
                lexical_diversity,
                burstiness,
                avg_sentence_length,
                model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            prediction,
            confidence,
            lexical_diversity,
            burstiness,
            avg_sentence_length,
            model_version
        ))
        conn.commit()


# ================================
# READ OPERATIONS
# ================================

def get_all_predictions():
    """
    Retrieve all records from the predictions table.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, prediction, confidence, model_version
            FROM predictions
            ORDER BY id DESC
        """)
        return cursor.fetchall()


def get_last_predictions(limit: int = 10):
    """
    Retrieve the most recent predictions.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, prediction, confidence, model_version
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        return cursor.fetchall()
