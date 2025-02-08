import duckdb

# Connect to ChromaDB's DuckDB file
conn = duckdb.connect("bavour_db/chroma.sqlite3")

# List tables
print(conn.execute("SHOW TABLES").fetchall())

# View first 10 rows from embeddings
print(conn.execute("SELECT * FROM embeddings LIMIT 100").fetchall())

# Close connection
conn.close()
