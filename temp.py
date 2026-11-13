from database import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:

        cur.execute("""
            SELECT
                current_database(),
                current_user,
                inet_server_addr(),
                inet_server_port(),
                COUNT(*)
            FROM articles;
        """)

        result = cur.fetchone()

        print("Database:", result[0])
        print("User:", result[1])
        print("Server:", result[2])
        print("Port:", result[3])
        print("Article count:", result[4])