import sqlite3
c = sqlite3.connect("monitor.db")
c.execute("DELETE FROM samples WHERE session_id IN (SELECT id FROM sessions WHERE kind LIKE 'nthu_img%')")
c.execute("DELETE FROM sessions WHERE kind LIKE 'nthu_img%'")
c.commit()
print("Da xoa du lieu NTHU cu")