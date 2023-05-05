import sqlite3 
conn = sqlite3.connect('chinook.db')
cur = conn.cursor() 

cur.execute("SELECT * FROM invoices WHERE BILLINGCOUNTRY = 'Germany'")

for row in cur:
    print(row)

conn.commit()
conn.close()