from datetime import datetime, date

now = datetime.now()
now = now.strftime("%d-%m-%Y %H:%M:%S")
print("Execution: ", now)

today = date.today()
today = today.strftime("%d/%m/%Y")
print("Execution date: ", today)

current_time = datetime.now().time()
current_time = current_time.strftime("%H:%M:%S")
print("Execution Time: ", current_time)