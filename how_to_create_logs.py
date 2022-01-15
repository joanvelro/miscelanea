import sys  # Para hacer exit()
import traceback
import pytz as pytz
import datetime as datetime

resultspath = "C:\\Users\\javelascor\\INDRA\\01_ATM"
utczone = pytz.timezone("utc")
sdate = datetime.datetime.now()
date_ = datetime.datetime(sdate.date().year, sdate.date().month, sdate.date().day, sdate.hour, sdate.minute,sdate.second)
startpoint = utczone.localize(date_)

inipred = startpoint.replace(microsecond=0, second=0, minute=0) + datetime.timedelta(hours=1)
mysep = "\n" + "-" * 64

mylog = "\n\n\n == === === === === === === == \n == EXECUTION DETAILS == \n == === === === === === === ==\n"
mylog += mysep + "\n 001. Loading parameters" + mysep
mylog += "\n - Loading data" + mysep
mylog += " Error Loading Model [<!>]\n"
logp = open(resultspath + "/" + "execution_log_" + str(inipred)[0:13].replace(" ", "_").replace(":", "_") + "00.txt", "w")
logp.write(mylog)
logp.close()


