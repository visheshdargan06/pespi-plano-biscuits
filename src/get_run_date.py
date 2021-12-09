import sys
import datetime


def get_run_date_from_today():
    today = datetime.date.today()
    if(today.weekday()<6):
        return today - datetime.timedelta(today.weekday()+2)
    else:
        return today - datetime.timedelta(1)

user_input = True
run_date = datetime.date.today()
if(len(sys.argv)==1):
    run_date = get_run_date_from_today()
    user_input = False
else:
    try:
        run_date = datetime.datetime.strptime(sys.argv[1], '%d-%b-%Y')
    except ValueError:
        run_date = get_run_date_from_today()
        user_input = False

print(run_date.strftime('%d-%b-%Y'))
if user_input:
    sys.exit(0)
else:
    sys.exit(1)
