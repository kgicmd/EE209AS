from django.shortcuts import render
from sd.models import uq_event
from django.shortcuts import render_to_response
from django.shortcuts import render
from django.contrib import messages
import re
import uuid

from django.http import HttpResponse
import json

import pytz
import time
import datetime
# Create your views here.

def index(request):
    #context = {}
    tz = pytz.timezone('America/Los_Angeles')
    today_year = datetime.datetime.now(tz).year
    today_month = datetime.datetime.now(tz).month
    today_date  = datetime.datetime.now(tz).day

    dic = {'evt_done':False, \
           'evt_year':today_year,\
           'evt_month':today_month,\
           'evt_date':today_date,\
           'urgent':False
           }
    events = uq_event.objects.filter(**dic).all()
    urgent_events = uq_event.objects.filter(urgent=True).all()
    return render(request, 'index.html',{'events':events, 'urgent_events':urgent_events})
    #return render(request, 'index.html', context)

def add_2_db(request):

    tz = pytz.timezone('America/Los_Angeles')
    today_year = datetime.datetime.now(tz).year
    today_month = datetime.datetime.now(tz).month
    today_date  = datetime.datetime.now(tz).day

    dic = {'evt_done':False, \
           'evt_year':today_year,\
           'evt_month':today_month,\
           'evt_date':today_date,\
           'urgent':False
           }

    # get from submitted contents
    whole_info = request.POST["new_event"]
    is_event_type = request.POST["is_event_type"]

    is_dea = False
    is_act = False
    is_rem = False

    if is_event_type == '1':
        is_rem = True
    elif is_event_type == '2':
        is_dea = True
    elif is_event_type == '3':
        is_act = True
    else:
        messages.error(request,'choose one')

    # indication of data integrity
    raise_flag_time = False # means no time problem
    raise_flag_place = False # means no place problem
    raise_flag_name = False # means no name problem

    # extract time
    # extract deadline:
    if is_dea:
        pattern = r'(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2})'
        exact_time, place, event_name, raise_flag_time, raise_flag_place, raise_flag_name = extract_info(whole_info, pattern)

        event = uq_event(evt_type='deadline')

        # save time
        proceeddd = 0 # do not move down
        if not raise_flag_time:
            event.evt_exact_time = exact_time
            yr_pattern = r'(\d{4}-\d{1,2}-\d{1,2})'
            mat = re.search(yr_pattern, exact_time)
            exact_time_raw = mat.groups()[0]
            exact_time_raw_list = exact_time_raw.split('-')
            event.evt_year , event.evt_month, event.evt_date = exact_time_raw_list[0], exact_time_raw_list[1], exact_time_raw_list[2]
            proceeddd = 1
        else:
            proceeddd = 0
            messages.error(request,'time wrong')
            events = uq_event.objects.filter(**dic).all()
            urgent_events = uq_event.objects.filter(urgent=True).all()
            return render(request, 'index.html',{'events':events, 'urgent_events':urgent_events})
        # save place
        if (not raise_flag_place) & proceeddd == 1:
            event.evt_place = place
        else:
            proceeddd = 0
            messages.error(request,'place wrong')
            events = uq_event.objects.filter(**dic).all()
            urgent_events = uq_event.objects.filter(urgent=True).all()
            return render(request, 'index.html',{'events':events, 'urgent_events':urgent_events})
        # save name
        if (not raise_flag_name) & proceeddd == 1:
            event.evt_name = event_name
        else:
            proceeddd = 0
            messages.error(request,'name wrong')
            events = uq_event.objects.filter(**dic).all()
            urgent_events = uq_event.objects.filter(urgent=True).all()
            return render(request, 'index.html',{'events':events, 'urgent_events':urgent_events})
        if proceeddd == 1:
            event.uuid = str(uuid.uuid1())
            event.unix_tmp = conv_unix(exact_time)
            event.save()
            messages.success(request,'successfully saved!')
            #return render(request, 'index.html')
            events = uq_event.objects.filter(**dic).all()
            urgent_events = uq_event.objects.filter(urgent=True).all()
            return render(request, 'index.html',{'events':events, 'urgent_events':urgent_events})
    elif is_act:
        pattern = r'(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2})'
        raise_flag_time = False # means no time problem
        #raise_flag_place = False # means no place problem
        raise_flag_name = False # means no name problem
        exact_time = event_name = ''
        mat = re.search(pattern,whole_info)
        # retrive exact time
        try:
            exact_time = mat.groups()[0]
        except:
            raise_flag_time = True

        if not raise_flag_time:
            event_name = whole_info.split('@')[0]
            event_name = event_name.replace(exact_time,'')
            event_name = event_name.rstrip()
            event_name = event_name.lstrip()

            if event_name:
                event_name = ' '.join(event_name.split())
            else:
                raise_flag_name = True


        event = uq_event(evt_type='goal')

        # save time
        proceeddd = 0 # do not move down
        if not raise_flag_time:
            event.evt_exact_time = exact_time
            yr_pattern = r'(\d{4}-\d{1,2}-\d{1,2})'
            mat = re.search(yr_pattern, exact_time)
            exact_time_raw = mat.groups()[0]
            exact_time_raw_list = exact_time_raw.split('-')
            event.evt_year , event.evt_month, event.evt_date = exact_time_raw_list[0], exact_time_raw_list[1], exact_time_raw_list[2]
            proceeddd = 1
        else:
            proceeddd = 0
            messages.error(request,'time wrong')
            events = uq_event.objects.filter(**dic).all()
            urgent_events = uq_event.objects.filter(urgent=True).all()
            return render(request, 'index.html',{'events':events, 'urgent_events':urgent_events})

        # save name
        if (not raise_flag_name) & proceeddd == 1:
            event.evt_name = event_name
        else:
            proceeddd = 0
            messages.error(request,'name wrong')
            events = uq_event.objects.filter(**dic).all()
            urgent_events = uq_event.objects.filter(urgent=True).all()
            return render(request, 'index.html',{'events':events, 'urgent_events':urgent_events})
        if proceeddd == 1:
            event.uuid = str(uuid.uuid1())
            event.unix_tmp = conv_unix(exact_time)
            event.save()
            messages.success(request,'successfully saved!')
            #return render(request, 'index.html')
            events = uq_event.objects.filter(**dic).all()
            urgent_events = uq_event.objects.filter(urgent=True).all()
            return render(request, 'index.html',{'events':events, 'urgent_events':urgent_events})
    else:
        pattern = r"(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}-\d{1,2}:\d{1,2})"
        exact_time, place, event_name, raise_flag_time, raise_flag_place, raise_flag_name = extract_info(whole_info, pattern)

        event = uq_event(evt_type="event")
        # save time
        proceeddd = 0 # do not move down
        if not raise_flag_time:
            #event.evt_exact_time = exact_time
            yr_pattern = r'(\d{4}-\d{1,2}-\d{1,2})'
            mat = re.search(yr_pattern, exact_time)
            exact_time_raw = mat.groups()[0]
            exact_time_raw_list = exact_time_raw.split('-')
            event.evt_year , event.evt_month, event.evt_date = exact_time_raw_list[0], exact_time_raw_list[1], exact_time_raw_list[2]

            time_pattern = re.compile(r'(\d{1,2}:\d{1,2})')
            ss = time_pattern.findall(exact_time)
            exact_start_time_raw = exact_time_raw + ' ' + ss[0]
            event.evt_exact_time = exact_start_time_raw
            exact_end_time_raw = ss[1]
            exact_end_time = exact_time_raw + ' ' + exact_end_time_raw
            event.evt_end_time = exact_end_time
            proceeddd = 1
        else:
            proceeddd = 0
            messages.error(request,'time wrong')
            events = uq_event.objects.filter(**dic).all()
            return render(request, 'index.html',{'events':events})
        # save place
        if (not raise_flag_place) & proceeddd == 1:
            event.evt_place = place
        else:
            proceeddd = 0
            messages.error(request,'place wrong')
            events = uq_event.objects.filter(**dic).all()
            return render(request, 'index.html',{'events':events})
        # save name
        if (not raise_flag_name) & proceeddd == 1:
            event.evt_name = event_name
        else:
            proceeddd = 0
            messages.error(request,'name wrong')
            events = uq_event.objects.filter(**dic).all()
            return render(request, 'index.html',{'events':events})
        if proceeddd == 1:
            event.uuid = str(uuid.uuid1())
            event.unix_tmp = conv_unix(exact_start_time_raw)
            event.save()
            messages.success(request,'successfully saved!')
            #return render(request, 'index.html')
            events = uq_event.objects.filter(**dic).all()
            urgent_events = uq_event.objects.filter(urgent=True).all()
            return render(request, 'index.html',{'events':events, 'urgent_events':urgent_events})

def extract_info(info, pattern):


    raise_flag_time = False # means no time problem
    raise_flag_place = False # means no place problem
    raise_flag_name = False # means no name problem

    exact_time = place = event_name = ''
    mat = re.search(pattern,info)
    # retrive exact time
    try:
        exact_time = mat.groups()[0]
    except:
        raise_flag_time = True

    # retrive place
    try:
        place = info.split('@')[1]
        place = place.lstrip()
        place = place.rstrip()
        if not place:
            raise_flag_place = True
    except:
        raise_flag_place = True

    # retrive event_name
    if not raise_flag_place:
        if not raise_flag_time:
            event_name = info.split('@')[0]
            event_name = event_name.replace(exact_time,'')
            event_name = event_name.rstrip()
            event_name = event_name.lstrip()

            if event_name:
                event_name = ' '.join(event_name.split())
            else:
                raise_flag_name = True
    return exact_time, place, event_name, raise_flag_time, raise_flag_place, raise_flag_name

def conv_unix(evt_time):
    """
    convert to unix timestamp
    """
    st = time.strptime(evt_time, '%Y-%m-%d %H:%M')
    return time.mktime(st)
