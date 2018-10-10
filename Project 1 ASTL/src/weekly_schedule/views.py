from django.shortcuts import render
from sd.models import uq_event
from django.http import HttpResponseRedirect

# Create your views here.
import time
import datetime

datetime.datetime(2012,4,1,0,0).timestamp()

def index(request):
    urgent_events = uq_event.objects.filter(urgent=True).order_by("-unix_tmp").all()
    missed_events = uq_event.objects.filter(urgent=False,evt_done=False,unix_tmp__lte=datetime.datetime.now().timestamp()).order_by("-unix_tmp").all()
    upc_events = uq_event.objects.filter(urgent=False, evt_done=False,unix_tmp__gte=datetime.datetime.now().timestamp()).order_by("unix_tmp").all()
    return render(request, 'weeklySchedule.html',{'upc_events':upc_events,'missed_events':missed_events,'urgent_events':urgent_events})

def mark_as_urgent(request, id):
    found_evnt = uq_event.objects.get(uuid=id)
    found_evnt.urgent = True
    found_evnt.save()
    return HttpResponseRedirect("/weekevent/")

def mark_as_dismiss(request, id):
    found_evnt = uq_event.objects.get(uuid=id)
    found_evnt.urgent = False
    found_evnt.save()
    return HttpResponseRedirect("/weekevent/")

def mark_event_done(request, id):

    found_evnt = uq_event.objects.get(uuid=id)
    found_evnt.evt_done = True
    found_evnt.save()
    return HttpResponseRedirect("/weekevent/")
    #return render(request, 'allEvents.html',{'events':events,'done_events':done_events})

def mark_recover(request, id):
    found_evnt = uq_event.objects.get(uuid=id)
    found_evnt.evt_done = False
    found_evnt.save()
    return HttpResponseRedirect("/weekevent/")
