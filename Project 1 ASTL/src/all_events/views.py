from django.shortcuts import render

from django.contrib import messages
import re

from sd.models import uq_event
from django.http import HttpResponseRedirect
# Create your views here.

#@ensure_csrf_cookie

def index(request):
    events = uq_event.objects.filter(evt_done=False).order_by("unix_tmp").all()
    done_events = uq_event.objects.filter(evt_done=True).order_by("-unix_tmp").all()
    return render(request, 'allEvents.html',{'events':events,'done_events':done_events})


def mark_event_done(request, id):

    found_evnt = uq_event.objects.get(uuid=id)
    found_evnt.evt_done = True
    found_evnt.save()
    return HttpResponseRedirect("/allevent/")
    #return render(request, 'allEvents.html',{'events':events,'done_events':done_events})

def mark_recover(request, id):
    found_evnt = uq_event.objects.get(uuid=id)
    found_evnt.evt_done = False
    found_evnt.save()
    return HttpResponseRedirect("/allevent/")

def delete_event(request, id):
    found_evnt = uq_event.objects.get(uuid=id)
    found_evnt.delete()
    return HttpResponseRedirect("/allevent/")
