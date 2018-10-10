"""toodoo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/dev/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from sd import views as sd_views
from all_events import views as all_events_views
from weekly_schedule import views as weekly_schedule_views

from django.conf.urls import url

urlpatterns = [
    url(r'^admin/', admin.site.urls, name="amdin"),
    url(r'^$', sd_views.index, name="sdviews"),
    url(r'^allevent/$',all_events_views.index, name="allevt"),
    url(r'^weekevent/$',weekly_schedule_views.index, name="wkevt"),

    # submit event
    url(r'^submit_event/', sd_views.add_2_db),
    # mark as done
    url(r'^allevent/mrkDone/(.*)', all_events_views.mark_event_done),
    # recover
    url(r'^allevent/rcvr/(.*)', all_events_views.mark_recover),
    # Delete
    url(r'^allevent/delete/(.*)', all_events_views.delete_event),

    # mark as urgent
    url(r'^weekevent/urgent/(.*)', weekly_schedule_views.mark_as_urgent),
    # dismiss
    url(r'^weekevent/urgentDismiss/(.*)', weekly_schedule_views.mark_as_dismiss),
    # mark as done
    url(r'^weekevent/mrkDone/(.*)', weekly_schedule_views.mark_event_done),
    # mark as done
    url(r'^weekevent/rcvr/(.*)', weekly_schedule_views.mark_recover),
]
