from django.db import models

# Create your models here.

class uq_event(models.Model):

    # event name
    evt_name = models.CharField(max_length=130)

    # event place
    evt_place = models.CharField(max_length=130, blank=True)

    # event exact time, used for deadlines and activities
    evt_exact_time = models.CharField(max_length=130, blank=True)
    evt_year =  models.CharField(max_length=4, blank=True)
    evt_month = models.CharField(max_length=2, blank=True)
    evt_date = models.CharField(max_length=2, blank=True)

    evt_end_time = models.CharField(max_length=130, blank=True)

    # event duration, used for Events
    evt_duration = models.CharField(max_length=130, blank=True)
    # whether it is done
    evt_done = models.BooleanField(default=False)

    # event type
    EVENT = 'EV'
    DEADLINE = 'DL'
    GOAL = 'GL'
    EVT_COHICES = (
        (EVENT, 'event'),
        (DEADLINE, 'deadline'),
        (GOAL, 'goal'),
    )
    evt_type = models.CharField(max_length=2,
                                choices=EVT_COHICES,
                                default=EVENT)

    uuid = models.CharField(max_length=130, blank=True)

    unix_tmp = models.FloatField(default=1000000000.0)

    urgent = models.BooleanField(default=False)
