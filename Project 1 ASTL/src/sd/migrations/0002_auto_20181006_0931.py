# Generated by Django 2.2.dev20181004154227 on 2018-10-06 09:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sd', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='uq_event',
            name='evt_date',
            field=models.CharField(blank=True, max_length=2),
        ),
        migrations.AddField(
            model_name='uq_event',
            name='evt_end_time',
            field=models.CharField(blank=True, max_length=130),
        ),
        migrations.AddField(
            model_name='uq_event',
            name='evt_month',
            field=models.CharField(blank=True, max_length=2),
        ),
        migrations.AddField(
            model_name='uq_event',
            name='evt_year',
            field=models.CharField(blank=True, max_length=4),
        ),
    ]