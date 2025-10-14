import pandas as pd
import nfl_data_py as nfl

from helperFunctions import get_team_records

schdules = nfl.import_schedules([2024])

print(schdules.columns.tolist())

records = get_team_records(2024)

print(records[['team','wins','losses']])

print(records[['wins']].mean())

# Find a s