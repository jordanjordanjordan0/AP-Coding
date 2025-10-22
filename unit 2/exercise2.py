import pandas as pd
import nfl_data_py as nfl

from helperFunctions import get_team_records, get_season_Results_By_team

#schdules = get_team_records(2025)
#print(schdules)

#top6_Teams = ('TB','IND','LA','BUF','SF','SEA','PIT')


#team_1 = get_season_Results_By_team(2025,'TB')
#team_2 = get_season_Results_By_team(2025,'IND')
#team_3 = get_season_Results_By_team(2025,'LA')
#team_4 = get_season_Results_By_team(2025,'BUF')
#team_5 = get_season_Results_By_team(2025,'SF')
#team_6 = get_season_Results_By_team(2025,'SEA')







def pdCheck():
    print("please enter a number")
    number = input()
    values = []

    while number != 'q':
        values.append(int(number))
        print(values)
        print("Please enter a number")
        number = input()
    else:
        print('doing calculation...')
        total = sum(values)
        print(total)


# The best home point difference this season is INDY 64

# the best away point difference this season is 
        

pdCheck()
