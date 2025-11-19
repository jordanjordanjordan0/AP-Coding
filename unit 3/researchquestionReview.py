# 2: This type of question is relational. I believe it to be relational due to how defensive turnovers can a lead to a team getting the ball more and winning. So since these correlate and it has something to do with one another it can be relational.

# 2 - The stats have shown that defensive turnovers led to more wins than the other teams statistically. 


# 3: I think this type of question is Comparitive. This is because you would have to compare player stats to each other to get your answer. While Tom Brady has the most passing yards of all time.








from helperLogic import get_player_stats_by_name, plot_weekly_player_stats,plot_player_stat, get_team_records, get_advanced_team_records, get_position_columns, get_season_Results_By_team , weeklyPlayerStats
import matplotlib.pyplot as plt

# Columns available to research based on year and position
columnData = get_position_columns(2024, "QB")
# print(columnData)

'1. How much does QB pass accuracy influence team wins ? '
teamRecord = get_team_records(2024)
print(teamRecord)

qbData = weeklyPlayerStats(2024, 'QB')
print(qbData)

'J.Allen'
'J.Hurts'

playerStat= get_player_stats_by_name(2024,'J.Hurts','QB')
print(playerStat)

