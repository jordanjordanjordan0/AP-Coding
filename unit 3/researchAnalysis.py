from helperLogic import get_player_stats_by_name, plot_weekly_player_stats,plot_player_stat, get_team_records, get_advanced_team_records, get_position_columns, get_season_Results_By_team , weeklyPlayerStats
import matplotlib.pyplot as plt

teamData = get_advanced_team_records(2024, 'BUF')
print(teamData)

teamRes = get_season_Results_By_team(2024, 'BUF')

# 1 - The Division with the strongest defense based on yards allowed per game is the AFC West, This type of question is comparative. You need to comepare Teams stats and add it up in the divisions

# 2 - The WR with the most Targets to Reception ratio is Amon Ra St Brown. To find this you need to once again compare player stats once again but this time for the WR catagory 

# 3 - 