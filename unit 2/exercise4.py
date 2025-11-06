from helperFunctions import weeklyPlayerStats, plot_weekly_player_stats, plot_player_stat
import matplotlib.pyplot as plt

stats = weeklyPlayerStats(2024, "Qb")
print(stats)

plot_player_stat(stats, stat="rushing_tds", top_n=10, title="WR rushing TDs (2024)", save_path="WR_rushing_tds_2024.png"  )

plot_weekly_player_stats(2024, "WR", stat="receiving_yards", top_n=15, week=[1,2,3], save_path="wr_rec_yards_wk1-3.png")


# 1 - 


# 2 - Derrick Henry had the most touchdowns with 21 which was most in the league


# 3 - In 2024 Jared Goff had the most passing yard with 4942 yards


# 4 - 