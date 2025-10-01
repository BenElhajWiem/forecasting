def generate_statistical_summary(df):

  avg_demand = df["TOTALDEMAND"].mean()
  avg_price = df["RRP"].mean()
  min_demand = df["TOTALDEMAND"].min()
  max_demand = df["TOTALDEMAND"].max()
  min_price = df["RRP"].min()
  max_price = df["RRP"].max()
  trend_TOTALDEMAND = "increasing" if df["TOTALDEMAND"].iloc[-1] > df["TOTALDEMAND"].iloc[0] else "decreasing"
  trend_RRP = "increasing" if df["RRP"].iloc[-1] > df["RRP"].iloc[0] else "decreasing"

  statistical_summary=f"""Statistical Summary:
      - Avg Demand: {avg_demand:.2f}
      - Avg Price: {avg_price:.2f}
      - Min Demand: {min_demand:.2f}, Max Demand: {max_demand:.2f}
      - Min Price: {min_price:.2f}, Max Price: {max_price:.2f}
      - General trend for TOTALDEMAND : {trend_TOTALDEMAND}
      - General trend for RRP : {trend_RRP}"""

  return statistical_summary