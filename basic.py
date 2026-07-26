from nselib import derivatives
df = derivatives.fno_bhav_copy(trade_date='24-07-2026')  # pick a recent trading day
print(df.head())