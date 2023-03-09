import pandas as pd
filename = "data-1675829728-0.04"
pd.DataFrame(pd.read_pickle("{}.pkl".format(filename))).to_csv("{}.csv".format(filename), index=False)