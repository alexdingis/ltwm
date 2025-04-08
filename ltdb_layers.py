# %%
# Load libraries
from collections import Counter
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sqlalchemy

env = os.path.join(os.getcwd(), ".env")
load_dotenv(env)

# %%
# Load dataframe
ltdb = os.getenv("LTDB_PATH")
print("Loaded LTDB path:", ltdb)

# Format dataframe
df            = pd.read_csv(ltdb)
df.columns    = [str(h).upper() for h in list(df)]
df["TRACT20"] = df["TRACT20"].astype(str).apply(lambda x: (str(x).replace(".0", "")).zfill(11))

# Display dataframe
print(f"\nRecords in dataframe: {len(df):,}\n")
print(df.head(12))

"""
I built the variables one-by-one because I was learning the data as I was going. The code is written to understand as I go rather than optimization, though it still runs quickly.
"""
# %%
# Continuous Shrinking (1970-2000)
"""
0: No data in 1970                | Gray
1: Did not meet definitions       | No fill, border only
2: Continuous shrinking 1970-1990 | #fc8d59
3: Continuous shrinking 1970-2000 | #ef6548
4: Continuous shrinking 1970-2010 | #d7301f
5: Continuous shrinking 1970-2020 | #990000
"""
dfa = df[["TRACT20", "YEAR", "X_POP"]].copy()

# Step 1: Filter for 1970 rows and get TRACT20s with X_POP >= 10
valid_tracts = dfa[dfa["YEAR"] == 1970]
valid_tracts = valid_tracts[valid_tracts["X_POP"] >= 10]["TRACT20"].unique()

# Step 1-B: Non-valid tracts for defining tracts without data for 1970
no_data_tracts = df.loc[(df["YEAR"] == 1970) & (df["X_POP"] < 10), "TRACT20"].unique()

# Step 2: Filter the full dataframe to include only those TRACT20s
dfb = dfa[dfa["TRACT20"].isin(valid_tracts)].copy()

# Step 3: Group by tract and compute whether each year's pop is less than the prior year
dfb["POP_DECLINE"] = dfb.groupby("TRACT20")["X_POP"].transform(lambda x: x.diff().lt(0))
"""
How many times did each census tract lose population? Population loss is not ordered
0: 13,188 (22.4%)
1: 16,305 (27.8%)
2: 12,721 (21.7%)
3:  8,439 (14.4%)
4:  5,405 ( 9.2%)
5:  2,688 ( 4.6%)

Ordered
1970-1990: 12,558 (21.4%) |  DIFF | Shrinking 1970-1990
1970-2000:  7,280 (12.4%) | 5,278 | Shrinking 1970-2000
1970-2010:  4,979 ( 8.5%) | 2,688 | Shrinking 1970-2010
1970-2020:  2,688 ( 4.6%) | 2,291 | Shrinking 1970-2020 

Note: 1970-1990 is just for reference and not intended
"""
# Determine continuous shrinking
# How many declined 1970-1980, 1980-1990, and 1990-2020?
result_1990  = dfb[dfb["YEAR"].isin([1980, 1990])                   & (dfb["POP_DECLINE"] == True)].groupby("TRACT20").filter(lambda g: len(g) == 2)["TRACT20"].unique().tolist()
result_2000  = dfb[dfb["YEAR"].isin([1980, 1990, 2000])             & (dfb["POP_DECLINE"] == True)].groupby("TRACT20").filter(lambda g: len(g) == 3)["TRACT20"].unique().tolist()
result_2010  = dfb[dfb["YEAR"].isin([1980, 1990, 2000, 2010])       & (dfb["POP_DECLINE"] == True)].groupby("TRACT20").filter(lambda g: len(g) == 4)["TRACT20"].unique().tolist()
result_2020  = dfb[dfb["YEAR"].isin([1980, 1990, 2000, 2010, 2020]) & (dfb["POP_DECLINE"] == True)].groupby("TRACT20").filter(lambda g: len(g) == 5)["TRACT20"].unique().tolist()

# Count the number of times each tract appears
combined     = result_1990 + result_2000 + result_2010 + result_2020
counts       = Counter(combined)
dfc          = pd.DataFrame(counts.items(), columns=["TRACT20", "FREQ"])
dfc["FREQ"]  = dfc["FREQ"] + 1 # This is so it matches the above schema

# Merge dataframe back together
geoids       = list(df["TRACT20"].unique())
dfd          = pd.DataFrame(geoids, columns=["TRACT20"])
dfd          = pd.merge(dfd, dfc, on="TRACT20", how="left")

# Calculate no data tracts and tracts that did not meet definition
labels       =  {0: "NO DATA", 1:"NOT MEET DEFINITION", 2:"1970-1990", 3:"1970-2000", 4:"1970-2010", 5:"1970-2020"}

dfd.loc[dfd["TRACT20"].isin(no_data_tracts), "FREQ"] = 0
dfd["FREQ"]  = dfd["FREQ"].fillna(1)
dfd.columns  = ["TRACT20", "SHRINKING"]
dfd["LABEL"] = dfd["SHRINKING"].map(labels)

print(f"\nRecords in dataframe: {len(dfd):,}\n")
# print(dfd.head(12))
vc           = (dfd["SHRINKING"].value_counts().reset_index())
vc.columns   = ["SHRINKING", "COUNT"]
vc           = vc.sort_values(by="SHRINKING", ascending=True).reset_index(drop=True)
vc["SHARE"]  = vc["COUNT"] / vc["COUNT"].sum()
vc["LABEL"]  = vc["SHRINKING"].map(labels)
print(vc)

# %%
grpby       = dfb.groupby(["TRACT20"])[["POP_DECLINE"]].sum().reset_index()
x           = (grpby["POP_DECLINE"].value_counts(dropna=False).reset_index())
x["SHARE"]  = round((x["count"] / x["count"].sum()) * 100, 3)
print(x)

# %%
# Population Change 1970-2020
# Categorize Change
def classify_pct_change(val):
    if pd.isna(val):
        return None
    elif val < -0.5:
        return "1: < -0.5"
    elif -0.5 <= val < -0.1:
        return "2: -0.5 to -0.1"
    elif -0.1 <= val < 0.1:
        return "3: -0.1 to 0.1"
    elif 0.1 <= val < 0.5:
        return "4: 0.1 to 0.5"
    elif 0.5 <= val < 1.0:
        return "5: 0.5 to 1.0"
    elif 1.0 <= val < 2.0:
        return "6: 1.0 to 2.0"
    elif 2.0 <= val < 3.0:
        return "7: 2.0 to 3.0"
    else:
        return ">8: 4.0"


# Calculate Change
dfe           = df.copy()
dfe           = dfe[["TRACT20", "YEAR", "X_POP"]]

dfe70         = dfe[dfe["YEAR"].isin([1970])]
dfe70         = dfe70[["TRACT20", "X_POP"]]
dfe70.columns = ["TRACT20", "X_POP_70"]

dfe20         = dfe[dfe["YEAR"].isin([2020])]
dfe20         = dfe20[["TRACT20", "X_POP"]]
dfe20.columns = ["TRACT20", "X_POP_20"]

dfe           = dfe70.merge(dfe20, how="outer", on="TRACT20")

# Calculate change
dfe["CHANGE"]         = (dfe["X_POP_20"] - dfe["X_POP_70"]).where(dfe["X_POP_70"] >= 10)
dfe["CHANGE_PCT"]     = (dfe["CHANGE"]   / dfe["X_POP_70"]).where(dfe["X_POP_70"] >= 10)
dfe["CHANGE_PCT_BIN"] =  dfe["CHANGE_PCT"].apply(classify_pct_change)

# print(dfe["CHANGE"].describe())
# print(dfe["CHANGE_PCT"].describe())
vc          = (dfe["CHANGE_PCT_BIN"].value_counts(dropna=False).reset_index())
vc.columns  = [str(h).upper() for h in list(vc)]
vc          = vc.sort_values(by="CHANGE_PCT_BIN", ascending=True)
vc["SHARE"] = vc["COUNT"] / vc["COUNT"].sum()
valid_total = vc[vc["CHANGE_PCT_BIN"].notna()]["COUNT"].sum()
vc["SHARE_WITHIN"] = np.where(
    pd.notna(vc["CHANGE_PCT_BIN"]),
    vc["COUNT"] / valid_total,
    np.nan
)   
print(vc)

vc["CHANGE_PCT_BIN"] = vc["CHANGE_PCT_BIN"].fillna("No Data in 1970")
vc.plot(kind="bar", x="CHANGE_PCT_BIN", y="COUNT", legend=False, figsize=(10, 5), title="Tract Counts by Shrinking Category")

# Display dataframe
# print(len(dfe))
# print(dfe.head(12))


# %%
# White Flight

"""
In 1970, what was the global percentage of white population? Mean census tract? Average census tract?
https://www2.census.gov/library/publications/decennial/1970/pc-s1-supplementary-reports/pc-s1-11.pdf
In 1970, 87.5% of the United States population was recorded as white. However, this category includes people who likely would not have recorded themselves as white, particularly Hispanics.
MEAN AND MEDIAN are for census tracts whereas TOTAL is the global percentage.
  YEAR     X_NHWHT       X_POP     WHITE_PCT_MEAN   WHITE_PCT_MEDIAN   WHITE_PCT_TOTAL
  -------------------------------------------------------------------------------------
  1970   128,454,922   147,865,939        89.894%           98.382%            86.873%
  1980   140,835,893   180,434,194        79.838%           91.433%            78.054%
  1990   188,041,869   248,316,677        76.812%           88.534%            75.727%
  2000   194,507,647   280,152,516        70.077%           82.014%            69.429%
  2010   196,817,509   308,745,559        64.101%           74.067%            63.748%
  2020   191,697,647   331,449,281        58.406%           65.924%            57.836%
"""
dff              = df.copy()
# avg              = dff.groupby("YEAR").agg({"X_NHWHT": "sum","X_POP": "sum", "WHITE_PCT": ["mean", "median"]}).reset_index()
# avg.columns      = ["YEAR", "X_NHWHT", "X_POP", "WHITE_PCT_MEAN", "WHITE_PCT_MEDIAN"]
# avg["WHITE_PCT"] = avg["X_NHWHT"] / avg["X_POP"]
# print(avg)

# Analyze 1970 census tracts
dfg              = df[(df["YEAR"] == 1980) & (df["X_POP"] >= 10)].copy()
white            = dfg["X_NHWHT"].sum()
total            = dfg["X_POP"].sum()
white_share      = (float(white) / float(total)) * 100
dfg["WHITE_PCT"] = dfg["X_NHWHT"] / dfg["X_POP"]
# print(white, total, white_share)
# print(dfg["WHITE_PCT"].describe())

# deleted dfh here
dfh = df[["TRACT20","YEAR", "X_POP", "X_NHWHT"]]
dfh = dfh[dfh["YEAR"].isin([1980, 2010])].copy()

# Step 1: Pivot to wide format
df_wide                    = dfh.pivot(index="TRACT20", columns="YEAR", values=["X_POP", "X_NHWHT"]).reset_index()

# Rename columns
df_wide.columns            = ["TRACT20", "POP_1980", "POP_2010", "WHT_1980", "WHT_2010"]

# Step 2: Compute white share
df_wide["SHARE_1980"]      = df_wide["WHT_1980"] / df_wide["POP_1980"]
df_wide["SHARE_2010"]      = df_wide["WHT_2010"] / df_wide["POP_2010"]

# Step 3: Calculate raw losses
df_wide["WHITE_LOSS"]      = df_wide["WHT_1980"] - df_wide["WHT_2010"]
df_wide["SHARE_LOSS"]      = df_wide["SHARE_1980"] - df_wide["SHARE_2010"]
df_wide["MAGN_COUNT_LOSS"] = df_wide["WHITE_LOSS"] / df_wide["WHT_1980"]
df_wide["MAGN_SHARE_LOSS"] = df_wide["SHARE_LOSS"] / df_wide["SHARE_1980"]

# Step 4: Compute thresholds based on tracts that lost white population or share
abs_thresh    = df_wide[df_wide["WHITE_LOSS"] > 0]["WHITE_LOSS"].mean() * 0.5
rel_thresh    = df_wide[df_wide["SHARE_LOSS"] > 0]["SHARE_LOSS"].mean() * 0.5
magn_c_thresh = df_wide[df_wide["WHITE_LOSS"] > 0]["MAGN_COUNT_LOSS"].mean() * 0.5
magn_s_thresh = df_wide[df_wide["SHARE_LOSS"] > 0]["MAGN_SHARE_LOSS"].mean() * 0.5

print(f"Thresholds:\n - Absolute loss: {abs_thresh:.2f}\n - Share point loss: {rel_thresh:.3f}\n - Magnitude count loss: {magn_c_thresh:.3f}\n - Magnitude share loss: {magn_s_thresh:.3f}")

# Step 5: Apply thresholds
df_wide["ABS_THRESHOLD_FLAG"]   = df_wide["WHITE_LOSS"] >= abs_thresh
df_wide["REL_THRESHOLD_FLAG"]   = df_wide["SHARE_LOSS"] >= rel_thresh
df_wide["MAGN_COUNT_FLAG"]      = df_wide["MAGN_COUNT_LOSS"] >= magn_c_thresh
df_wide["MAGN_SHARE_FLAG"]      = df_wide["MAGN_SHARE_LOSS"] >= magn_s_thresh

# Final white flight flag
df_wide["WHITE_FLIGHT"] = (
    df_wide["ABS_THRESHOLD_FLAG"] &
    df_wide["REL_THRESHOLD_FLAG"] &
    df_wide["MAGN_COUNT_FLAG"] &
    df_wide["MAGN_SHARE_FLAG"]
)

# Filter to flagged tracts and display
dfi = df_wide[df_wide["WHITE_FLIGHT"] == True]
print(f"Records where white flight was found: {len(dfi):,}")

# Optional: Maryland tracts only
dfj = dfi[dfi["TRACT20"].str.startswith("24033")]
print(len(dfj))
print(dfj.head(10))

# %%
# Join and output data
dfk = pd.merge(dfi[["TRACT20", "WHITE_FLIGHT"]], dfe[["TRACT20", "CHANGE", "CHANGE_PCT", "CHANGE_PCT_BIN"]], how="outer", on="TRACT20")
dfk = pd.merge(dfk, dfd[["TRACT20", "SHRINKING", "LABEL"]])
dfk = dfk[['TRACT20', 'CHANGE', 'CHANGE_PCT', 'CHANGE_PCT_BIN', 'SHRINKING', 'LABEL','WHITE_FLIGHT']]
# print(len(dfk))
# print(list(dfk))
# print(dfk.head(10))
for col in dfk.columns:
    if col == "TRACT20":
        continue
    missing = dfk[col].isna().sum()
    dtype = dfk[col].dtype
    print(f"\nColumn: {col}")
    print(f"Data type: {dtype}")
    print(f"Missing values: {missing}")
    if pd.api.types.is_object_dtype(dfk[col]):
        print("Value counts:")
        print(dfk[col].value_counts())
    elif pd.api.types.is_numeric_dtype(dfk[col]):
        print("Summary statistics:")
        print(dfk[col].describe())
# Output
out_path = os.path.join(os.getenv("layer_dir"), "LTDB_CHANGE.xlsx")
writer   = pd.ExcelWriter(out_path)
dfk.to_excel(writer, index=False)
writer.close()

# %%
# Write to PostgreSQL

dfl = dfk.copy()
dfl["WHITE_FLIGHT"] = dfl["WHITE_FLIGHT"].astype("boolean")

engine = sqlalchemy.create_engine(os.getenv("DATABASE_URL"))
dfl.to_sql(
    name="LTDB_CHANGE",
    con=engine,
    if_exists="replace", 
    index=False,
    dtype={
        "TRACT20":        sqlalchemy.types.Text(),
        "CHANGE":         sqlalchemy.types.Float(),
        "CHANGE_PCT":     sqlalchemy.types.Float(),
        "CHANGE_PCT_BIN": sqlalchemy.types.Text(),
        "SHRINKING":      sqlalchemy.types.Float(),
        "LABEL":          sqlalchemy.types.Text(),
        "WHITE_FLIGHT":   sqlalchemy.types.Boolean()
    }
)
print("\n~~~~~~~~~ Wrote to PostgreSQL ~~~~~~~~~\n")
# %%
n = 10
df_head = pd.read_sql(f"SELECT * FROM LTDB_CHANGE LIMIT {n}", con=engine)

print(df_head)
