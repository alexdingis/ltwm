# %%
# Load libraries
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Loading the .env file was being a pain
env_path = os.path.join(os.getcwd(), ".env")
result   = load_dotenv(dotenv_path=env_path)
print("load_dotenv(dotenv_path=...) result:", result)

"""
FEMA Large Disasters
https://www.fema.gov/openfema-data-page/individual-assistance-housing-registrants-large-disasters-v1
he Large Disasters dataset contains a curated set of disasters for which individuals applied for Individual Assistance. This dataset only contains some of the largest disasters. No new disasters will be added to this dataset.

"""
# %%
# Load dataframe
fema = os.getenv("fema_large_disasters_path") 
print("Loaded FEMA path:", fema)

# Format dataframe
df            = pd.read_csv(fema)
df.columns    = [str(h).upper() for h in list(df)]
df["CENSUSBLOCKID"] = df["CENSUSBLOCKID"].astype(str).apply(lambda x: (str(x).replace(".0", "")).zfill(15))

# Display dataframe
print(f"\nRecords in dataframe: {len(df):,}\n")
c = 0
m = max([len(x) for x in list(df)])
for item in list(df):
    print("{0} | {1} | {2} | {3}".format(
        str(c).zfill(3),
        str(df[item].dtype).ljust(7),
        str(item).ljust(m),
        str(df[item].iloc[0])
    ))
    c = c + 1
# %%
# Load census block crosswalk
xwalk = os.getenv("blk10_to_tr20_path") 
dfx   = pd.read_csv(xwalk)

# Format xwalk dataframe
dfx.columns      = [str(h).upper() for h in list(dfx)]
dfx["BLK2010GE"] = dfx["BLK2010GE"].astype(str).apply(lambda x: (str(x).replace(".0", "")).zfill(15))
dfx["TR2020GE"]  = dfx["TR2020GE"].astype(str).apply(lambda x: (str(x).replace(".0", "")).zfill(11))

# Display xwalk dataframe
print(f"Records in len(dfx)")
print(dfx.head(5))

# %%
# Create a dataframe that is all unique 2020 census tract GEOIDs
# This will be used later to merge in SERIOUS and SEVERE damage counts
geoids    = list(dfx["TR2020GE"].unique())
df_unique = pd.DataFrame(geoids,columns=["TRACT20"])
df_unique = df_unique.sort_values(by="TRACT20",ascending=True).reset_index(drop=True)
print(f"Records in unique TRACT20 dataframe: {len(df_unique):,}")
print(df_unique.head(10))

# %%
# Load Connecticut census tract GEOIDs which will be used at the very end to correct for the changed GEOIDs
# This wasn't needed

# %% 
# Calculate the flags
"""
From Todd Richardson (these are written into CDBG-DR appendices somewhere):
Serious Damage = for homeowners, FEMA inspected real property damage of $8,000 or more or personal property greater than $3,500; for renters personal property damage of $2,000 or more; or flooding of 1 foot or more on first floor.
Severe Damage  = for homeowners, FEMA inspected real property damage of $28,000 or more or personal property greater than $9,000; for renters personal property damage of $7,500 or more; or flooding of 6 foot or more on first floor; or recorded by FEMA as destroyed.

Severe_Tract   = 1 if the count of severe damaged homes is greater than or equal to 50.
Serious_Tract  = 1 if the count of serous damaged homes is greater than or equal to 100 or it is also a Severe_Tract.

"""
# Create Serious Damage flag
df["SERIOUS_DAMAGE"] = (
    ((df["OWNRENT"] == "Owner") & (
        (df["RPFVL"] >= 8000) |
        (df["PPFVL"] >= 3500) |
        (df["WATERLEVEL"] >= 12)
    )) |
    ((df["OWNRENT"] == "Renter") & (
        (df["PPFVL"] >= 2000) |
        (df["WATERLEVEL"] >= 12)
    ))
)

print("\nTotal records:", len(df))
print("\nSerious damage cases:", df['SERIOUS_DAMAGE'].sum())
print("Percent serious damage:", round(df['SERIOUS_DAMAGE'].mean() * 100, 2), "%")

# Create Severe Damage flag
df["SEVERE_DAMAGE"] = (
    ((df["OWNRENT"] == "Owner") & (
        (df["RPFVL"] >= 28000) |
        (df["PPFVL"] >= 9000) |
        (df["WATERLEVEL"] >= 72) |
        (df["DESTROYED"] == 1)
    )) |
    ((df["OWNRENT"] == "Renter") & (
        (df["PPFVL"] >= 7500) |
        (df["WATERLEVEL"] >= 72) |
        (df["DESTROYED"] == 1)
    ))
)

print("\nSevere damage cases:", df['SEVERE_DAMAGE'].sum())
print("Percent severe damage:", round(df['SEVERE_DAMAGE'].mean() * 100, 2), "%")

# %%
# Ensure 2020 census tracts for all reocrds

# --- Step 1: Clean dfx to keep only the highest-weight row per 2010 block ---
dfx_clean = dfx.sort_values("WEIGHT", ascending=False).drop_duplicates("BLK2010GE")

# --- Step 2: Split FEMA data into 2020 and 2010 groups ---
fema_2020 = df[df["CENSUSYEAR"] == 2020.0].copy()
fema_2010 = df[df["CENSUSYEAR"] == 2010.0].copy()

# --- Step 3: For 2020, extract first 11 digits of CENSUSBLOCKID as tract ---
fema_2020["TRACT20"] = fema_2020["CENSUSBLOCKID"].astype(str).str.slice(0, 11)

# --- Step 4: For 2010, merge with cleaned crosswalk to get TR2020GE ---
fema_2010 = fema_2010.merge(
    dfx_clean[["BLK2010GE", "TR2020GE"]],
    left_on="CENSUSBLOCKID",
    right_on="BLK2010GE",
    how="left"
)
fema_2010["TRACT20"] = fema_2010["TR2020GE"]

# --- Step 5: Combine both subsets back together ---
df_with_tract = pd.concat([fema_2020, fema_2010], ignore_index=True)
# %%
# Inspect result
"""
98.8% of census tracts had a CENSUSYEAR value and could be correctly mapped
This loss varies -1.2% to -2.0% depending on the flag
                 Serious                      Serious
                  TRUE        FALSE            TRUE        FALSE
Pre TRACT20     360,896    6,006,438         42,832    6,324,502
Post TRACT20    355,396    5,935,698         41,995    6,249,099
Missing          -5,500      -70,740           -837      -75,403
Missing Share     -1.5%        -1.2%           -2.0%        -1.2%

"""
print(df["SERIOUS_DAMAGE"].value_counts(dropna=False), df["SEVERE_DAMAGE"].value_counts(dropna=False))
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print(df_with_tract["SERIOUS_DAMAGE"].value_counts(dropna=False), df_with_tract["SEVERE_DAMAGE"].value_counts(dropna=False))
# %%
"""
Perform groupby by to understand the number of SERIOUS and SEVERE damage counts by state declaration
DISASTERNUMBER     DR        SERIOUS_DAMAGE    SEVERE_DAMAGE
4332              TX-4332          152,512           21,545
4337              FL-4337           25,283            1,261
4339              PR-4339           73,507            8,471
4393              NC-4393           17,356            1,531
4399              FL-4399            9,979            1,968
4559              LA-4559           17,818            1,497
4586              TX-4586            5,602               84
4611              LA-4611           53,339            5,638
"""
df_with_tract["DR"] = (df_with_tract["DAMAGEDSTATEABBREVIATION"].astype(str)+ "-"+ df_with_tract["DISASTERNUMBER"].astype(int).astype(str).str.zfill(4))
grpby               = df_with_tract.groupby(["DISASTERNUMBER", "DR"])[["SERIOUS_DAMAGE", "SEVERE_DAMAGE"]].sum().reset_index()
grpby               = grpby[(grpby["SERIOUS_DAMAGE"] > 0) | (grpby["SEVERE_DAMAGE"] > 0)]
print(f"Number of records in state-level DR groupby: {len(grpby):,}")
print(grpby)
# %%
# 
# Step 1: Filter for damage cases
df_filtered = df_with_tract[df_with_tract["SERIOUS_DAMAGE"] | df_with_tract["SEVERE_DAMAGE"]].copy()

# Step 2: Group by tract and DR, count incidents
tract_dr_counts = df_filtered.groupby(["TRACT20", "DR"])[["SERIOUS_DAMAGE", "SEVERE_DAMAGE"]].sum().reset_index()

# Step 3: Create a flag based on thresholds
tract_dr_counts["FLAG"] = (
    (tract_dr_counts["SERIOUS_DAMAGE"] >= 100) |
    (tract_dr_counts["SEVERE_DAMAGE"] >= 50)
).astype(int)

# Step 4: Pivot to wide format → 1 column per DR with _FLAG suffix
tract_flags = tract_dr_counts.pivot(index="TRACT20", columns="DR", values="FLAG").fillna(0).astype(int)
tract_flags.columns = [f"{col}_FLAG" for col in tract_flags.columns]
tract_flags = tract_flags.reset_index()

print(len(tract_flags))
print(tract_flags.head(10))

# %%
# Step 1: Filter only the rows where either flag is True
df_filtered = df_with_tract[df_with_tract["SERIOUS_DAMAGE"] | df_with_tract["SEVERE_DAMAGE"]].copy()
print(f"Records in filtered dataframe: {len(df_filtered):,}\n")

# Step 2: Melt into long format so we can group by flag type
df_long        = df_filtered.melt(
    id_vars    = ["TRACT20", "DR"],
    value_vars = ["SERIOUS_DAMAGE", "SEVERE_DAMAGE"],
    var_name   = "flag",
    value_name = "value"
)

# Step 3: Keep only rows where flag is True
df_long = df_long[df_long["value"] == True]

# Step 4: Create a combined column for pivoting
df_long["colname"] = df_long["DR"] + "_" + df_long["flag"].str.replace("_DAMAGE", "", regex=False)

# Step 5: Group and count
df_counts = df_long.groupby(["TRACT20", "colname"]).size().unstack(fill_value=0)

# Step 6: Reindex to include all TRACT20s from df_unique
counts = df_unique[["TRACT20"]].drop_duplicates().set_index("TRACT20").join(df_counts)

# Optional: Reset index if you want TRACT20 as a column
counts = counts.reset_index()

print(counts.head(10))
out_path = os.path.join(os.getenv("layer_dir"), "FEMA_IA_Large_Disasters_Counts.xlsx")
writer   = pd.ExcelWriter(out_path)
counts.to_excel(writer, index=False)
writer.close()
# %%
counts = df_long[df_long["value"] == True]\
    .groupby(["TRACT20", "DR", "flag"])\
    .size()\
    .unstack(fill_value=0)\
    .reset_index()

# Add up total flags
counts["SEVERE_TOTAL"]  = counts.get("SEVERE_DAMAGE", 0)
counts["SERIOUS_TOTAL"] = counts.get("SERIOUS_DAMAGE", 0)

# Final flag
counts["TRACT_DAMAGE_FLAG"] = 0
counts.loc[counts["SERIOUS_TOTAL"] >= 100, "TRACT_DAMAGE_FLAG"] = 1
counts.loc[counts["SEVERE_TOTAL"] >= 50,  "TRACT_DAMAGE_FLAG"] = 2
print(counts.head(5))
# %%
cols = list(set([str(h).upper().replace("_SERIOUS", "").replace("_SEVERE", "") for h in list(flags) if h != "TRACT20"]))
print(cols)
print(df["DR"])
# %%
# Create flags depending on number of observations in each cell
# 0 = null or 0
# 1 for SEVERE if SEVERE < 50
# 1 for SERIOUS if SERIOUS < 100
# 2 for SEVERE if SEVERE >= 50
# 2 for SERIOUS if SERIOUS >= 100
# Make a copy to preserve original
flags = counts.copy()

# Iterate over each column (excluding TRACT20)
for col in flags.columns:
    if col == "TRACT20":
        continue

    # Set initial flag to 0
    flags[col] = flags[col].fillna(0)

    # Apply thresholds
    flags[col] = np.where(flags[col] >= 100 if "SERIOUS" in col else flags[col] >= 50, 2,
                  np.where(flags[col] > 0, 1, 0))
    
print(flags)
# %%
# Smush the SERIOUS and SEVERE flags into one
cols = [str(h).upper() for h in list(flags) if h != "TRACT20"]
vc = flags[cols].value_counts(dropna=False).reset_index()
print(vc)
# %%
# Output the large disasters dataframe
out_path = os.path.join(os.getenv("layer_dir"), "FEMA_IA_Large_Disasters_Flags.xlsx")
writer   = pd.ExcelWriter(out_path)
flags.to_excel(writer, index=False)
writer.close()