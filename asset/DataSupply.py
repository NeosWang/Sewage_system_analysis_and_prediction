import pandas as pd
import numpy as np
import os
import zipfile
from datetime import datetime, timedelta
from tqdm import tqdm

pumpStations = [
    "bokhoven",
    "haarsteeg",
    "helftheuvelweg",
    "engelerschans",
    "oude_engelenseweg",
    "maaspoort",
    "de_rompert",
]

data = {
    (
        "bokhoven",
        True,
        True,
    ): "../../data/sewer_data/data_pump/RG8180_L0_09-2019_07-2020.zip",  # level new
    (
        "bokhoven",
        True,
        False,
    ): "../../data/sewer_data/data_pump/RG8180_L0.zip",  # level old
    (
        "bokhoven",
        False,
        True,
    ): "../../data/sewer_data/data_pump/RG8180_Q0_09-2019_07-2020.zip",  # flow new
    (
        "bokhoven",
        False,
        False,
    ): "../../data/sewer_data/data_pump/RG8180_Q0.zip",  # flow old
    (
        "haarsteeg",
        True,
        True,
    ): "../../data/sewer_data/data_pump/rg8170_N99_09-2019_07-2020.zip",  # level new
    (
        "haarsteeg",
        True,
        False,
    ): "../../data/sewer_data/data_pump/rg8170_N99.zip",  # level old
    (
        "haarsteeg",
        False,
        True,
    ): "../../data/sewer_data/data_pump/rg8170_99_09-2019_07-2020.zip",  # flow new
    (
        "haarsteeg",
        False,
        False,
    ): "../../data/sewer_data/data_pump/rg8170_99.zip",  # flow old
    (
        "helftheuvelweg",
        False,
        True,
    ): "../../data/sewer_data_db/data_pump_flow/1210FIT301_99_09-2019_07-2020.zip",
    (
        "helftheuvelweg",
        False,
        False,
    ): "../../data/sewer_data_db/data_pump_flow/1210FIT301_99.zip",
    (
        "engelerschans",
        False,
        True,
    ): "../../data/sewer_data_db/data_pump_flow/1210FIT201_99_09-2019_07-2020.zip",
    (
        "engelerschans",
        False,
        False,
    ): "../../data/sewer_data_db/data_pump_flow/1210FIT201_99.zip",
    (
        "oude_engelenseweg",
        False,
        True,
    ): "../../data/sewer_data_db/data_pump_flow/1210FIT401_94_09-2019_07-2020.zip",
    (
        "oude_engelenseweg",
        False,
        False,
    ): "../../data/sewer_data_db/data_pump_flow/1210FIT401_94.zip",
    (
        "maaspoort",
        False,
        True,
    ): "../../data/sewer_data_db/data_pump_flow/1210FIT501_99_09-2019_07-2020.zip",
    (
        "maaspoort",
        False,
        False,
    ): "../../data/sewer_data_db/data_pump_flow/1210FIT501_99.zip",
    (
        "de_rompert",
        False,
        True,
    ): "../../data/sewer_data_db/data_pump_flow/1210FIT501_99_09-2019_07-2020.zip",
    (
        "de_rompert",
        False,
        False,
    ): "../../data/sewer_data_db/data_pump_flow/1210FIT501_99.zip",
}

dir_HEMOR_level_all = "../../data/sewer_data_db/data_pump_level"
pump_level_cols = {
    "helftheuvelweg": "003: Helftheuvelweg Niveau (cm)",
    "engelerschans": "004: Engelerschans Niveau trend niveau DWA(cm)",
    "maaspoort": "006: Maaspoort Niveau actueel (1&2)(cm)",
    "oude_engelenseweg": "002: Oude Engelenseweg Niveau actueel (1&2)(cm)",
    "de_rompert": "005: De Rompert Niveau (cm)",
}

def GetRainfallByPumps(lstNames):
    '''
    summarized by group 5 last year
    '''

    lstNames = [x.lower() for x in lstNames]

    dir_areas = "../asset/data/upstream_rain_locations"
    dir_rain_historical = "../../data/rainfall/rain_timeseries"

    dct = dict()

    for name in lstNames:
        if name in pumpStations:
            file = f"pumps_for_{name}.txt"
            with open(os.path.join(dir_areas, file)) as f:
                sewers = f.readlines()
                sewers = [x.strip() for x in sewers]
                dct[name] = sewers

    setAreas = set([y for x in dct.values() for y in x])

    df_rain_historical = ParseEndTimeToIndex(
        ReadAllCsvToDf(dir_rain_historical, skiprows=2), "Eind", "%d-%m-%Y %H:%M:%S"
    )[setAreas]

    return df_rain_historical, dct



def DatetimeRange(series, delta):
    start, end = series[0], series[-1]
    current = start
    while current <= end:
        yield current
        current += delta


def RoundDataPerMinuteStep(df, isLevel):
    dfCopy = df.copy()
    dfCopy["time"] = dfCopy.index.round("Min")
    dfCopy = dfCopy.set_index("time")
    dfCopy = dfCopy[~dfCopy.index.duplicated(keep="last")]

    newIndex = [dt for dt in DatetimeRange(dfCopy.index, timedelta(minutes=1))]
    dfNew = pd.DataFrame(index=newIndex)
    dfNew = dfNew.join(dfCopy, how="left")
    return (
        dfNew.interpolate(method="linear") if isLevel else dfNew.fillna(method="ffill")
    )


def ReadAllCsvToDf(directory, sep=",", skiprows=0):
    df = pd.DataFrame()
    for file in tqdm(os.listdir(directory)):
        if file.endswith(".csv"):
            with open(os.path.join(directory, file)) as f:
                df = df.append(
                    pd.read_csv(f, sep=sep, skiprows=skiprows, low_memory=False)
                )
    return df


def ReadZipToDf(directory):
    zip_file = zipfile.ZipFile(directory, "r")
    df = pd.DataFrame()
    for i in tqdm(zip_file.namelist()):
        df = df.append(pd.read_csv(zip_file.open(i)))
    return df


def ParseEndTimeToIndex(df, col, format):
    df[col] = df[col].apply(pd.to_datetime, format=format)
    df.rename(columns={col: "end"}, inplace=True)
    df.set_index("end", inplace=True)
    df.sort_index(inplace=True)
    return df[~df.index.duplicated(keep="first")]


def dmY2Ymd(string):
    date = string.split("-")
    if int(date[0]) < 100:
        date[0], date[-1] = date[-1], date[0]
        return "-".join(date)
    return string


def WaterLevelSmoothing(df_input):
    """
    as the water level is discretedly scaled, such that the water level remain the same
    in a interval, this function is for smoothing the data so that water level in/decreasing
    gradually in each datapoint
    """
    df = df_input.copy()
    df["level_diff"] = df.diff()["level_value"]
    df["level_value_s"] = df["level_value"]
    i = 0
    j = 0
    lastSign = 0
    lastNegDiff = 0
    idx_arr = df.index
    for idx, row in tqdm(df.iterrows()):
        if row["level_diff"] > 0:
            if lastSign != 1:
                i += 1
            diff_step = row["level_diff"] / (j - i)
            for m in range(i + 1, j):
                df.at[idx_arr[m], "level_value_s"] += diff_step * (m - i)
            lastSign = 1
            i = j
        elif row["level_diff"] < 0:
            if lastSign == -1:
                diff_step = lastNegDiff / (j - i + 1)
                for m in range(i + 1, j - 1):
                    df.at[idx_arr[m], "level_value_s"] -= diff_step * (j - m)
            else:
                lastSign = -1
            lastNegDiff = row["level_diff"]
            i = j - 1
        j += 1
    df["level_diff_s"] = df.diff()["level_value_s"]
    return df


def MarkDryWeather(df_input, location):
    """
    dry weather  = precedent 24hr and current hr were no rain
    """
    df = df_input.copy()
    df[f"{location}_dry"] = False
    for i in range(24, len(df[location])):
        if df[location][i - 24 : i + 1].sum() == 0:
            df[f"{location}_dry"][i] = True
    return df


def GetSewageAndRainfall():
    df_sewage_area = pd.read_csv(
        "../../data/sewer_model/surfacearea_sewersystems.txt", sep=";"
    )

    df_sewage_pump_connection = pd.read_excel(
        "../../data/sewer_model/20180717_dump riodat rioleringsdeelgebieden_DB.xlsx",
        skiprows=8,
        header=1,
    )

    df_sewage_pump_connection["Area"] = 0
    df_sewage_pump_connection["NAAMRGD"] = np.nan
    df_sewage_pump_connection.loc[
        df_sewage_pump_connection["Code"].isin(df_sewage_area["RGDIDENT"]),
        ["Area", "NAAMRGD"],
    ] = df_sewage_area[["Area", "NAAMRGD"]]

    df_sewage_pump_connection["NAAMRGD"] = df_sewage_pump_connection[
        "NAAMRGD"
    ].str.replace("'", "")

    dir_rain_historical = "../../data/rainfall/rain_timeseries"
    df_rain_historical = ParseEndTimeToIndex(
        ReadAllCsvToDf(dir_rain_historical, skiprows=2), "Eind", "%d-%m-%Y %H:%M:%S"
    )
    df_rain_historical = df_rain_historical[
        [
            x
            for x in df_rain_historical.columns.values
            if x
            in [y for y in df_sewage_pump_connection["NAAMRGD"] if not pd.isnull(y)]
        ]
    ]

    return df_sewage_pump_connection, df_rain_historical


def GetData(name, isLevel, isNew):
    name = name.lower()
    prefix = "level" if isLevel else "flow"
    renameCols = {
        "historianKwaliteit": f"{prefix}_quality",
        "hstWaarde": f"{prefix}_value",
    }
    if isNew:
        colEnd = "dem"
        selectedCols = ["hstWaarde"]
    else:
        colEnd = "datumEindeMeting"
        selectedCols = ["historianKwaliteit", "hstWaarde"]

    if (name in pumpStations and not isLevel) or (name in ["bokhoven", "haarsteeg"]):
        df = ParseEndTimeToIndex(
            ReadZipToDf(data[(name, isLevel, isNew)]), colEnd, "%Y-%m-%d %H:%M:%S"
        )[selectedCols]
        df.rename(columns=renameCols, inplace=True)
        if not isLevel:
            df[f"{prefix}_value"]=df[f"{prefix}_value"]/60
        return df
    elif (
        name
        in [
            "helftheuvelweg",
            "engelerschans",
            "oude_engelenseweg",
            "maaspoort",
            "de_rompert",
        ]
        and isLevel
    ):
        df = ReadAllCsvToDf(dir_HEMOR_level_all, sep=";")
        df["Datum"] = df["Datum"].apply(dmY2Ymd)
        df["end"] = df["Datum"] + " " + df["Tijd"]
        df = ParseEndTimeToIndex(df, "end", "%Y-%m-%d %H:%M:%S")[
            list(pump_level_cols.values())
        ]
        df = df.rename(
            columns=dict(
                [(value, key) for key, value in pump_level_cols.items()], inplace=True
            )
        )
        df = df.apply(lambda x: x.str.replace(",", "."))
        if isNew:
            df = df[[name]]["2019-09-01 00:00:00":].rename(
                columns={name: "level_value"})
        else:
            df = df[[name]][:"2019-09-01 00:00:00"].rename(
                columns={name: "level_value"})
        return df.astype(float)



def InitDict():
    return {
        "up1": None,
        "up2": 0,
        "level_up": 0,
        "out": [],
        "start_level": 0,
        "end_level": 0,
    }


def GetSpecification(lstNames):
    for name in lstNames:
        if name not in pumpStations:
            print(f"{name} is not correct")
            return None

    print("loading rainfall")
    df_rain_historical, dct_areas = GetRainfallByPumps(lstNames)

    output = {}
    for name in dct_areas:

        print(f"!!! {name} === start !!!")

        location = dct_areas[name][0]
        target_col = f"{location}_dry"

        print("checking dry weather")
        df_rain_historical = MarkDryWeather(df_rain_historical, location)

        print("loading level data")
        level_old = GetData(name, True, False)
        if name not in ["bokhoven", "haarsteeg"]:
            level_old = RoundDataPerMinuteStep(level_old, isLevel=True)

        minLevel = min(level_old["level_value"])  # to be returned

        print("loading flow data")
        flow_old = GetData(name, False, False)

        df_join = level_old.join(flow_old, how="inner")

        print("smoothing level data")
        df_join = WaterLevelSmoothing(df_join)

        print("mark dry weather hour")
        df_join["end_hour"] = df_join.index.ceil("h")
        df_join["dry"] = False
        for idx, row in tqdm(df_join.iterrows()):
            if df_rain_historical.at[row["end_hour"], target_col]:
                df_join.at[idx, "dry"] = True

        if name in ["bokhoven", "haarsteeg"]:
            df_dry = df_join[
                (df_join["dry"])
                & (df_join["level_quality"] == 100)
                & (df_join["flow_quality"] == 100)
            ].copy()
        else:
            df_dry = df_join[(df_join["dry"]) & (df_join["flow_quality"] == 100)][
                "2018-04-25 00:00:00":
            ].copy()

        df_dry["adj_level_diff_s"] = df_dry["level_diff_s"]
        df_dry.loc[(df_dry["adj_level_diff_s"] < 0), "adj_level_diff_s"] = np.nan

        df_dry["adj_level_diff_s"] = (
            df_dry["adj_level_diff_s"].ffill() + df_dry["adj_level_diff_s"].bfill()
        ) / 2
        df_dry["adj_level_diff_s"] = df_dry["adj_level_diff_s"].fillna(0)

        print("calculating volume of water per meter level")
        td = timedelta(minutes=1)
        t0 = datetime(1900, 1, 1)
        lst_m3_per_m = []
        dct = InitDict()
        for idx, row in tqdm(df_dry.iterrows()):
            if idx - t0 > td:  # if time is continuous
                dct = InitDict()
            else:
                if row["level_diff_s"] < 0 and dct["up1"] != 0:  # down after up
                    dct["out"].append(row["flow_value"])
                if len(dct["out"]) == 0:  # no pump-out be recorded
                    if row["level_diff_s"] > 0:
                        if dct["up1"] is None:
                            dct["start_level"] = row["level_value_s"]
                        dct["up1"] = row["level_diff_s"]

                else:
                    if row["level_diff_s"] > 0:
                        dct["up2"] = row["level_diff_s"]
                        sss = np.sum(dct["out"]) / (
                            dct["level_up"] - dct["end_level"] + dct["start_level"]
                        )
                        lst_m3_per_m.append(sss)  #  m3/m water-level
                        dct = InitDict()
            t0 = idx
            dct["end_level"] = row["level_value_s"]
            dct["level_up"] += row["adj_level_diff_s"]

        avg_m3_per_m = np.average(lst_m3_per_m)
        output[name] = {"min_level": minLevel, "m3_per_m": avg_m3_per_m}
    return output



def GenerateMLdata(name, rate, min_level, df_rain_historical, dct_areas):

    df_level_new = GetData(name, True, True)
    df_flow_new = GetData(name, False, True)

    df_level_new = RoundDataPerMinuteStep(df_level_new, isLevel=True)
    df_flow_new = RoundDataPerMinuteStep(df_flow_new, isLevel=False)

    df_level_new = WaterLevelSmoothing(df_level_new)

    location = dct_areas[name][0]
    target_col = f"{location}_dry"
    df_rain_historical = MarkDryWeather(df_rain_historical, location)

    df_level_new["end_hour"] = df_level_new.index.ceil("h")
    df_level_new["dry"] = False
    for idx, row in tqdm(df_level_new.iterrows()):
        if df_rain_historical.at[row["end_hour"], target_col]:
            df_level_new.at[idx, "dry"] = True

    df_join = df_level_new.join(df_flow_new, how="inner")
    df_dry = df_join[df_join["dry"]].copy()
    df_dry["adj_level_diff_s"] = df_dry["level_value"].diff()
    df_dry.loc[(df_dry["adj_level_diff_s"] < 0), "adj_level_diff_s"] = np.nan
    df_dry["adj_level_diff_s"] = (
        df_dry["adj_level_diff_s"].ffill() + df_dry["adj_level_diff_s"].bfill()
    ) / 2
    df_dry["adj_level_diff_s"] = df_dry["adj_level_diff_s"].fillna(0)
    df_dry["in_flow_vol"] = df_dry["adj_level_diff_s"] * rate
    data = df_dry.groupby("end_hour", as_index=True).sum()[
        ["in_flow_vol", "flow_value"]
    ][:-1]
    data["level_value"] = np.nan
    for idx, row in tqdm(data.iterrows()):
        data.at[idx, "level_value"] = df_level_new.at[idx, "level_value"]
    data["last_in_flow_vol"] = data["in_flow_vol"].shift()
    data["vol_remain"] = (data["level_value"] - min_level) * rate

    return data.rename(columns=lambda x: f"{name}_{x}")