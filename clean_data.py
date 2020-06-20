import csv
import re
import statistics


def get_bearing(nat_lang):
    if nat_lang == "variable wind direction" or nat_lang == "Calm, no wind":
        return -1.

    cardinal = nat_lang.split(" ")[-1]
    return wind_directions.index(cardinal) * 22.5


def get_clouds(nat_lang):
    if nat_lang == "No Significant Clouds":
        return 0.

    return statistics.mean([int(item) for sublist in [number.strip("()%").split("-") for number in
                                                      re.findall("\(\d+-?\d+%\)", nat_lang)] for item in sublist])


def get_visibility(nat_lang):
    if nat_lang == "10.0 and more":
        return 10.

    return float(nat_lang)


def get_precipitation(nat_lang):
    intensity = nat_lang.split(",")[0]

    if intensity == "Light rain":
        return 1.
    elif intensity == "Rain":
        return 2.
    elif intensity == "Heavy rain":
        return 3.
    else:
        return 0.


wind_directions = ["north", "north-northeast", "north-east", "east-northeast", "east", "east-southeast", "south-east",
                   "south-southeast", "south", "south-southwest", "south-west", "west-southwest", "west",
                   "west-northwest", "north-west", "north-northwest"]


with open("data/dev_dirty.csv", "r") as read_file:
    reader = csv.reader(read_file, delimiter=";")

    with open("data/dev.csv", "w") as write_file:
        writer = csv.writer(write_file)
        writer.writerow(["datetime", "T", "TD", "P0", "U", "DD", "FF", "WW", "c", "VV"])

        prev_row = []
        for row in reversed(list(reader)):
            try:
                time = row[0]
                temp = float(row[1])
                dew_temp = float(row[12])
                pressure = float(row[2])
                humidity = float(row[4])
            except (ValueError, IndexError):
                print(row)
                continue

            if row[5] == "":
                wind = get_bearing(prev_row[5])
                row[5] = prev_row[5]
            else:
                wind = get_bearing(row[5])

            if row[6] == "":
                wind_speed = float(prev_row[6])
                row[6] = prev_row[6]
            else:
                wind_speed = float(row[6])

            precipitation = get_precipitation(row[8])

            if row[10] != "No Significant Clouds" and not re.search("\d", row[10]):
                clouds = get_clouds(prev_row[10])
                row[10] = prev_row[10]
            else:
                clouds = get_clouds(row[10])

            if row[11] == "":
                visibility = get_visibility(prev_row[11])
                row[11] = prev_row[11]
            else:
                visibility = get_visibility(row[11])

            writer.writerow([time, temp, dew_temp, pressure, humidity, wind, wind_speed, precipitation, clouds, visibility])
            prev_row = row
