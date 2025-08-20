import requests
import json

def fetch_tles():
    norad_ids = [28874, 25544, 25338, 858, 39199, 36112, 33401, 39197, 25560, 22824]
    tle_data = {}
    
    for norad in norad_ids:
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT=TLE"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            tle_lines = resp.text.strip().split('\n')
            if len(tle_lines) == 3:
                tle_data[norad] = tle_lines
        except requests.exceptions.RequestException as e:
            print(f"Error fetching TLE for NORAD {norad}: {e}")
    
    # Save to JSON file
    with open("tle_data.json", "w") as f:
        json.dump(tle_data, f, indent=2)

    print(f"Saved TLEs for {len(tle_data)} satellites.")

if __name__ == "__main__":
    fetch_tles()
