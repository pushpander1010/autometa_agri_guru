import requests
import os
from datetime import date, timedelta
from openmeteo_requests import Client
import pandas as pd
import requests_cache
from retry_requests import retry
from dotenv import load_dotenv
from langchain.tools import tool
from typing import Optional

load_dotenv()


@tool
def get_curr_loc_tool() -> dict:
    """Get user's current location (lat, lon, city, state, country)."""
    try:
        ip = get_public_ip()
        url = f"https://free.freeipapi.com/api/json/{ip}"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            return response.json()
        else:
            # fallback default location
            return {
                'ipVersion': 4,
                'ipAddress': '1.187.225.205',
                'latitude': 30.7363,
                'longitude': 76.7884,
                'countryName': 'India',
                'countryCode': 'IN',
                'capital': 'New Delhi',
                'phoneCodes': [91],
                'timeZones': ['Asia/Kolkata'],
                'cityName': 'Chandigarh',
                'regionName': 'Chandigarh',
                'continent': 'Asia',
                'continentCode': 'AS',
                'currencies': ['INR'],
                'languages': ['hi', 'en'],
                'asn': '45271',
                'asnOrganization': 'Vodafone Idea Ltd. (VIL)'
            }
    except Exception as e:
        return {"error": f"Failed to get location: {str(e)}"}

def get_curr_loc() -> dict:
    """Get user's current location (lat, lon, city, state, country) based on their public IP address."""
    try:
        ip = get_public_ip()
        url = f"https://free.freeipapi.com/api/json/{ip}"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            return response.json()
        else:
            # fallback default location
            return {
                'ipVersion': 4,
                'ipAddress': '1.187.225.205',
                'latitude': 30.7363,
                'longitude': 76.7884,
                'countryName': 'India',
                'countryCode': 'IN',
                'capital': 'New Delhi',
                'phoneCodes': [91],
                'timeZones': ['Asia/Kolkata'],
                'cityName': 'Chandigarh',
                'regionName': 'Chandigarh',
                'continent': 'Asia',
                'continentCode': 'AS',
                'currencies': ['INR'],
                'languages': ['hi', 'en'],
                'asn': '45271',
                'asnOrganization': 'Vodafone Idea Ltd. (VIL)'
            }
    except Exception as e:
        return {"error": f"Failed to get location: {str(e)}"}
    
def get_public_ip():
    ip="24.48.0.1" #dummy ip
    response = requests.get("https://api.ipify.org?format=json")
    if response.status_code == 200:
        return response.json()['ip']
    else:
        return ip

def format_weather_data(api_response):
    daily_data = api_response.get("daily", {})
    dates = daily_data.get("time", [])
    max_temps = daily_data.get("temperature_2m_max", [])
    min_temps = daily_data.get("temperature_2m_min", [])

    formatted = []
    for date, max_t, min_t in zip(dates, max_temps, min_temps):
        if max_t is None or min_t is None:
            continue  # skip missing data
        formatted.append({
            "date": date,
            "max_temp": f"{max_t}°C",
            "min_temp": f"{min_t}°C"
        })

    return formatted  # limit if needed

def get_weather_history(latitude,longitude,start_date,end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max",
                   "temperature_2m_min",
                   "soil_temperature_0cm", "soil_temperature_6cm",
            "soil_temperature_18cm", "soil_temperature_54cm",
            "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
            "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm"],
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return format_weather_data(response.json())
    except requests.RequestException as e:
        print("Error:", e)
        return None

def expand_weather_keys(data):
    key_map = {
        'date': 'Date',
        'tavg': 'Average Temperature (°C)',
        'tmin': 'Minimum Temperature (°C)',
        'tmax': 'Maximum Temperature (°C)',
        'prcp': 'Precipitation (mm)',
        'wspd': 'Wind Speed (km/h)',
        'pres': 'Pressure (hPa)',
        'tsun': 'Sunshine Duration (minutes)'
    }

    expanded_data = []

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(f"[Warning] Skipping non-dict entry at index {i}: {entry}")
            continue

        formatted_entry = {}
        for key, value in entry.items():
            full_key = key_map.get(key, key)  # fallback to original key
            if value is not None:
                formatted_entry[full_key] = value
        expanded_data.append(formatted_entry)

    return expanded_data

@tool
def get_seasonal_weather_data() -> str:
    """Returns weather summary grouped by quarter (Q1-Q4) for the past ~3 years at the user's location."""
    try:
        currLoc = get_curr_loc()
        today = date.today()
        start_date = (today - timedelta(days=1000)).isoformat()
        end_date = today.isoformat()

        url = "https://meteostat.p.rapidapi.com/point/monthly"
        querystring = {
            "lat": currLoc["latitude"],
            "lon": currLoc["longitude"],
            "start": start_date,
            "end": end_date
        }
        headers = {
            "x-rapidapi-key": os.getenv("x-rapidapi-key"),
            "x-rapidapi-host": "meteostat.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        response.raise_for_status()
        data = response.json().get("data", [])

        if not data:
            return "No weather data available for this location."

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["quarter"] = df["date"].dt.to_period("Q").astype(str)

        grouped = df.groupby("quarter").agg({
            "tavg": "mean",
            "tmin": "min",
            "tmax": "max",
            "prcp": "sum"
        }).reset_index()

        summary_lines = [
            f"**Quarterly Weather Summary for Past ~3 Years at lat {currLoc['latitude']}, lon {currLoc['longitude']}**\n"
        ]

        for _, row in grouped.iterrows():
            if not row.all():
                continue
            summary_lines.append(
                f"**{row['quarter']}**:\n"
                f"- Avg Temp: {round(row['tavg'], 1)}°C\n"
                f"- Min Temp: {round(row['tmin'], 1)}°C\n"
                f"- Max Temp: {round(row['tmax'], 1)}°C\n"
                f"- Total Rainfall: {round(row['prcp'], 1)} mm\n"
            )

        return "\n".join(summary_lines)

    except requests.exceptions.RequestException as e:
        return f"[Error] API request failed: {str(e)}"
    except Exception as e:
        return f"[Error] Unexpected error: {str(e)}"

@tool
def get_farm_prices(commodity: str) -> str:
    """Get mandi prices for a given farm commodity based on the user's current state. Returns data as a formatted string."""
    try:
        currLoc = get_curr_loc()
        api = os.getenv("FARM_PRICE_API")
        stateName = currLoc["regionName"]

        url = (
            f"https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
            f"?api-key={api}&format=json&filters%5BState%5D={stateName}&filters%5BCommodity%5D={commodity}"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        records = data.get("records", [])
        #print(records)
        if not records:
            return f"No mandi price data found for {commodity} in {stateName}."

        # Format records for display
        output = f"Mandi Prices for {commodity} in {stateName}:\n"
        for rec in records:  # show top 5 entries only
            market = rec.get("Market", "N/A")
            min_price = rec.get("Min_Price", "N/A")
            max_price = rec.get("Max_Price", "N/A")
            modal_price = rec.get("Modal_Price", "N/A")
            date = rec.get("Arrival_Date", "N/A")
            grade=rec.get("Grade", "N/A")
            variety=rec.get("Variety", "N/A")
            output += f"- {market} ({date}): ₹{modal_price} (Min: ₹{min_price}, Max: ₹{max_price}, Grade: {grade}, Variety: {variety})\n"

        return output

    except requests.exceptions.RequestException as e:
        return f"[Error] Network or API failure: {str(e)}"

    except Exception as e:
        return f"[Error] Unexpected issue: {str(e)}"

@tool
def get_soil_properties() -> str:
    """Get detailed soil properties using SoilGrids v2.0."""
    try:
        currLoc=get_curr_loc()
        lat, lon = currLoc['latitude'], currLoc['longitude']  # You can modify or pass dynamically
        url = (
            f"https://rest.isric.org/soilgrids/v2.0/properties/query?"
            f"lon={lon}&lat={lat}&property=bdod&property=cec&property=cfvo&property=clay&property=nitrogen"
            f"&property=ocd&property=ocs&property=phh2o&property=sand&property=silt&property=soc"
            f"&property=wv0010&property=wv0033&property=wv1500"
        )

        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        properties = data.get("properties", {}).get("layers", [])
        if not properties:
            return f"No soil properties found at lat {lat}, lon {lon}."

        output = f"**Soil Properties at Latitude {lat}, Longitude {lon}**\n\n"

        for prop in properties:
            name = prop.get("name", "Unknown")
            unit = prop.get("unit_measure", {}).get("target_units", "")
            output += f"**{name.upper()}** ({unit})\n"
            for depth in prop.get("depths", []):
                label = depth.get("label", "")
                values = depth.get("values", {})
                mean_val = values.get("mean")
                if mean_val is not None:
                    output += f" - Depth {label}: {mean_val} {unit}\n"
            output += "\n"

        return output.strip()

    except requests.exceptions.RequestException as e:
        return f"[Error] Failed to fetch soil data: {str(e)}"
    except Exception as e:
        return f"[Error] Unexpected error: {str(e)}"


@tool
def get_weather_and_soil_data() -> str:
    """Summarizes the past 80 days of weather and soil data for the user's location in a natural-language format."""
    try:
        # Setup caching and retry
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = Client(session=retry_session)

        # Get location
        currLoc = get_curr_loc()
        params = {
            "latitude": currLoc["latitude"],
            "longitude": currLoc["longitude"],
            "start_date": (date.today() - timedelta(days=80)).isoformat(),
            "end_date": date.today().isoformat(),
            "daily": [
                "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "precipitation_sum", "rain_sum", "sunshine_duration",
                "uv_index_max", "et0_fao_evapotranspiration"
            ]
        }

        # Fetch data
        responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        response = responses[0]
        daily = response.Daily()

        # Date range
        dates = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )

        # Build DataFrame
        df = pd.DataFrame({
            "date": dates,
            "temp_max": daily.Variables(0).ValuesAsNumpy(),
            "temp_min": daily.Variables(1).ValuesAsNumpy(),
            "temp_mean": daily.Variables(2).ValuesAsNumpy(),
            "precip": daily.Variables(3).ValuesAsNumpy(),
            "rain": daily.Variables(4).ValuesAsNumpy(),
            "sunshine": daily.Variables(5).ValuesAsNumpy(),
            "uv": daily.Variables(6).ValuesAsNumpy(),
            "evapotranspiration": daily.Variables(7).ValuesAsNumpy()
        })

        df.dropna(inplace=True)

        # Calculate summary stats
        summary = {
            "days": len(df),
            "avg_temp": round(df["temp_mean"].mean(), 1),
            "max_temp": round(df["temp_max"].max(), 1),
            "min_temp": round(df["temp_min"].min(), 1),
            "total_rain": round(df["rain"].sum(), 1),
            "total_precip": round(df["precip"].sum(), 1),
            "sunshine_hrs": round(df["sunshine"].sum() / 60, 1),  # minutes to hours
            "avg_uv": round(df["uv"].mean(), 1),
            "avg_evapo": round(df["evapotranspiration"].mean(), 1)
        }

        return (
            f"**Weather & Soil Summary (Last {summary['days']} Days)**\n"
            f"- **Average Temperature:** {summary['avg_temp']}°C\n"
            f"- **Maximum Temperature Recorded:** {summary['max_temp']}°C\n"
            f"- **Minimum Temperature Recorded:** {summary['min_temp']}°C\n"
            f"- **Total Rainfall:** {summary['total_rain']} mm\n"
            f"- **Total Precipitation (Rain + Showers):** {summary['total_precip']} mm\n"
            f"- **Total Sunshine Duration:** {summary['sunshine_hrs']} hours\n"
            f"- **Average UV Index:** {summary['avg_uv']}\n"
            f"- **Average Evapotranspiration:** {summary['avg_evapo']} mm/day"
        )

    except Exception as e:
        return f"[Error] Failed to fetch weather and soil data: {str(e)}"

