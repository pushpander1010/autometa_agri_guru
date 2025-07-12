from langchain.tools import tool
from datetime import date, timedelta
import requests
import pandas as pd
import os
from openmeteo_requests import Client
import requests_cache
from retry_requests import retry
from dotenv import load_dotenv

load_dotenv()

# === Static State Coordinates Dictionary ===
STATE_COORDINATES = {
    "andhra pradesh": (15.9129, 79.7400),
    "arunachal pradesh": (28.2180, 94.7278),
    "assam": (26.2006, 92.9376),
    "bihar": (25.0961, 85.3131),
    "chhattisgarh": (21.2787, 81.8661),
    "goa": (15.2993, 74.1240),
    "gujarat": (22.2587, 71.1924),
    "haryana": (29.0588, 76.0856),
    "himachal pradesh": (31.1048, 77.1734),
    "jharkhand": (23.6102, 85.2799),
    "karnataka": (15.3173, 75.7139),
    "kerala": (10.8505, 76.2711),
    "madhya pradesh": (22.9734, 78.6569),
    "maharashtra": (19.7515, 75.7139),
    "manipur": (24.6637, 93.9063),
    "meghalaya": (25.4670, 91.3662),
    "mizoram": (23.1645, 92.9376),
    "nagaland": (26.1584, 94.5624),
    "odisha": (20.9517, 85.0985),
    "punjab": (31.1471, 75.3412),
    "rajasthan": (27.0238, 74.2179),
    "sikkim": (27.5330, 88.5122),
    "tamil nadu": (11.1271, 78.6569),
    "telangana": (18.1124, 79.0193),
    "tripura": (23.9408, 91.9882),
    "uttar pradesh": (26.8467, 80.9462),
    "uttarakhand": (30.0668, 79.0193),
    "west bengal": (22.9868, 87.8550),
    "delhi": (28.7041, 77.1025),
    "jammu and kashmir": (33.7782, 76.5762),
    "ladakh": (34.1526, 77.5770)
}


@tool
def get_curr_loc_tool() -> dict:
    """
    Returns the user's current geolocation based on their IP address.

    Output:
        Dictionary with keys like:
        - latitude, longitude
        - cityName, regionName (state), countryName
        - ipAddress, continent, etc.
        
    Example:
        {
            "latitude": 30.7363,
            "longitude": 76.7884,
            "cityName": "Chandigarh",
            "regionName": "Chandigarh",
            "countryName": "India"
        }
    """
    try:
        ip = get_public_ip()
        url = f"https://free.freeipapi.com/api/json/{ip}"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            return response.json()
        else:
            return {
                'ipVersion': 4,
                'ipAddress': '1.187.225.205',
                'latitude': 30.7363,
                'longitude': 76.7884,
                'countryName': 'India',
                'regionName': 'Chandigarh',
                'cityName': 'Chandigarh',
                'currencies': ['INR']
            }
    except Exception as e:
        return {"error": f"Location fetch failed: {str(e)}"}

def get_public_ip():
    """Helper function to get public IP address."""
    try:
        response = requests.get("https://api.ipify.org?format=json", timeout=5)
        if response.status_code == 200:
            return response.json()["ip"]
    except:
        pass
    return "1.187.225.205"  # fallback IP


# === Tool: Get Coordinates of a State ===
@tool
def get_state_coordinates(state: str) -> dict:
    """
    Given an Indian state name, returns its latitude and longitude.

    Args:
        state (str): Full name of the Indian state (e.g. 'Punjab')

    Returns:
        dict: Dictionary containing 'latitude' and 'longitude'
              or an error message if not found.
    """
    key = state.strip().lower()
    if key in STATE_COORDINATES:
        lat, lon = STATE_COORDINATES[key]
        return {"state": state.title(), "latitude": lat, "longitude": lon}
    return {"error": f"Coordinates not found for state '{state}'"}

# === Tool: Get Seasonal Weather Summary ===
@tool
def get_seasonal_weather_data(latitude: float, longitude: float) -> str:
    """
    Provides a quarterly summary (Q1-Q4) of average temperature, rainfall etc.
    over the last ~3 years for a specific latitude and longitude.

    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location

    Returns:
        str: Human-readable summary of weather data by quarter
    """
    try:
        today = date.today()
        start_date = (today - timedelta(days=1000)).isoformat()
        end_date = today.isoformat()

        url = "https://meteostat.p.rapidapi.com/point/monthly"
        querystring = {
            "lat": latitude, "lon": longitude,
            "start": start_date, "end": end_date
        }
        headers = {
            "x-rapidapi-key": os.getenv("x-rapidapi-key"),
            "x-rapidapi-host": "meteostat.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            return "No weather data available."

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["quarter"] = df["date"].dt.to_period("Q").astype(str)
        grouped = df.groupby("quarter").agg({
            "tavg": "mean", "tmin": "min", "tmax": "max", "prcp": "sum"
        }).reset_index()

        summary = f"**Quarterly Weather Summary for lat {latitude}, lon {longitude}**\n"
        for _, row in grouped.iterrows():
            summary += (
                f"**{row['quarter']}**:\n"
                f"- Avg Temp: {round(row['tavg'], 1)}°C\n"
                f"- Min Temp: {round(row['tmin'], 1)}°C\n"
                f"- Max Temp: {round(row['tmax'], 1)}°C\n"
                f"- Rainfall: {round(row['prcp'], 1)} mm\n"
            )
        return summary

    except Exception as e:
        return f"[Error] {str(e)}"

# === Tool: Get Soil Properties ===
@tool
def get_soil_properties(latitude: float, longitude: float) -> str:
    """
    Gets soil data from SoilGrids v2.0 for a given latitude and longitude.

    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location

    Returns:
        str: Human-readable soil property summary with values by depth
    """
    try:
        url = (
            f"https://rest.isric.org/soilgrids/v2.0/properties/query?"
            f"lon={longitude}&lat={latitude}&"
            f"property=bdod&property=cec&property=cfvo&property=clay&property=nitrogen&"
            f"property=ocd&property=ocs&property=phh2o&property=sand&property=silt&"
            f"property=soc&property=wv0010&property=wv0033&property=wv1500"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        layers = data.get("properties", {}).get("layers", [])
        if not layers:
            return f"No soil data found for lat {latitude}, lon {longitude}."

        result = f"**Soil Properties at Latitude {latitude}, Longitude {longitude}**\n\n"
        for layer in layers:
            name = layer.get("name", "Unknown")
            unit = layer.get("unit_measure", {}).get("target_units", "")
            result += f"**{name.upper()}** ({unit})\n"
            for d in layer.get("depths", []):
                label = d.get("label")
                mean_val = d.get("values", {}).get("mean")
                if mean_val:
                    result += f"- {label}: {mean_val} {unit}\n"
            result += "\n"

        return result.strip()
    except Exception as e:
        return f"[Error] {str(e)}"

# === Tool: Get Mandi (Farm) Prices ===
@tool
def get_farm_prices(stateName: str, commodity: str) -> str:
    """
    Retrieves mandi prices for a given farm commodity in the specified state.

    Args:
        stateName (str): State name in title case (e.g. 'Punjab')
        commodity (str): Name of farm produce (e.g. 'Wheat')

    Returns:
        str: Formatted mandi price summary
    """
    try:
        api = os.getenv("FARM_PRICE_API")
        url = (
            "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
            f"?api-key={api}&format=json&filters%5BState%5D={stateName}"
            f"&filters%5BCommodity%5D={commodity}"
        )

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        records = response.json().get("records", [])

        if not records:
            return f"No mandi price data found for {commodity} in {stateName}."

        output = f"**Mandi Prices for {commodity} in {stateName}:**\n"
        for rec in records[:5]:
            output += (
                f"- {rec.get('Market', 'N/A')} ({rec.get('Arrival_Date', 'N/A')}): "
                f"₹{rec.get('Modal_Price', 'N/A')} "
                f"(Min: ₹{rec.get('Min_Price', 'N/A')}, Max: ₹{rec.get('Max_Price', 'N/A')}, "
                f"Grade: {rec.get('Grade', 'N/A')}, Variety: {rec.get('Variety', 'N/A')})\n"
            )
        return output
    except Exception as e:
        return f"[Error] {str(e)}"

# === Tool: Get Weather & Soil Summary (Last 80 Days) ===
@tool
def get_weather_and_soil_data(latitude: float, longitude: float) -> str:
    """
    Summarizes 80-day weather data for the provided coordinates using Open-Meteo API.

    Args:
        latitude (float): Latitude
        longitude (float): Longitude

    Returns:
        str: Summary with avg temp, rain, sunshine, evapotranspiration, etc.
    """
    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = Client(session=retry_session)

        params = {
            "latitude": latitude, "longitude": longitude,
            "start_date": (date.today() - timedelta(days=80)).isoformat(),
            "end_date": date.today().isoformat(),
            "daily": [
                "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "precipitation_sum", "rain_sum", "sunshine_duration",
                "uv_index_max", "et0_fao_evapotranspiration"
            ]
        }

        responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        daily = responses[0].Daily()
        df = pd.DataFrame({
            "temp_max": daily.Variables(0).ValuesAsNumpy(),
            "temp_min": daily.Variables(1).ValuesAsNumpy(),
            "temp_mean": daily.Variables(2).ValuesAsNumpy(),
            "precip": daily.Variables(3).ValuesAsNumpy(),
            "rain": daily.Variables(4).ValuesAsNumpy(),
            "sunshine": daily.Variables(5).ValuesAsNumpy(),
            "uv": daily.Variables(6).ValuesAsNumpy(),
            "evapo": daily.Variables(7).ValuesAsNumpy(),
        })

        return (
            f"**Weather & Soil Summary (Last 80 Days at {latitude}, {longitude})**\n"
            f"- Avg Temp: {round(df['temp_mean'].mean(), 1)}°C\n"
            f"- Max Temp: {round(df['temp_max'].max(), 1)}°C\n"
            f"- Min Temp: {round(df['temp_min'].min(), 1)}°C\n"
            f"- Total Rain: {round(df['rain'].sum(), 1)} mm\n"
            f"- Total Precipitation: {round(df['precip'].sum(), 1)} mm\n"
            f"- Total Sunshine: {round(df['sunshine'].sum()/60, 1)} hrs\n"
            f"- Avg UV Index: {round(df['uv'].mean(), 1)}\n"
            f"- Avg Evapotranspiration: {round(df['evapo'].mean(), 1)} mm/day"
        )
    except Exception as e:
        return f"[Error] {str(e)}"
