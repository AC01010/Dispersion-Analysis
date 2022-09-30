import math
import requests
def API_launch_site(latitude,longitude, zoom_level,length_level,height_level,scale_level,format_type,maptype,key):
    url = "https://maps.googleapis.com/maps/api/staticmap"
    d = {
        "center": "{},{}".format(latitude,longitude),
        "zoom": zoom_level,
        "size": "{}x{}".format(length_level,height_level),
        "scale": scale_level,
        "format": format_type,
        "maptype": maptype,
        "key": key
    }
    response = requests.get(url, params=d)
    with open("staticmap.png", "wb") as f:
        f.write(response.content)
    print(response.url)    

latitude="33.9127"
longitude="-84.9417"
meter_per_pixel = 3
# The relationship between the zoom and the map scale is given by https://stackoverflow.com/questions/9356724/google-map-api-zoom-range 
# meter_per_pixel=156543.03392 * math.cos(math.radians(float(latitude))) / math.pow(2, float(zoom_level))
zoom_level = str(int(math.log(math.cos(math.radians(float(latitude)))/meter_per_pixel*156543.03392,2))) # Invert meter_per_pixel to zoom_level
print(zoom_level)
length_level="1000"
height_level="1000"
scale_level="2"
format_type="png"
maptype="satellite"
key="AIzaSyB2QS2FISG_FjyxpniKZcHXkBlQAOpkQ6c" #if it doesn't work you will need to create a new key by creating a Google Coloud Platform account
API_launch_site(latitude,longitude, zoom_level,length_level,height_level,scale_level,format_type,maptype,key)
print ("The image has", length_level,"x",height_level," pixels that correspond to ", float(length_level)*meter_per_pixel,"x", float(height_level)*meter_per_pixel, "meters.")