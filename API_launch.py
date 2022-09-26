import math
def API_launch_site(latitude,longitude, zoom_level,length_level,height_level,scale_level,format_type,maptype,key):
    # ALL INPUTS ARE STRINGS
    # Longitude and Latitude must be string inputs of 6 decimal digits precission, zoom level ranges from 1 (world zoom) to 20 (buildings zoom)
    API="https://maps.googleapis.com/maps/api/staticmap?"
    # We now add the location parameters to the URL: center ({latitude, longitude} coordinates) and zoom (we specify zoom to streets level, with value 15):
    API=API+"center="+latitude+","+longitude+"&zoom="+zoom_level
    # Now we add the map parameters:size, scale (default is scale=1, for high resolution use scale=2),format (.png is default),maptype (select among roadmap,satellite,hybrid, and terrain)
    API=API+"&size="+length_level+"x"+height_level+"&scale="+scale_level+"&format="+format_type+"&maptype="+maptype
    # Finally, the key is added:
    API=API+"&key="+key
    print(API)

latitude="35.3505823"
longitude="-117.8108475"
zoom_level="10"
# The relationship between the zoom and the map scale is given by https://stackoverflow.com/questions/9356724/google-map-api-zoom-range 
meter_per_pixel=156543.03392 * math.cos(float(latitude)* math.pi / 180) / math.pow(2, float(zoom_level))
length_level="400"
height_level="400"
scale_level="2"
format_type="png"
maptype="satellite"
key="AIzaSyB2QS2FISG_FjyxpniKZcHXkBlQAOpkQ6c" #if it doesn't work you will need to create a new key by creating a Google Coloud Platform account
API_launch_site(latitude,longitude, zoom_level,length_level,height_level,scale_level,format_type,maptype,key)
print ("The image has", length_level,"x",height_level," pixels that correspond to ", float(length_level)*meter_per_pixel,"x", float(height_level)*meter_per_pixel, "meters.")