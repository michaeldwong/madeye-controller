import requests
import time
import random
# curl "http://128.112.34.129/cgi-bin/ptzctrl.cgi?ptzcmd&abs&24&20&0000&0000"
url = "http://128.112.34.129/cgi-bin/ptzctrl.cgi"

last_request = ''
url = 'http://'

urls = [
    "http://128.112.34.129/cgi-bin/ptzctrl.cgi?ptzcmd&abs&24&20&FF80&0000", # left
    "http://128.112.34.129/cgi-bin/ptzctrl.cgi?ptzcmd&abs&24&20&FE80&0070", # middle, up
    "http://128.112.34.129/cgi-bin/ptzctrl.cgi?ptzcmd&abs&24&20&FD80&0000", # right
    "http://128.112.34.129/cgi-bin/ptzctrl.cgi?ptzcmd&abs&24&20&FE80&FF70", # mddle down
    "http://128.112.34.129/cgi-bin/ptzctrl.cgi?ptzcmd&abs&24&20&FE80&0000" # mddle 
]
idx = 0
for i in range(0,100):
    url = urls[idx] 
    idx += 1
    if idx >= 5:
        idx = 0

#    c = random.random()
#    last_request = url
#    if c <= 0.25: 
#        url = "http://128.112.34.129/cgi-bin/ptzctrl.cgi?ptzcmd&abs&24&20&0000&0200"
#    elif c <= 0.5: 
#        url = "http://128.112.34.129/cgi-bin/ptzctrl.cgi?ptzcmd&abs&24&20&FE80&0100"
#    elif c <= 0.75: 
#        url = "http://128.112.34.129/cgi-bin/ptzctrl.cgi?ptzcmd&abs&24&20&FC80&0000"
#    else:
#        url = "http://128.112.34.129/cgi-bin/ptzctrl.cgi?ptzcmd&abs&24&20&FD80&0100"
#    if url == last_request:
#        continue
    response = requests.get(url)


    if response.status_code == 200:
        print("Request successful!")
        print("Response content:")
        print(response.text)
    else:
        print("Request failed with status code:", response.status_code)
    time.sleep(0.5)

