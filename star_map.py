from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time,TimezoneInfo
from astropy.coordinates import SkyCoord
#from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astroquery.simbad import Simbad
import astropy.units as u
import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime

#目的：複数の画像の星の対応を正確に取りたい
#手法：各画像の星と星のカタログで対応を取る

simbad = Simbad()
simbad.add_votable_fields('flux(V)')
hoshi = simbad.query_criteria('Vmag<4',otype='star')

LOCATION = EarthLocation(lon=139.3370196674786*u.deg, lat=36.41357867541122*u.deg, height=122*u.m)
utcoffset = 0*u.hour
tz = TimezoneInfo(9*u.hour) # 時間帯を決める。
toki = datetime.datetime(2023,10,21,3,52,38,tzinfo=tz)
OBSTIME = Time(toki)
OBSERVER = AltAz(location= LOCATION, obstime = OBSTIME)

RA=hoshi['RA']
DEC=hoshi['DEC']
STAR_COORDINATES = SkyCoord(RA,DEC, unit=['hourangle','deg'])
STAR_ALTAZ       = STAR_COORDINATES.transform_to(OBSERVER)
seiza = STAR_ALTAZ.get_constellation()
z = (seiza[:,None]==np.unique(seiza)).argmax(1)
iro = np.stack([z/87,z%5/4,1-z%4/4],1)
s = (5-hoshi['FLUX_V'])*1

AZ  = STAR_ALTAZ.az.deg
ALT = STAR_ALTAZ.alt.deg
stars=np.array([AZ,ALT])
stars=stars.T

#AZ N 0 : E 90 : S 180 : W 270
center_x=191
width=86
center_y=43
height=50

#top=center_y+height/2
#bottom=center_y-height/2
#left=center_x-width/2
#right=center_x+width/2
top=center_y+height/2
bottom=center_y-height/2
left=146
right=240
st = [s for s in stars if left<s[0] and s[0]<right]
stars = [s for s in st if bottom<s[1] and s[1]<top]
stars=np.array(stars,dtype='int32')
print(stars.shape)

filename='/home/kunitofukuda/WorkSpace/Meteor/OpticalObserv/semic/test_crop/undistort_20231021035238.jpg'
img = cv2.imread(filename)

h,w = img.shape[:2]
print(h,w)
ws=w/(right-left)
hs=h/(top-bottom)

stars[:,0]=(stars[:,0]-left)*ws
stars[:,1]=h-(stars[:,1]-bottom)*hs

stars=np.array(stars,dtype='int32')
for s_point in stars:
#	print(s_point)
	cv2.drawMarker(img, s_point, (0,0,255), markerType=cv2.MARKER_STAR, markerSize=20, thickness=1, line_type=cv2.LINE_8)
center_point=np.array([w/2,h/2],dtype='int32')
cv2.drawMarker(img, center_point, (0,255,0), markerType=cv2.MARKER_STAR, markerSize=20, thickness=1, line_type=cv2.LINE_8)
cv2.imwrite('out.jpg', img)

plt.figure(figsize=[8,4])
#plt.gca(xlim=[0,360],ylim=[-90,90],aspect=1,facecolor='k')
plt.gca(xlim=[left,right],ylim=[bottom,top],aspect=1,facecolor='k')
#plt.gca(facecolor='k',aspect=0.5,xlim=[0,90],ylim=[0,90])
plt.scatter(AZ,ALT,c='w',s=s)
plt.show()

#plt.gca(facecolor='k',aspect=1,title='Orion')
#o = (seiza=='Orion')
#plt.scatter(AZ[o],ALT[o],c='w',s=s[o])
#plt.show()
