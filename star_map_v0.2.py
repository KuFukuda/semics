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

from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
import pandas as pd
import datetime
import os

#目的：複数の画像の星の対応を正確に取りたい
#手法：各画像の星と星のカタログで対応を取る

def star_detect(image):
	#グレースケール画像にする
	img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	#明るさに閾値を設ける(ここでは適当に200)
	ret, new = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
	#画像は黒背景に白点になっているため、白点の輪郭を検出
	contours, hierarchy = cv2.findContours(new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#各輪郭について、重心を求めて配列に入れる
	stars = []
	for cnt in contours:
	    M = cv2.moments(cnt)
	    if M['m00'] != 0:
	        cx = int(M['m10']/M['m00'])
	        cy = int(M['m01']/M['m00'])
	        stars.append([cx,cy])
	    else:
	        stars.append(cnt[0][0])
	stars=np.array(stars,dtype='int32') 
	return stars

def draw_stars(img,stars,color_s):
	for s_point in stars:
		cv2.drawMarker(img, s_point, color_s, markerType=cv2.MARKER_STAR, markerSize=20, thickness=1, line_type=cv2.LINE_8)
	

class Application(tk.Frame):
	def __init__(self,master=None):
#		filename='/home/kunitofukuda/WorkSpace/Meteor/OpticalObserv/semic/test_crop/undistort_20231021035238.jpg'
#		self.img = cv2.imread(filename)
#		basename = os.path.splitext(os.path.basename(filename))[0]
#		t_base=basename[10:]
#		
#		stars_img=star_detect(self.img)
#		
#		simbad = Simbad()
#		simbad.add_votable_fields('flux(V)')
#		hoshi = simbad.query_criteria('Vmag<4',otype='star')
#		
#		LOCATION = EarthLocation(lon=139.3370196674786*u.deg, lat=36.41357867541122*u.deg, height=122*u.m)
#		utcoffset = 0*u.hour
#		tz = TimezoneInfo(9*u.hour) # 時間帯を決める。
#		
#		#toki = datetime.datetime(2023,10,21,3,52,38,tzinfo=tz)
#		print(t_base)
#		toki = datetime.datetime(int(t_base[:4]),int(t_base[4:6]),int(t_base[6:8]),int(t_base[8:10]),int(t_base[10:12]),int(t_base[12:]),tzinfo=tz)
#		OBSTIME = Time(toki)
#		OBSERVER = AltAz(location= LOCATION, obstime = OBSTIME)
#		
#		RA=hoshi['RA']
#		DEC=hoshi['DEC']
#		STAR_COORDINATES = SkyCoord(RA,DEC, unit=['hourangle','deg'])
#		STAR_ALTAZ       = STAR_COORDINATES.transform_to(OBSERVER)
#		seiza = STAR_ALTAZ.get_constellation()
#		z = (seiza[:,None]==np.unique(seiza)).argmax(1)
#		iro = np.stack([z/87,z%5/4,1-z%4/4],1)
#		s = (5-hoshi['FLUX_V'])*1
#		
#		AZ  = STAR_ALTAZ.az.deg
#		ALT = STAR_ALTAZ.alt.deg
#		stars_catalog=np.array([AZ,ALT])
#		stars_catalog=stars_catalog.T
#		
#		#AZ N 0 : E 90 : S 180 : W 270
#		center_x=191
#		width=86
#		center_y=43
#		height=50
#		
#		#top=center_y+height/2
#		#bottom=center_y-height/2
#		#left=center_x-width/2
#		#right=center_x+width/2
#		top=center_y+height/2
#		bottom=center_y-height/2
#		left=146
#		right=240
#		st = [s for s in stars_catalog if left<s[0] and s[0]<right]
#		stars_catalog = [s for s in st if bottom<s[1] and s[1]<top]
#		stars_catalog=np.array(stars_catalog,dtype='int32')
#		print(stars_catalog.shape)
#		
#		h,w = self.img.shape[:2]
#		print(h,w)
#		ws=w/(right-left)
#		hs=h/(top-bottom)
#		
#		stars_catalog[:,0]=(stars_catalog[:,0]-left)*ws
#		stars_catalog[:,1]=h-(stars_catalog[:,1]-bottom)*hs
#		
#		stars_catalog=np.array(stars_catalog,dtype='int32')
#		
#		color_catalog=(0,255,0)
#		draw_stars(self.img,stars_catalog,color_catalog)
#		color_img=(0,0,255)
#		draw_stars(self.img,stars_img,color_img)
#		cv2.imwrite('out.jpg', self.img)
#		
##		plt.figure(figsize=[8,4])
##		#plt.gca(xlim=[0,360],ylim=[-90,90],aspect=1,facecolor='k')
##		plt.gca(xlim=[left,right],ylim=[bottom,top],aspect=1,facecolor='k')
##		#plt.gca(facecolor='k',aspect=0.5,xlim=[0,90],ylim=[0,90])
##		plt.scatter(AZ,ALT,c='w',s=s)
##		plt.show()
#		
#		#plt.gca(facecolor='k',aspect=1,title='Orion')
#		#o = (seiza=='Orion')
#		#plt.scatter(AZ[o],ALT[o],c='w',s=s[o])
#		#plt.show()
		
		
		super().__init__(master)
		# ウィンドウのタイトルを設定
		self.master.title("MOSAIC")

		# ウィンドウのサイズを設定
		self.master.geometry("829x600")

		self.creat_widgets()

	def creat_widgets(self):
		# 画像を表示するキャンバスを作成
		self.canvas = tk.Canvas(self.master, width=630, height=400)
#		self.canvas.pack(side=tk.TOP)
		self.canvas.pack()

		# 画像をtkinter用に変換
		print("test1")
		filename='/home/kunitofukuda/WorkSpace/Meteor/OpticalObserv/semic/test_crop/undistort_20231021035238.jpg'
		img = cv2.imread(filename)
		img2 = cv2.resize(img, dsize=(630,400))
		img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
		img_pil = Image.fromarray(img_rgb) # RGBからPILフォーマットへ変換
		img_tk  = ImageTk.PhotoImage(img_pil) # ImageTkフォーマットへ変換
		print("test2")
#		photo = ImageTk.PhotoImage(img_tk)
		# キャンバスに新しい画像を表示
		print(img_tk)
#		self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
		self.canvas.create_image(315, 200, anchor=tk.NW,image=img_tk)
		print("test3")
		cv2.imwrite('test.jpg', img2)


def main():
	# tkinterのウィンドウを作成
	window = tk.Tk()
	app = Application(master=window)
	# tkinterのメインループを開始
	app.mainloop()

if __name__ == "__main__":
    main()
