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
import math

#目的：複数の画像の星の対応を正確に取りたい
#手法：各画像の星と星のカタログで対応を取る

#v0.4 調整がうまくいった際に自動で星を対応させる機能を追加

def star_detect(image):
	#グレースケール画像にする
	img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	#明るさに閾値を設ける(ここでは適当に200)
	threshold=150
	ret, new = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
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

#def draw_stars(img,stars,color_s):
def draw_stars(img,stars,color_s,maker):
	for s_point in stars:
#		cv2.drawMarker(img, s_point, color_s, markerType=cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_8)
		cv2.drawMarker(img, s_point, color_s, markerType=maker, markerSize=10, thickness=1, line_type=cv2.LINE_8)
	
class MainApplication(tk.Frame):
	def __init__(self, master):
		super().__init__(master)
		self.master = master
		self.master.title("MOSAIC")
		self.master.geometry('1000x700')

		# 画像を読み込み
		self.read_img()
#画像から星を抽出
		self.stars_img=star_detect(self.img)
		
		self.map_catalog()
		self.draw_stars_d()

#		img_star_catalog = np.full(self.img.shape, 0, dtype=np.uint8)
#		color_img=(255,255,255)
##		draw_stars(img_star_catalog,self.stars_catalog,color_img)
#		for s_point in self.stars_catalog:
##			cv2.drawMarker(img_star_catalog, s_point, color_img, markerType=cv2.MARKER_DIAMOND, markerSize=5, thickness=1, line_type=cv2.LINE_8)
#			cv2.circle(img_star_catalog, s_point, 2, color_img, thickness=1, lineType=cv2.LINE_AA, shift=0)
#		cv2.imwrite('20231021032633.jpg', img_star_catalog)
		
		self.kp_catalog=[]
		self.kp_catalog_original=[]
		self.kp_img=[]
		self.flag=0
		self.mtx_old=np.array([[1,0,0],[0,1,0]])
		self.line=[]

		self.create_widget()

	def create_widget(self):
		h,w=self.img.shape[:2]
		self.canvas1 = tk.Canvas(self.master, width=w, height=h)
		self.canvas1.pack()
		self.canvas1.place(x=0, y=0)

		self.label1 = tk.Label(self.master, bg="white", width=10, height=3)
		self.label1.place(x=100, y=600)
		self.label2 = tk.Label(self.master, bg="green", width=10, height=3)
		self.label2.place(x=400, y=600)
		self.label3 = tk.Label(self.master, bg="red", width=10, height=3)
		self.label3.place(x=550, y=600)

		self.button = tk.Button(self.master, text="Adjust",command=self.remake_img)
		self.button.place(x=650, y=600)

#		self.button = tk.Button(self.master, text="Auto stars",command=self.remake_img)
		self.button = tk.Button(self.master, text="Auto stars",command=self.auto)
		self.button.place(x=750, y=600)

		self.disp_img()

		# canvas1にマウスが乗った場合、離れた場合のイベントをセット。
		self.canvas1.bind('<Motion>', self.mouse_motion)
		self.canvas1.bind("<ButtonPress-1>", self.point_get)
#		self.button.bind("<ButtonPress-1>", self.point_get)
#		self.canvas1.bind('<KeyPress>',self.key_evnet)

#		font = tk.font.Font(family='Arial', size=16, weight='bold')
#		image_title = tk.Label(text='=>', bg = "white", font=font)
		image_title = tk.Label(text='=>', bg = "white")
		image_title.place(x=500, y=610, anchor=tk.NW)

	def Reline(self):
#		p1=[1,1]
#		p2=[2,2]
		self.kp_img=np.array(self.kp_img)
		for i in range(self.kp_img.shape[0]):
			p1=self.kp_catalog_original[i]
			p2=self.kp_img[i]
			
			a=(p1[1]-p2[1])/(p1[0]-p2[0])
			a=-1/a
	
			mx=(p1[0]+p2[0])/2
			my=(p1[1]+p2[1])/2
	
		#切片
			b=my-a*mx
	
			self.line.append([a,b])
#		print(self.line)

	def calc_center(self):
		a0,b0=self.line[0]
		a1,b1=self.line[1]

		cx=-(b0-b1)/(a0-a1)
		cy=(a0*b1-a1*b0)/(a0-a1)
		self.center=np.array([cx,cy])
		print(self.center)

		add_vect=0
		for line in self.line:
			a,b=line
			A=np.array([0,b])
			B=np.array([1,a+b])
			ab=B-A
			ap=self.center-A

			ai_norm=np.dot(ap,ab)/np.linalg.norm(ab)
			neighbor_point=a+(ab)/np.linalg.norm(ab)*ai_norm
#			add_vect+=self.center-neighbor_point
			print(neighbor_point)
			print(neighbor_point-self.center)
			add_vect+=neighbor_point-self.center

		print(add_vect)
		self.center+=add_vect
		self.center=np.array(self.center,dtype="int32")
		print("center")
		print(self.center)

	#緑の星に最近接の赤の星を自動的に対応付ける
	#距離で制限する
	def auto(self):
		R=10
		for catalog_indx in range(self.stars_catalog.shape[0]):
			dist_list=[]	#距離のリスト
			for star_img in self.stars_img:
				dist=np.linalg.norm(self.stars_catalog[catalog_indx]-star_img)
				dist_list.append(dist)
			min_dist=min(dist_list)
			if min_dist<R:
				min_dist_indx=np.argmin(dist_list)	#最短のstarsのインデックス
				self.kp_catalog.append(self.stars_catalog[catalog_indx])
				self.kp_catalog_original.append(self.stars_catalog_original[catalog_indx]) #add
				self.kp_img.append(self.stars_img[min_dist_indx])
				cv2.line(self.img, self.kp_catalog[-1], self.kp_img[-1], (255,255,255), thickness=1)
		self.disp_img()
		

	def map_catalog(self):
		simbad = Simbad()
		simbad.add_votable_fields('flux(V)')
		hoshi = simbad.query_criteria('Vmag<4',otype='star')
		
		LOCATION = EarthLocation(lon=139.3370196674786*u.deg, lat=36.41357867541122*u.deg, height=122*u.m)
		utcoffset = 0*u.hour
		tz = TimezoneInfo(9*u.hour) # 時間帯を決める。
		basename = os.path.splitext(os.path.basename(self.filename))[0]
#		basename = "20231021023633"
#		t_base=basename[10:]
#		t_base=basename[5:]	#trim_
		t_base=basename
		print(t_base)
		toki = datetime.datetime(int(t_base[:4]),int(t_base[4:6]),int(t_base[6:8]),int(t_base[8:10]),int(t_base[10:12]),int(t_base[12:]),tzinfo=tz)
		OBSTIME = Time(toki)
		OBSERVER = AltAz(location= LOCATION, obstime = OBSTIME)
		
		RA=hoshi['RA']
		DEC=hoshi['DEC']
		STAR_COORDINATES = SkyCoord(RA,DEC, unit=['hourangle','deg'])
		STAR_ALTAZ       = STAR_COORDINATES.transform_to(OBSERVER)
		self.seiza = STAR_ALTAZ.get_constellation()
		z = (self.seiza[:,None]==np.unique(self.seiza)).argmax(1)
		iro = np.stack([z/87,z%5/4,1-z%4/4],1)
		self.s = (5-hoshi['FLUX_V'])*1
		
		self.AZ  = STAR_ALTAZ.az.deg
		self.ALT = STAR_ALTAZ.alt.deg
		self.stars_catalog=np.array([self.AZ,self.ALT])
		self.stars_catalog=self.stars_catalog.T
		
		#AZ N 0 : E 90 : S 180 : W 270
#		center_x=191
#		width=86
#		center_y=43
#		height=50
		
		#top=center_y+height/2
		#bottom=center_y-height/2
		#left=center_x-width/2
		#right=center_x+width/2
		self.top=68
		self.bottom=18
		self.left=146
		self.right=240

		h,w = self.img.shape[:2]
#		print(h,w)
		ws=w/(self.right-self.left)
		hs=h/(self.top-self.bottom)

		self.stars_catalog[:,0]=(self.stars_catalog[:,0]-self.left)*ws
		self.stars_catalog[:,1]=h-(self.stars_catalog[:,1]-self.bottom)*hs
		
#		self.stars_catalog=np.array(self.stars_catalog,dtype='int32')
#		print(self.stars_catalog.shape)

#		st = [s for s in self.stars_catalog if self.left-10<s[0] and s[0]<self.right+10]
#		self.stars_catalog = [s for s in st if self.bottom-10<s[1] and s[1]<self.top+10]
		st = [s for s in self.stars_catalog if -100<s[0] and s[0]<w+100]
		self.stars_catalog = [s for s in st if -100<s[1] and s[1]<h+100]
		
		self.stars_catalog=np.array(self.stars_catalog,dtype='int32')
		self.stars_catalog_original=self.stars_catalog
#		print(self.stars_catalog.shape)
		
	def draw_stars_d(self):
		color_catalog=(0,255,0)
#		draw_stars(self.img,self.stars_catalog,color_catalog)
		draw_stars(self.img,self.stars_catalog,color_catalog,cv2.MARKER_STAR)
		color_img=(0,0,255)
#		draw_stars(self.img,self.stars_img,color_img)
		draw_stars(self.img,self.stars_img,color_img,cv2.MARKER_SQUARE)
		cv2.imwrite('out.jpg', self.img)

#		plt.figure(figsize=[8,4])
#		#plt.gca(xlim=[0,360],ylim=[-90,90],aspect=1,facecolor='k')
#		plt.gca(xlim=[self.left,self.right],ylim=[self.bottom,self.top],aspect=1,facecolor='k')
#		#plt.gca(facecolor='k',aspect=0.5,xlim=[0,90],ylim=[0,90])
#		plt.scatter(self.AZ,self.ALT,c='w',s=self.s)
##		plt.savefig("save_1.jpg")
#		plt.show()
#		
#		plt.gca(facecolor='k',aspect=1,title='Orion')
#		o = (self.seiza=='Orion')
#		plt.scatter(self.AZ[o],self.ALT[o],c='w',s=self.s[o])
#		plt.show()

	def disp_img(self):
		self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
		self.img_pil = Image.fromarray(self.img_rgb) # RGBからPILフォーマットへ変換
		self.img_tk  = ImageTk.PhotoImage(self.img_pil) # ImageTkフォーマットへ変換
		#photo = ImageTk.PhotoImage(img_tk)
		# キャンバスに新しい画像を表示
		self.canvas1.create_image(0, 0, anchor=tk.NW, image=self.img_tk,tag="img")

	def read_img(self):
		self.filename="20231021032633.jpg"
#		self.filename="/home/kunitofukuda/workspace/Meteor/Optical2Wave/20231021/trim_img/20231021023633.jpg"
#		self.filename="20231021035238.jpg"
		self.img = cv2.imread(self.filename)
		h,w = self.img.shape[:2]
	#画像を半分のサイズに変更
#		self.img = cv2.resize(self.img, dsize=(int(w/2),int(h/2)))

	def remake_img(self):
		self.canvas1.delete("img")
		self.read_img()	#画像を再構成
#画像から星を抽出
		self.stars_img=star_detect(self.img)
#		self.map_catalog()

		self.Reline()
		self.calc_center()


#カタログの星を移動させる
		self.stars_adjust()
		self.draw_stars_d()
		self.disp_img()

		self.kp_catalog=[]
		self.kp_catalog_original=[]
		self.kp_img=[]

	def stars_adjust(self):
		self.kp_catalog=np.array(self.kp_catalog)
		self.kp_catalog_original=np.array(self.kp_catalog_original)
		self.kp_img=np.array(self.kp_img)
#		print(self.kp_catalog,self.kp_img)
		if self.kp_catalog.shape[0]>5:
			self.mtx,inliers=cv2.estimateAffinePartial2D(self.kp_catalog_original,self.kp_img)
#			mtx,inliers=cv2.estimateAffinePartial2D(self.kp_img,self.kp_catalog)
#			mtx,inliers=cv2.estimateAffine2D(self.kp_catalog_original,self.kp_img)
		else:
			self.mtx=np.array([[1,0,0],[0,1,0]])
			inliers=np.array([[0]])
#		print(mtx.shape)
#		print(self.mtx_old.shape)
		insert_m=[0,0,1]
		self.mtx=np.insert(self.mtx,2,insert_m,axis=0)
		self.mtx_old=np.insert(self.mtx_old,2,insert_m,axis=0)
#		print(self.mtx)
#		print(self.mtx_old)
		self.mtx=np.dot(self.mtx,self.mtx_old)
#		mtx=np.delete(mtx,2,axis=0)
#		print(inliers)
#		print(sum(inliers))
#		print(self.mtx)
#回転角を示す
		degree = np.rad2deg(-np.arctan2(self.mtx[0, 1], self.mtx[0, 0]))
#		print(degree)

#		self.calc_stars_point()
		self.rotate_point()
#		print(self.mtx_old)
		self.mxt_old=self.mtx
		self.mtx_old=self.mtx_old[:2,:]
#		print(self.mtx_old)

	def calc_stars_point(self):
		stars_catalog_calc=np.insert(self.stars_catalog_original, 2, 1, axis=1)

		stars_catalog_re=[]
		for s_point in stars_catalog_calc:
			s_point=np.dot(self.mtx,s_point)
			s_point=np.array(s_point,dtype='int32')
			stars_catalog_re.append(s_point)
		stars_catalog_re=np.array(stars_catalog_re)
		stars_catalog_re=stars_catalog_re[:,:2]
		self.stars_catalog=stars_catalog_re

	def rotate_point(self):
		angle=math.radians(1)
		sin_angle = math.sin(angle)
		cos_angle = math.cos(angle)
#		center=[348,1483]
		center=self.center
		
		# 回転した座標を格納するリスト
		rotated_points = []
		
#		for i in range(0, len(point), 2):
#		for point in self.stars_catalog_original:
		for point in self.stars_catalog:
			x=point[0]-center[0]
			y=point[1]-center[1]
			
			# 回転後の座標を計算
			rotated_x = x * cos_angle - y * sin_angle + center[0]
			rotated_y = y * cos_angle + x * sin_angle + center[1]
			
			rotated_points.append([rotated_x, rotated_y])

		self.stars_catalog_re=np.array(rotated_points,dtype="int32")
#		self.stars_catalog_re=self.stars_catalog_re.reshape(self.stars_catalog_original.shape)
		self.stars_catalog_re=self.stars_catalog_re.reshape(self.stars_catalog.shape)
		self.stars_catalog=self.stars_catalog_re
		print(self.stars_catalog[0])	
		print(self.stars_catalog[1])	

	def point_get(self,event):
		sd=self.nearst(event)
#		print(sd,self.flag)
		if sd!=0:
			if self.flag==0:
				self.kp_catalog.append(self.stars_catalog[sd])
				self.kp_catalog_original.append(self.stars_catalog_original[sd]) #add
#				self.label2 = tk.Label(self.master, bg="gray", width=10, height=3)
				self.label2["text"]=str(self.stars_catalog[sd])
#				print(self.stars_catalog[sd])
				self.flag=1
			else:
				self.kp_img.append(self.stars_img[sd])
				self.label3["text"]=str(self.stars_img[sd])
#				print(self.stars_img[sd])
				cv2.line(self.img, self.kp_catalog[-1], self.kp_img[-1], (255,255,255), thickness=1)
				self.disp_img()
				self.flag=0
#		print(self.stars)


#flagはカタログ0か画像1かの違い
#	def nearst(self,event,flag):
	def nearst(self,event):
		R=50
		mause_point=np.array([event.x,event.y])
		if self.flag==0:
			stars=self.stars_catalog
		else:
			stars=self.stars_img
		dist_list=[]	#距離のリスト
		for star in stars:
#			print(star)
			dist=np.linalg.norm(star-mause_point)
			dist_list.append(dist)
#		print(self.stars_img)
		min_dist=min(dist_list)
		if min_dist<R:
			sd=np.argmin(dist_list)	#最短のstarsのインデックス
		else:
			sd=0
#		print(mause_point,self.stars_catalog[sd],min_dist)
		return sd

	def mouse_motion(self, event):
		# マウス最近傍の星の座標を得る
		x = event.x
		y = event.y
		self.label1["text"] = str([x,y]) 

def main():
	root = tk.Tk()
	app = MainApplication(master = root)
# tkinterのメインループを開始
	app.mainloop()

if __name__ == '__main__':
	main()
