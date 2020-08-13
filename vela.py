import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from math import*
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.fftpack import fftshift
import matplotlib.patches as mpatches
from PyAstronomy.pyasl import foldAt
#load data from the array stored in the directory. The data array was extracted from the raw file using dataextract.py
data= np.load('datafile.npy') 
#seperating the two coulmns into seperate array
north=data[:,0] 
south=data[:,1]
#Array initializations
array2d=[]
corarray=[]
array_new=[]
rolled=[]

x=0 	#to use as a index of the raw data

for j in range (1,1001):
	for i in range(1,61): 	#averaging the spectrum in sets of 60 to 1msec resolution as suggested in vela reference

		ffty=fft(north[(x-1)*512: x*512-1])		#fft of 512 points of north data
		ffts=fft(south[(x-1)*512: x*512-1])		#fft of 512 points of south data

		corarray=(ffty*np.conj(ffts)) #correlating the fft of north and south array
		realpart1= np.square(corarray.real)
		imagpart1=np.square(corarray.imag)
		amp1=np.sqrt(realpart1+ imagpart1) 
		array2d.append(amp1[256:511]) 	#storing the correlation result in an array 
		x += 1		 #incrementing raw data index after each loop
	array2dx=np.vstack(array2d)				#stacking each spectrum vertically
	array2d_mean=array2dx.mean(0)           #calculate columnwise mean
	array_new.append(array2d_mean)			#storing the data in an array after averaging
	array2d=[] 				#clearing the array2d after averaging each set of 60*512/33 points
img=np.vstack(array_new)    #stacking each 1msec of averaged spectrum vertically to from freq vs time array
med=np.median(img,axis=0)	#substracting the mean value of the array from the array
img_n=np.subtract(img,med)
imgf=np.transpose(img_n)	#transposing the array to get time on the x-axis

#plotting dispersed spectra
plt.subplot(2,2,1)
plt.imshow(imgf,cmap="hot",interpolation="gaussian")
plt.xlabel('Time in milli seconds')
plt.ylabel('freq channels')
plt.title('Dispersed Vela Spectra ')

#dedispersion
y=1 #array index
for i in range (1,255): 	#implementing delta t equation
	v=318.25+ (i*0.0645) 	
	dv=(i-255) * 0.0645
	q=(dv/(v*v*v))
	t=int((67.99*8.3e+6*q))
	roller=np.roll(imgf[y,:],t) 	#correcting for delta t difference
	rolled.append(roller)   		#storing the corrected (dedispersed) array
	y+=1

im2=np.vstack(rolled) #stacking dedispersed array
med2=np.median(im2,axis=0) #substracting the mean value of the array from the array
img3=np.subtract(im2,med2)

#plotting dedispersed spectra
plt.subplot(2,2,2)
plt.imshow(img3,cmap="hot",interpolation="gaussian")
plt.xlabel('Time in milli seconds')
plt.ylabel('freq channels')
plt.title('Dedispersed Vela Spectra')


#summing the intesities over freq after dedispersion
im4=[]
img5=[]
for i in range (1,1000): #for 1000 msec
	img4=img3[:,i].mean(0)
	img5.append(img4)

X=[]
p=97

for i in range (0,p):
 s=0                             # Folding it at p ms
 for j in range(len(img5)/p):    # Time bins and averaging
  s=s+img5[j*p+i]                           
 s=s/(len(img5)/p)                
 X.append(s)



#plotting the folded data
plt.subplot(2,2,3)
plt.plot(X)
plt.xlabel('Time in milli seconds')
plt.ylabel('Intensity')
plt.title('Pulses')





plt.show() 
