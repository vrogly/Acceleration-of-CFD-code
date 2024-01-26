import numpy as np
import sys
from scipy.optimize import fsolve
resArray = [0.192, 0.609, 1, 1.925, 6.086, 13.608, 19.245,43.033]
xGuess = [6.3,1.019,1.01,1.005,1.00016, 1.00002, 1.000005,1.000001]
yGuess = [6.5,1.893,1.1,1.01,1.0001,1.00002,1.000005,1.000001]
# corresponding to 
# 1k, 10k, 27k (Original mesh), 100k, 1M, 5M, 10M, 50M
ii = -1
for resFactor in resArray:
    ii += 1
    print("----------------------------------")
    ni=int(300*resFactor)
    nj=int(90*resFactor)

    print(f"Cell count : {ni*nj}")

    xfac=np.power(1.01,1/resFactor)
    yfac=np.power(1.1,1/resFactor)

    dx=0.03/resFactor
    dy=7.83208E-004/resFactor

    yc=np.zeros(nj+1)
    yc[0]=0.
    yc[1]=7.83208E-004 # Do not want first cell outside of boundary closer than the other simulation since unclear which wall functions are used

    ustar=1/25
    viscos=3.57E-5
    yplus_initial=yc[1]*ustar/viscos
    # u_inf=1 => ustar=1.25 for this particular simulation

    gridname = "meshes/x" + str(ni) + "y" + str(nj)

    def fx(xfac):
        dx=0.03/resFactor
        xc=0
        for i in range(1,ni+1):
            xc=xc+dx
            if dx < 0.5:
                dx=dx*xfac
        return xc-55.64836332566466 
    xfac = fsolve(fx,xGuess[ii])
    print("x:  ",ii,xfac,fx(xfac))
    
    def fy(yfac):
        dy=7.83208E-004*yfac/resFactor
        yc=0
        for j in range(2,nj+1):
            yc=yc+dy
            if dy < 0.05:
                dy=dy*yfac
        return yc-2.8984869717985204 
    yfac = fsolve(fy,yGuess[ii])
    print("y:  ",ii,yfac,yGuess[ii],fy(yfac))
    
    dy=yfac*dy
    for j in range(2,nj+1): # orginally loops from 1
        yc[j]=yc[j-1]+dy
        #yplus=yc[j]*ustar/viscos
        if dy < 0.05:
           dy=yfac*dy
        #print('j=%d, y=%7.2E, yplus=%7.2E, dy=%7.2E'%(j,yc[j],yplus,dy))

    ymax_scale=yc[nj]

    # make it 2D
    y2d=np.repeat(yc[None,:], repeats=ni+1, axis=0)

    y2d=np.append(y2d,nj)
    np.savetxt(gridname + 'y2d.dat', y2d)

    xc=np.zeros(ni+1)
    for i in range(1,ni+1):
       xc[i]=xc[i-1]+dx
       if dx < 0.5:
          dx=dx*xfac
       #print('i=%d, x=%7.2E, dx=%7.2E'%(i,xc[i],dx))



    # make it 2D
    x2d=np.repeat(xc[:,None], repeats=nj+1, axis=1)
    x2d_org=x2d
    x2d=np.append(x2d,ni)
    np.savetxt(gridname + 'x2d.dat', x2d)


    # check it
    datay= np.loadtxt(gridname + "y2d.dat")
    y=datay[0:-1]
    nj=int(datay[-1])

    y2=np.zeros((ni+1,nj+1))
    y2=np.reshape(y,(ni+1,nj+1))

    datax= np.loadtxt(gridname + "x2d.dat")
    x=datax[0:-1]
    ni=int(datax[-1])

    x2=np.zeros((ni+1,nj+1))
    x2=np.reshape(x,(ni+1,nj+1))
    print(ni*nj,xc[1],xc[-1],yc[1],yc[-1])
