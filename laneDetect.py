#Code for Lane Detection

import numpy as np
import cv2
from math import  degrees
from shapely.geometry import LineString

def calcX1Y1X2Y2(rads):                   
    a = np.cos(rads)
    b = np.sin(rads)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 2000*(-b))
    y1 = int(y0 + 2000*(a))
    x2 = int(x0 - 2000*(-b))
    y2 = int(y0 - 2000*(a))    
    return x1,y1,x2,y2
    
'''def rightIntercept(x1,y1,x2,y2,frame,i,j):
    cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),3)
    line1 = LineString([(x1,y1),(x2,y2)])
    line2 = LineString([(0,i), (j,i)])
    p = (line1.intersection(line2)) 
    if not line1.intersection(line2):
        line2 = LineString([(j,0), (j, i)])
        p = (line1.intersection(line2)) 
    cv2.circle(frame,(int(p.x), int(p.y)),10,(0,255,0),3)
    print "Right:",p.x," ",p.y
    cv2.imshow('Right',frame)
    return  p.y

    
def leftIntercept(x1,y1,x2,y2,frame,i,j):
    cv2.line(frame,(x1,y1),(x2,y2), (255, 0,0),3)    
    line1 = LineString([(x1,y1),(x2,y2)])
    line2 = LineString([(0,i), (j,i)])
    p = (line1.intersection(line2))
    if not line1.intersection(line2):
        line2 = LineString([(0,0), (0, i)])
        p = (line1.intersection(line2))
    cv2.circle(frame,(int(p.x), int(p.y)),10,(255,0,0),3)
    print "Left:",p.x," ",p.y
    cv2.imshow('Left',frame)
    return p.y'''
    
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    
    #cv2.imshow('original', frame)

    edges = cv2.Canny(frame,150,200,apertureSize = 3)
    #cv2.imshow('canny', edges)
    #cv2.namedWindow('Lane Markers', cv2.WINDOW_NORMAL )

    i,j = edges.shape
    print i," ",j
    #cv2.line(img,(0,i),(j,i),(0,255,0),3)
    #for calculating ROI
    for x in range(i/4):
        for y in range(j):
            edges[x][y] = 0
            
    '''for x in range(i*85/100, i):
        for y in range(j):
            edges[x][y] = 0'''        
            
    for x in range(i*65/100, i):
        for y in range(j*50/100, j*70/100):
            edges[x][y] = 0

    k = j/3
    start = i/4

    while(k>0):
        for x in range(start, i):
            for y in range(0, k):
                edges[x][y] = 0
            k -= 2 
      	
    k = j*2/3
    while(k<=j):
        for x in range(start, i):	    
            for y in range(k, j):
                edges[x][y] = 0
                k += 2
    cv2.imshow('roi', edges)

    lines = cv2.HoughLines (edges,.45,np.pi/180,75)

    rads = []
    if lines != None:
        for index in range(len(lines)):
            for rho,theta in lines[index]:
        
                radian = theta
                theta = degrees(theta)
                if (theta >= 30 and theta <= 70):    
                    rads.append(radian)
                if (theta >=100 and theta <=140 ):                                 
                    rads.append(radian)
    
    
    if(len(rads) > 0 ):
        minr = min(rads)
        for index in range(len(lines)):
            #print "s"
            for rho,theta in lines[index]:
                
                if max(rads) == theta:
                    maxr = max(rads)
                    x1,y1,x2,y2 = calcX1Y1X2Y2(max(rads))
                    cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),3)
                    line1 = LineString([(x1,y1),(x2,y2)])
                    line2 = LineString([(0,i), (j,i)])
                    p = (line1.intersection(line2)) 
                    if not line1.intersection(line2):
                        line2 = LineString([(j,0), (j, i)])
                        p = (line1.intersection(line2)) 
                    cv2.circle(frame,(int(p.x), int(p.y)),10,(0,255,0),3)
                    #print "Right:",p.y
                    #cv2.imshow('Right',frame)
                    Y2 = p.y
                    print Y2
                    
                if min(rads) == theta:                                    
                    minr = min(rads)
                    x1,y1,x2,y2 = calcX1Y1X2Y2(min(rads))
                    cv2.line(frame,(x1,y1),(x2,y2), (255, 0,0),3)    
                    line1 = LineString([(x1,y1),(x2,y2)])
                    line2 = LineString([(0,i), (j,i)])
                    p = (line1.intersection(line2))
                    if not line1.intersection(line2):
                        line2 = LineString([(0,0), (0, i)])
                        p = (line1.intersection(line2))
                        
                    cv2.circle(frame,(int(p.x), int(p.y)),10,(255,0,0),3)
                    #print "Left:",p.y
                    #cv2.imshow('Left',frame)
                    Y1 = p.y
                    print Y1
    
                d = Y1-Y2         
                font = cv2.FONT_HERSHEY_SIMPLEX
                a=j/10
                b=9*i/10
                if abs(d) >= i/10:
                    if d>0:
                        cv2.putText(frame,"Shift Right.", (a,b),font,1,(255,255,0),2,cv2.LINE_AA)
                    else:
                        cv2.putText(frame,"Shift Left.", (a,b),font,1,(255,255,0),2,cv2.LINE_AA)
                else:
                    cv2.putText(frame,"You are on the right track.", (a,b),font,1,(255,255,0),2,cv2.LINE_AA)
            
    print lines
    #print rads
    cv2.imshow('Lane Markers',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
