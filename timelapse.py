import cv2
import numpy as np
import glob
import os, shutil
import time
import datetime
from moviepy.editor import *

now = datetime.datetime.now()

if int(now.strftime("%M")) != 0:
    print('sover ' + str((60-int(now.strftime("%M")))) + ' minuter')
    time.sleep(60*(60-int(now.strftime("%M"))))

def procedure():
    print("sover 1h")
    time.sleep(60*60)

def skapaMappar(mapp):
    path = "kittelfjall/timelapse/" + mapp + "/" + (now + datetime.timedelta(days=-1)).strftime("%Y_%m_%d") + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    path = "kittelfjall/timelapse/" + mapp + "/" + (now + datetime.timedelta(days=-1)).strftime("%Y_%m_%d") + "/tmp"
    if not os.path.exists(path):
        os.makedirs(path)
        
def rensaMappar(mapp):
    tidAttRensa = (now + datetime.timedelta(days=-5)).strftime("%Y_%m_%d")
    path = "kittelfjall/timelapse/" + mapp + "/" + tidAttRensa + "/"
    if os.path.exists(path):
        shutil.rmtree(path)
        print("rensar mapp " + path)
  
def rensaTmp(mapp):
    path = "kittelfjall/timelapse/" + mapp + "/" + (now + datetime.timedelta(days=-1)).strftime("%Y_%m_%d") + "/tmp"
    if os.path.exists(path):
        shutil.rmtree(path)
        print("rensar " + mapp + " tmp")

def skapafilm(img_array, height, size, count, mapp):
    out = cv2.VideoWriter('kittelfjall/timelapse/' + mapp +'/' + (now + datetime.timedelta(days=-1)).strftime("%Y_%m_%d") + '/tmp/' + str(count) + '.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 24, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('skapat timelapse ' + mapp + ' part ' + str(count))
        
def skapaDelfilmer(mapp):
    img_array = []
    count = 1
    for filename in sorted(glob.glob('kittelfjall/' + mapp + '/' + (now + datetime.timedelta(days=-1)).strftime("%Y_%m_%d") + '/*.jpg')):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
        if len(img_array) > 1000:
            skapafilm(img_array, height, size, count, mapp)
            count+=1
            img_array = []
    if len(img_array) > 0:
         skapafilm(img_array, height, size, count, mapp)
         img_array = []
         print('skapat alla delfilmer ' + mapp)

def skapaTimelapse(mapp):
    filmer = []
    filenames = sorted(glob.glob("kittelfjall/timelapse/" + mapp + "/" + (now + datetime.timedelta(days=-1)).strftime("%Y_%m_%d") + "/tmp" + '/*.mp4'))
    for filename in filenames:
        video = VideoFileClip(filename)
        filmer.append(video)
    final_clip = concatenate_videoclips(filmer)
    final_clip.write_videofile('kittelfjall/timelapse/' + mapp + '/' + (now + datetime.timedelta(days=-1)).strftime("%Y_%m_%d") + '/kittel_' + mapp + '_' + (now + datetime.timedelta(days=-1)).strftime("%Y_%m_%d") + '.mp4', fps=24)

while True:
    now = datetime.datetime.now()
    skapaMappar('borkan')
    skapaMappar('hotell')
    skapaMappar('express')    
    
    if int(now.strftime("%H")) == 0:    
        skapaDelfilmer('borkan')
        skapaTimelapse('borkan')
        rensaTmp('borkan')
        rensaMappar('borkan')
        
        skapaDelfilmer('express')
        skapaTimelapse('express')
        rensaTmp('express')
        rensaMappar('express')
        
        skapaDelfilmer('hotell')
        skapaTimelapse('hotell')
        rensaTmp('hotell')
        rensaMappar('hotell')
    procedure()