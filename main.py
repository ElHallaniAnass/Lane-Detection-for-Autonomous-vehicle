import numpy as np
import cv2 as cv
from gpiozero import AngularServo
import time

servoMotor = AngularServo(17, min_pulse_width=0.0006, max_pulse_width=0.0023)

def prepareImg(fram):
    img = cv.cvtColor(fram, cv.COLOR_BGR2GRAY)
    img = np.array(img)
    return img

def reduirBruit(img):
    return cv.bilateralFilter(img, 5, 50, 50)

def meanImg(img):
    means, _ = cv.meanStdDev(img)
    return int(means[0][0])

def mask(img, v):
    mask = np.zeros_like(img)
    chaine_count = 1
    match_mask_color = (255,) * chaine_count
    cv.fillPoly(mask, v, match_mask_color)
    mask_img = cv.bitwise_and(img, mask)
    return mask_img

def maskImg(img):
    v = [(-50, img.shape[0]), (img.shape[1]/2-60, img.shape[0]/1.6-30), (img.shape[1]/2+20, img.shape[0]/1.6-30), (img.shape[1]+50, img.shape[0])]
    return mask(img, np.array([v], np.int32),)

def seuilImg(img, mean, masked_img):
    img = cv.split(img)[0]
    (_, newImg) = cv.threshold(masked_img, mean + 55, 255, cv.THRESH_BINARY)
    return newImg

def servo(sleep, angle):
    servoMotor.angle = angle
    time.sleep(sleep)
    # servoMotor.angle = 0


def contour(fram, newImg):
    contours, hierarchy = cv.findContours(image=newImg, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    image_draw = fram.copy()
    cv.drawContours(image=image_draw, contours=contours, contourIdx=-1, color=(0, 0, 0), thickness=5, lineType=cv.LINE_AA)
    return image_draw

def autoCorrection(newImg, axe=460, ligne=400):
    disGauche, disDroit, direction, valeurRoutation = 0, 0, 0, 0
    m = (np.where(newImg[ligne,:]==255))
    pose = [ (m[0][i],m[0][i-1]) for i in range(len(m[0])) if m[0][i]-m[0][i-1]>300]
    if len(pose)==1:
        disDroit = np.abs(pose[0][0]-axe)
        disGauche = np.abs(pose[0][1]-axe)
        if disGauche > disDroit:
            direction, valeurRoutation = -1, np.abs(disDroit-disGauche)
        elif disDroit > disGauche:
            direction, valeurRoutation = 1, np.abs(disDroit-disGauche)
        else:
            direction, valeurRoutation = 0, np.abs(disDroit-disGauche)
        print(disDroit)
        
    return direction, valeurRoutation

def traitement(frame):
    ligne = 400
    img = prepareImg(frame)
    img = reduirBruit(img)
    mean = meanImg(img)
    mask = maskImg(img)
    seuillage = seuilImg(img, mean, mask)
    direction, valeurRoutation = autoCorrection(seuillage, ligne=ligne)
    # print(valeurRoutation)
    angle = np.arccos(valeurRoutation/(img.shape[0]-ligne))*180/np.pi
    servo(0.01, angle if direction else -angle)
    contourImg = contour(frame, seuillage)
    return contourImg
    
def main():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret == True:
            cv.imshow('frame',traitement(frame))
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()


