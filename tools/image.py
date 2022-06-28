import numpy as np
import cv2
import easyocr
import re

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5) #kích cỡ càng to thì càng mờ
ADAPTIVE_THRESH_BLOCK_SIZE = 19 
ADAPTIVE_THRESH_WEIGHT = 9  
reader = easyocr.Reader(['en'])


def extractROIimg_fromPoints(img,points):
  xmin, ymin, xmax, ymax = points
  box = img[int(ymin)-2:int(ymax)+2, int(xmin)-2:int(xmax)+2]
  box = cv2.resize(box, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
  # blur = cv2.GaussianBlur(box, (3,3), 0)
  return box

def draw_bbox(img,points,name='license plates',color=(255,0,255),font_size=1):
  xmin, ymin, xmax, ymax = points
  start_point = (int(xmin), int(ymin))
  end_point = (int(xmax), int(ymax))
  image = cv2.rectangle(img.copy(), start_point, end_point, (36,255,12), 1)
  cv2.putText(image, name, start_point, cv2.FONT_HERSHEY_SIMPLEX, font_size,color, 2)
  return image


def maximizeContrast(imgGrayscale):
    #Làm cho độ tương phản lớn nhất 
    height, width = imgGrayscale.shape
    
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #tạo bộ lọc kernel
    
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations = 10) #nổi bật chi tiết sáng trong nền tối
    #cv2.imwrite("tophat.jpg",imgTopHat)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10) #Nổi bật chi tiết tối trong nền sáng
    #cv2.imwrite("blackhat.jpg",imgBlackHat)
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat) 
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    #cv2.imshow("imgGrayscalePlusTopHatMinusBlackHat",imgGrayscalePlusTopHatMinusBlackHat)
    #Kết quả cuối là ảnh đã tăng độ tương phản 
    return imgGrayscalePlusTopHatMinusBlackHat
  
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    
    #màu sắc, độ bão hòa, giá trị cường độ sáng
    #Không chọn màu RBG vì vd ảnh màu đỏ sẽ còn lẫn các màu khác nữa nên khó xđ ra "một màu" 
    return imgValue
  
def preprocess(imgOriginal):

    imgGrayscale = extractValue(imgOriginal)
    # imgGrayscale = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2GRAY) nên dùng hệ màu HSV
    # Trả về giá trị cường độ sáng ==> ảnh gray
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale) #để làm nổi bật biển số hơn, dễ tách khỏi nền
    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    
    #Làm mịn ảnh bằng bộ lọc Gauss 5x5, sigma = 0

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    #Tạo ảnh nhị phân
    return imgGrayscale, imgThresh
  
def detect_character(ROI_img, imgThresh):
  try:
      contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  except:
      ret_img, contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # sort contours left-to-right
  sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
  # create copy of gray image
  im2 = ROI_img.copy()
  # loop through contours and find individual letters and numbers in license plate
  for cnt in sorted_contours:
      x,y,w,h = cv2.boundingRect(cnt)
      height, width, _ = im2.shape
      # # if height of box is not tall enough relative to total height then skip
      # if height / float(h) > 2.: continue

      ratio = h / float(w)
      # # if height to width ratio is less than 1.5 skip
      if ratio < 1.5: continue

      # if width is not wide enough relative to total width then skip
      if width / float(w) > 10: continue

      area = h * w
      # if area is less than 100 pixels skip
      if area < 200: continue
      
      color = list(np.random.random(size=3) * 256) # (0,0,255)
      try:
        ROI = ROI_img.copy()[int(y)-5:int(y+h)+5, int(x)-5:int(x+w)+5]
        ocr_result = reader.readtext(ROI)
        c_result = re.sub('[^A-Z0-9]+', '', ocr_result[0][1].upper())
        if c_result!= '':
          im2 = cv2.rectangle(im2, (x,y), (x+w, y+h), color,1)
          im2 = cv2.putText(im2, str(c_result).upper(), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,color,3)
      except:
        continue
  return im2