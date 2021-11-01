import cv2

for i in range(1,1300):
    b = f"./train/1xxx/{i}.jpg"
    # print(type(b))
    img = cv2.imread(b)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(gray,50,255, cv2.THRESH_BINARY)

    # 임계값 이상 = 0, 임계값 이하 = 원본값
    cv2.imwrite(f"./binary/1xxx/{i}.jpg", dst)