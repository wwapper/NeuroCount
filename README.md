# NeuroCount
<div>
<h3>
NeuroCount - программный код  для автоматического подсчета тел нейронов на фотографиях гистологических срезов головного мозга.
</h3>
</div>

![Иллюстрация к проекту](https://github.com/wwapper/NeuroCount/blob/master/program/images/photo1674043539.jpeg)

<div>
<h4>
Подсчет нейронов необходим при исследовании нейродегенеративных заболеваний, для сравнения количества нейронов в срезах здорового и больного мозга, чтобы исследовать влияние различных факторов на развитие патологических процессов.

Зачастую, количество нейронов, окрашенных по методу Ниссля или с помощью иммуногистохимических методов анализируются визуально и подсчитываются вручную с использованием программ PhotoM и ImageJ. Существуют программы автоматического подсчета клеток, но они дают недостоверные результаты при подсчете тел нейронов.
</h4>
 
<h3>
Описание датасета:
  </h3>
  
  <h4>
Набор данных, предоставленный лабораторией НИИ РАН, содержит 205 изображений размерами примерно 800 * 600 пикселей, которые представлены в двух видах (нейроны иммунопозитивные к тирозингидроксилазе, где использовался биотин-стрептавидиновый метод иммуногистохимического окрашивания, и нейроны окрашенные по методу Ниссля). 
  
Изображения охватывают достаточное разнообразие возможного количества нейронов на одной картинке (от 50 до 200), однако включают только два вида окрашиваний.
Расширение датасета возможно при дополнении его изображениями, содержащими:
- другие типа окрашиваний
- срезы других отделов мозга

 </h4>
</div>
  
<h2> Ниже приведен код с подробными комментариями </h2>
<div>
  <h3>Импортируем необходимые библиотеки:
  <h4> 
    
    import numpy as np
    import cv2
    import imutils
    
  </h4>
   <h5>
   Наш проект реализован с помощью библиотеки OpenCV, включающей в себя большое число алгоритмов компьютерного зрения
   </h5> 
  </h3>
</div>

<div>
  <h3>Определим функцию, отвечающую за увеличение яркости изображения:</h3>
  <h4>

    def increase_brightness(img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
              
   </h4>
    <h5>В качестве аргументов функции выступают картинка, яркость которой необходимо поменять, и значение параметра, отвечающего за степень увеличения яркости изображени (подбирается вручную, по умолчанию 30).
    </h5>
</div>
  

<div>
  <h3>Загружаем интересующее нас изображение:</h3>
  <h4>

    image = cv2.imread("example.jpg")
    
  </h4>
  
![Иллюстрация к проекту](https://github.com/wwapper/NeuroCount/blob/master/program/images/photo1674033491.jpeg)
  
    <h3>Обводим изображение белыми рамками:</h3>
  <h4>

    image = cv2.copyMakeBorder(src=image, top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_CONSTANT,value=(255, 255, 255))
    
  </h4>
  <h5>Этот шаг необходим для того, чтобы алгоритм видел те нейроны, которые находятся на краю изображения. Без рамки алгоритм не обводит граничные клетки.
  </h5>
      <h3>Увеличиваем яркость изображения, используя определенную ранее функцию:</h3>
  <h4>

    image = increase_brightness(image, value=10)
    
  </h4>
  <h5>Увеличения яркости является необходимым шагом при удалении с изображения шумового окрашивания, которые мешают подсчету количества нейронов на фотографии среза.
  </h5>
        <h3>Увеличиваем резкость изображения и переводим его в черно-белую цветовую гамму:</h3>
  <h4>

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])  
    image = cv2.filter2D(image, -1, kernel)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.jpg", gray_image)
    </h4>
    
![Иллюстрация к проекту](https://github.com/wwapper/NeuroCount/blob/master/program/images/gray.jpeg)
    
  <h5>Матрица kernel является линейной сверткой, которая берет значение яркости окружающих данную точку пиеселей, умножает на -1 и складываем с яркостью центрального пикселя, умноженной на 9. Результат - увеличение резкости изображения. Этот шаг нужен для того, чтобы получить более четкие границы нейронов на черно-белом изображении
  </h5>
          <h3>Применяем размытие к черно-белому изображению:</h3>
  <h4>

    gray_image_blur = cv2.medianBlur(gray_image, 5)
    cv2.imwrite("gray_image_blur.jpg", gray_image_blur)
      </h4>
    
 ![Иллюстрация к проекту](https://github.com/wwapper/NeuroCount/blob/master/program/images/blur.jpeg)
    
  <h5>Размытие необходимо для удаления аксонов с изображения.
  </h5>
            <h3>Перевод изображения в бинарное черно-белое:</h3>
  <h4>

    ret, threshold_image = cv2.threshold(gray_image_blur, 120, 255, cv2.THRESH_BINARY)
    cv2.imwrite("threshold_image.jpg", threshold_image)
    
 ![Иллюстрация к проекту](https://github.com/wwapper/NeuroCount/blob/master/program/images/threshold.jpeg)
    
  </h4>
  <h5>Результатом работы этой части кода является бинарное черно-белое изображение. Оно получается из исходного заменой пикселей: если его яркость ниже 120, то он заменяется на 0 (черный), если выше, то на 255 (белый). Пороговое значение подбирается вручную.
  </h5>
              <h3>Повторяем шаги размытия изображения, и ещё раз переводим его в бинарное черно-белое:</h3>
  <h4>

    threshold_image_blur = cv2.GaussianBlur(threshold_image, (3, 3), 5)
    cv2.imwrite("threshold_image_blur.jpg", threshold_image_blur)
    
![Иллюстрация к проекту](https://github.com/wwapper/NeuroCount/blob/master/program/images/threshold%20blur.jpeg)

    ret, threshold_image_new = cv2.threshold(threshold_image_blur, 120, 255, cv2.THRESH_BINARY)
    cv2.imwrite("threshold_image_new.jpg", threshold_image_new)
    
![Иллюстрация к проекту](https://github.com/wwapper/NeuroCount/blob/master/program/images/threshold%20new.jpeg)
    
  </h4>
  <h5>Опыт использования данного кода на нескольких картинках показал необходимость этого шага, которая состоит в очередном шаге удаления аксонов на изображении. Параметры размытия и перевода в бинарное чб подбирались вручную.
  </h5>
                <h3>Контурирование полученного изображения:</h3>
  <h4>
    
    edged = cv2.Canny(threshold_image_new, 0, 300, apertureSize = 7, L2gradient=True)
    cv2.imwrite("edged.jpg", edged)
      </h4>
    
![Иллюстрация к проекту](https://github.com/wwapper/NeuroCount/blob/master/program/images/edged.jpeg)
    

  <h5>На данном шаге были определены контуры объектов на изображении. 
  </h5>
                  <h3>Удаление слишком маленьких/больших контуров на изображении:</h3>
  <h4>
    
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(edged)
    sizes = stats[:, -1]
    sizes = sizes[1:]
    nb_blobs -= 1
    min_size = 40
    max_size = 10000
    im_result = np.zeros((edged.shape))
    for blob in range(nb_blobs):
        if max_size >= sizes[blob] >= min_size:
            im_result[im_with_separated_blobs == blob + 1] = 255
    cv2.imwrite("final.jpg", im_result)
    </h4>
    
![Иллюстрация к проекту](https://github.com/wwapper/NeuroCount/blob/master/program/images/final.jpeg)
    
  <h5>Данный этап необходим для удаления различных артефактов, исходя из их размера. Параметры max_size и min_size, отвечающие соответственно за верхнюю и нижнюю границы условия на размер, были выбраны вручную.
  </h5>  
                    <h3>Замыкание контуров объектов на изображении:</h3>
  <h4>
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_final = cv2.morphologyEx(im_result, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("closed_final.jpg", closed_final)
    closed_final = closed_final.astype(np.uint8)
  </h4>
    
 ![Иллюстрация к проекту](https://github.com/wwapper/NeuroCount/blob/master/program/images/closed%20final.jpeg)
    
  <h5>Для корректного подсчета нейронов необходимо замкнуть близко расположенные точки на контуре клетки.
  </h5>
                      <h3>Отображение контуров поверх исходного изображения:</h3>
  <h4>
    
    image1 = image.copy()
    im_result = im_result.astype(np.uint8)
    contours, hierarchy = cv2.findContours(im_result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image1, contours, -1, (0,0,0), 2, cv2.LINE_AA)
    cv2.imwrite('contours.jpg', image1)
  </h4>
    
 ![Иллюстрация к проекту](https://github.com/wwapper/NeuroCount/blob/master/program/images/contours.jpeg)
    
  <h5>На этом шаге происходит сопоставление полученных контуров нейронов с исходным изоюражением.
  </h5>
                        <h3>Удаление контурных артефактов и сопоставление "очищенной" карты контуров с исходным изображением:</h3>
  <h4>
    
    closed_final = closed_final.astype(np.uint8)
    contours, hierarchy = cv2.findContours(closed_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area/perimeter > 2.7:
            new_contours.append(cnt)

    cv2.drawContours(image, new_contours, -1, (0,0,0), 3, cv2.LINE_AA)
    cv2.imwrite('contours_new.jpg', image)
   </h4>
    
 ![Иллюстрация к проекту](https://github.com/wwapper/NeuroCount/blob/master/program/images/contours%20new.jpeg)
    
  <h5>Этот шаг необходим, чтобы удалить разного рода артефакты, которые не были удалены по условию на размер (на картинке с контурами этими артефактами являются линии). Было предложено условие на отношение площади к периметру (как известно, для линий это отношение близко к 1). Соответственно, если это отношение больше 2.7 (выбрано вручную), то контур удаляется.
  </h5>
                          <h3>Подсчет количества нейронов на картинке:</h3>
  <h4>
    
    coord = []
    for c in new_contours:
      M = cv2.moments(c)
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
      cv2.drawContours(image, [c], -1, (0, 0, 0), 2)
      cv2.circle(image, (cX, cY), 7, (0, 0, 0), -1)
      cv2.putText(image, "center", (cX - 20, cY - 20),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
      coord.append((cX,cY))
    cv2.imwrite('contours_count.jpg', image)
    number_of_neurons = len(coord)
    print(number_of_neurons)
    
    
  </h4>
  <h5>На этом шаге происходит нахождение центров нейронов на картинке и их изображение (чтобы оценить правильность подсчета контуров нейронов). Центры контуров записываются в отдельный список, количество элементов в котором является числом нейронов на исходной картинке
  </h5>
  
![Иллюстрация к проекту](https://github.com/wwapper/NeuroCount/blob/master/program/images/contours_count.jpeg)
              
    
