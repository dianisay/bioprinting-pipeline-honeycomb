El sistema integra principalmente tres elementos: una cámara para localizar el objetivo, una plataforma cartesiana XY para realizar el posicionamiento inicial y el brazo robótico MyCobot 280 para ejecutar el posicionamiento final.
Primero, el programa inicializa la cámara a una resolución de 1280 × 720 y carga previamente los parámetros de calibración de la cámara. Estos parámetros permiten corregir la distorsión de la imagen y convertir las coordenadas obtenidas en píxeles a coordenadas físicas aproximadas en milímetros. Para esta conversión se utiliza actualmente una profundidad de trabajo de 650 mm.
También se carga una red neuronal previamente entrenada, cuya función es transformar las coordenadas obtenidas a partir de la cámara ((X_c,Y_c)) en una estimación de la posición del objetivo dentro del sistema de coordenadas del robot.
La primera parte del movimiento se realiza mediante la plataforma cartesiana XY. Al iniciar, la plataforma hace su procedimiento de homing y posteriormente establece un origen de trabajo.
Sobre esta plataforma se implementa un controlador PD en los ejes X y Y. El controlador calcula continuamente el error entre la posición estimada del objetivo y la posición actual de la plataforma, y genera pequeños desplazamientos para reducirlo. Actualmente se tiene una tolerancia de aproximadamente 5 mm y un desplazamiento máximo de 15 mm por ciclo para evitar movimientos demasiado bruscos.
Durante esta etapa, la cámara trabaja en tiempo real. En cada imagen se realiza:
Corrección de la distorsión de la cámara.
Conversión de la imagen a escala de grises.
Ajuste de contraste.
Segmentación binaria.
Operaciones morfológicas para eliminar ruido y cerrar regiones.
Identificación de la región de mayor área.
Obtención del centroide del objeto detectado.
Ese centroide se utiliza como referencia para determinar dónde se encuentra el objetivo.
Posteriormente, las coordenadas obtenidas a partir de la imagen se transforman de píxeles a milímetros y pasan por la red neuronal. La salida de la red representa la posición estimada del objetivo dentro del sistema de referencia utilizado para el posicionamiento.
El sistema lee, cuando es posible, la posición real de la plataforma cartesiana. El código está preparado para interpretar retroalimentación procedente de Marlin/RepRap, GRBL o Klipper. Si temporalmente no se consigue una lectura válida, utiliza una estimación interna de la posición y reduce automáticamente la agresividad del controlador para continuar de manera más conservadora.
La plataforma continúa corrigiendo su posición hasta que la distancia con respecto al objetivo es de aproximadamente 4 mm o menor.
Cuando se alcanza esta condición, la plataforma cartesiana se detiene y comienza la segunda etapa, correspondiente al MyCobot 280.
Antes de mover el brazo se toma una nueva imagen para recalcular el objetivo y disminuir el error que pudiera haberse acumulado durante el movimiento de la plataforma.
En esta etapa también se obtiene la orientación real del MyCobot, específicamente su ángulo de yaw, y se compensa la posición del objetivo de acuerdo con dicha orientación.
Además, el código limita las coordenadas al espacio de trabajo permitido del robot y utiliza actualmente un radio de seguridad de 260 mm. Si la posición calculada se encuentra fuera de este radio, el objetivo se desplaza automáticamente al límite seguro antes de ejecutar el movimiento.
Una vez definido el punto final, el MyCobot genera una trayectoria suave mediante funciones sigmoidales. La posición en X, Y y Z cambia progresivamente desde la posición inicial hasta la posición objetivo y, al mismo tiempo, se modifica gradualmente el yaw para evitar cambios bruscos de orientación.
Actualmente, el punto final de esta etapa se define aproximadamente como:

[X_{objetivo},Y_{objetivo},320 { mm}]

conservando roll y pitch y modificando principalmente el yaw hacia el objetivo.
Durante el movimiento del MyCobot se obtiene telemetría de su posición real. Con estos datos se compara la trayectoria deseada con la trayectoria ejecutada y se calcula el error en X, Y y Z.
Al finalizar, el programa calcula:
Error final con respecto a la posición objetivo.
RMSE en X.
RMSE en Y.
RMSE en Z.
RMSE tridimensional.
También genera gráficas de la trayectoria deseada contra la trayectoria real, el error por cada eje y la norma total del error.
Programas y dependencias necesarias
Para ejecutar el sistema es necesario contar principalmente con:
1. MATLAB
Es el programa principal desde el cual se ejecuta todo el sistema.
En MATLAB se realizan:
Adquisición y procesamiento de las imágenes.
Segmentación del objetivo.
Conversión de píxeles a milímetros.
Ejecución de la red neuronal.
Control PD de la plataforma XY.
Comunicación serial con la plataforma.
Comunicación con Python.
Generación de trayectorias.
Cálculo de errores y RMSE.
Generación de las gráficas de resultados.
Por las funciones utilizadas, MATLAB debe tener disponibles las herramientas correspondientes a procesamiento de imágenes, adquisición de cámara y redes neuronales.
2. Python 3.11
El control del MyCobot se realiza desde MATLAB utilizando Python.
Actualmente el código está configurado para utilizar específicamente:
Python 3.11
MATLAB abre Python en modo OutOfProcess y posteriormente importa el módulo que realiza la comunicación con el robot.
3. Archivo/módulo mc_bridge.py
Este archivo funciona como puente entre MATLAB y el MyCobot.
Desde MATLAB se importa como:
mc_bridge
y mediante este módulo se ejecutan funciones como:
Home del robot.
Lectura de la pose actual.
Selección del modo de movimiento.
Movimiento cartesiano del MyCobot.
Por lo tanto, este archivo debe mantenerse disponible en la carpeta del proyecto o en una ruta que Python pueda reconocer.
4. Librería de Python para comunicación con el MyCobot
El archivo mc_bridge.py necesita las librerías correspondientes para comunicarse con el MyCobot.
El código principal que te comparto no muestra directamente qué paquetes importa internamente mc_bridge.py, por lo que las dependencias exactas deben revisarse dentro de ese archivo. Normalmente esta parte contiene la comunicación específica con el robot.
Archivos necesarios que ya se encuentran en la carpeta del proyecto
Además del código principal, hay varios archivos que deben permanecer junto con el proyecto:
cameraParams.mat
Contiene los parámetros intrínsecos obtenidos durante la calibración de la cámara.
De este archivo se obtienen parámetros como:
(f_x)
(f_y)
(c_x)
(c_y)
y se utiliza también para corregir la distorsión de cada imagen.
trainedNet.mat
Contiene la red neuronal previamente entrenada y los parámetros utilizados para normalizar y desnormalizar los datos.
Específicamente el programa carga:
net
psX
psT
Sin este archivo no se puede realizar la transformación entre las coordenadas detectadas por la cámara y las coordenadas utilizadas para localizar el objetivo.
mc_bridge.py
Es el archivo de comunicación entre MATLAB/Python y el MyCobot.
Código principal de MATLAB
Es el archivo que coordina todas las etapas anteriores y desde el cual se debe iniciar el sistema.
Comunicación con la plataforma cartesiana
La plataforma XY se conecta por puerto serial.
Actualmente el código tiene configurado:
COM5 a 115200 baud
Por lo tanto, antes de ejecutar el programa únicamente hay que verificar que Windows haya asignado el mismo puerto COM. Si cambia, se modifica esa línea del código.
El firmware/controlador de la plataforma debe responder a alguno de los protocolos contemplados en el programa:
Marlin/RepRap.
GRBL.
Klipper.
El código intenta identificar automáticamente cuál de estas respuestas está disponible para obtener la posición X-Y actual.
Cámara
También es necesario que la cámara esté conectada y sea reconocida por MATLAB.
Actualmente se solicita una resolución de 1280 × 720
La cámara debe permanecer aproximadamente en la configuración geométrica con la que se realizó la calibración, especialmente porque la transformación de las coordenadas utiliza los parámetros de calibración y una profundidad de referencia de 650 mm.
MyCobot
El MyCobot también debe estar conectado y disponible para Python antes de ejecutar la segunda parte del programa.
Al iniciar, el código manda al brazo a la configuración HOME:
[0,-80,120,60,30,0]
y posteriormente utiliza mc_bridge.py para obtener su posición y enviar los comandos cartesianos.
En resumen, la estructura de software queda de esta forma:
MATLAB
↓
Cámara + procesamiento de imagen
↓
cameraParams.mat
↓
Detección del centroide
↓
trainedNet.mat / red neuronal
↓
Coordenadas objetivo
↓
Control PD → plataforma XY por puerto serial
↓
Python 3.11
↓
mc_bridge.py
↓
MyCobot 280
↓
Trayectoria final + telemetría + cálculo de errores
Y el flujo funcional completo es:
Cámara → segmentación del objetivo → centroide → conversión píxel-mm → red neuronal → coordenadas objetivo → control PD de plataforma XY → corrección final con cámara → compensación por orientación del MyCobot → trayectoria suave del brazo → posicionamiento final → cálculo del error.
La idea es que la plataforma cartesiana realice el posicionamiento inicial y reduzca el error global, mientras que el MyCobot se encargue de la aproximación final al punto identificado mediante visión.
Los archivos complementarios mencionados (cameraParams.mat, trainedNet.mat y mc_bridge.py) se encuentran dentro de la carpeta que te comparto junto con el código principal para que se mantenga toda la configuración del sistema en un mismo lugar.
