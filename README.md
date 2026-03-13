# Autoencoder con PyTorch sobre CIFAR-10

Implementación de un Autoencoder (AE) en PyTorch orientado a reconstrucción y compresión de imágenes sobre el dataset CIFAR-10.

El proyecto está planteado con un enfoque didáctico para entender cómo una red encoder-decoder aprende una representación comprimida de los datos sin etiquetas, y cómo esa representación puede usarse para reconstruir imágenes manteniendo su estructura visual principal.

El proyecto incluye:

* carga y normalización de CIFAR-10
* arquitectura encoder-decoder con capas densas
* entrenamiento con función de pérdida MSE
* reconstrucción de imágenes de test
* comparación visual entre imágenes originales y reconstruidas
* compresión extrema de aproximadamente un 99%

> Objetivo del repositorio: entender cómo un autoencoder aprende a comprimir información visual en un espacio latente muy pequeño y reconstruirla después sin supervisión explícita.

* * *

## Contenido del repositorio

* `autoencoder.ipynb`: notebook principal con la implementación completa del autoencoder, entrenamiento y visualización de reconstrucciones.
* `README.md`: descripción general del proyecto.

* * *

## Arquitectura implementada

La red sigue una estructura encoder-decoder sencilla pero muy clara desde el punto de vista conceptual:

    Input (3 x 32 x 32 = 3072)
        ↓
    Linear (3072 → 256)
        ↓
    ReLU
        ↓
    Linear (256 → 128)
        ↓
    ReLU
        ↓
    Linear (128 → 64)
        ↓
    ReLU
        ↓
    Linear (64 → 32)
        ↓
    Espacio latente
        ↓
    Linear (32 → 64)
        ↓
    ReLU
        ↓
    Linear (64 → 128)
        ↓
    ReLU
        ↓
    Linear (128 → 3072)
        ↓
    Tanh
        ↓
    Reconstrucción (3 x 32 x 32)

La compresión es muy agresiva: se pasa de 3072 valores por imagen a solo 32 valores en el cuello de botella.

Esto supone una reducción aproximada del **98.96%**, que puede describirse de forma práctica como una **compresión del 99%**.

* * *

## Componentes principales

### 1. Encoder

El encoder se encarga de transformar la imagen original en una representación latente compacta.

Características:

* entrada aplanada de tamaño `3*32*32`
* reducción progresiva de dimensionalidad
* activaciones `ReLU` entre capas
* cuello de botella final de solo `32` dimensiones

Su papel es conservar la máxima información útil posible en un espacio extremadamente reducido.

### 2. Decoder

El decoder toma la representación comprimida y trata de reconstruir la imagen original.

Características:

* expansión progresiva desde `32` dimensiones hasta `3072`
* activaciones `ReLU` en capas intermedias
* activación final `Tanh`

Se usa `Tanh` porque las imágenes fueron normalizadas al rango `[-1, 1]`, así que la salida de la red queda alineada con la escala de entrada.

### 3. Función de pérdida

Se utiliza:

    nn.MSELoss()

Es decir, error cuadrático medio entre la imagen original y la reconstruida.

Esta elección tiene sentido porque el problema no es de clasificación, sino de reconstrucción continua píxel a píxel.

* * *

## Dataset

Se utiliza el dataset **CIFAR-10** cargado desde `torchvision.datasets`.

Características del dataset:

* imágenes RGB de tamaño `32x32`
* 10 clases diferentes
* las etiquetas no se usan durante el entrenamiento del autoencoder
* el aprendizaje es no supervisado respecto al objetivo de reconstrucción

En el notebook, las imágenes se transforman con:

    transforms.ToTensor()
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

Esto deja los valores aproximadamente en el rango:

    [-1, 1]

que encaja con la activación final `Tanh` del decoder.

* * *

## Entrenamiento

El notebook incluye un entrenamiento estándar del autoencoder con PyTorch usando:

* optimizador `Adam`
* función de pérdida `MSELoss`
* entrenamiento durante `15` épocas
* batch size de `32`

Configuración usada en el notebook:

    batch_size = 32
    lr = 0.001
    epochs = 15

Durante cada iteración:

1. la imagen se carga desde CIFAR-10
2. se aplana a dimensión `3072`
3. se pasa por encoder y decoder
4. se compara la salida con la entrada original
5. se calcula el error de reconstrucción
6. se actualizan los pesos por backpropagation

* * *

## Resultados observados

Según la ejecución guardada en el notebook, la pérdida de entrenamiento evoluciona así:

    Época [1/15], Loss: 0.0927
    Época [2/15], Loss: 0.0705
    Época [3/15], Loss: 0.0660
    Época [4/15], Loss: 0.0629
    Época [5/15], Loss: 0.0599
    Época [6/15], Loss: 0.0580
    Época [7/15], Loss: 0.0569
    Época [8/15], Loss: 0.0563
    Época [9/15], Loss: 0.0553
    Época [10/15], Loss: 0.0544
    Época [11/15], Loss: 0.0542
    Época [12/15], Loss: 0.0540
    Época [13/15], Loss: 0.0539
    Época [14/15], Loss: 0.0535
    Época [15/15], Loss: 0.0530

Se observa una reducción progresiva y estable del error, lo que indica que el modelo aprende una representación latente útil para reconstruir las imágenes.

Aunque no se busca clasificación ni métricas como accuracy, visualmente el modelo logra conservar bastante bien:

* estructura global
* distribución de color
* formas generales de los objetos

Sin embargo, al usar una compresión tan extrema, se pierden detalles finos y nitidez en la reconstrucción. Y claro: magia gratis no existe; si aplastas 3072 números en 32, algo se tiene que sacrificar.

* * *

## Visualización incluida

El notebook compara directamente imágenes originales y reconstruidas.

Incluye:

* selección de imágenes del conjunto de test
* paso del batch por el autoencoder ya entrenado
* reconstrucción y reordenación a formato `3x32x32`
* visualización lado a lado

Esto permite inspeccionar de forma cualitativa qué información logra preservar el espacio latente y qué detalles se degradan tras la compresión.

* * *

## Cómo ejecutarlo

### Requisitos

Instala las dependencias necesarias:

    pip install numpy matplotlib torch torchvision

### Ejecución

1. Clona el repositorio:

    git clone https://github.com/davidba10/Autoencoder-CIFAR10.git
    cd Autoencoder-CIFAR10

2. Abre el notebook:

    jupyter notebook autoencoder.ipynb

3. Ejecuta las celdas en orden.

* * *

## Qué demuestra este proyecto

Este repositorio demuestra comprensión práctica de:

* aprendizaje no supervisado
* compresión de información visual con redes neuronales
* uso de espacios latentes
* reconstrucción de imágenes
* entrenamiento encoder-decoder en PyTorch
* relación entre nivel de compresión y pérdida de detalle
* diseño básico de autoencoders fully connected

En otras palabras: aquí la red aprende a hacer de maleta extrema. Mete casi toda la imagen en un bolsillo ridículamente pequeño y luego intenta volver a montarla.

* * *

## Limitaciones actuales

Como proyecto de aprendizaje, tiene varias limitaciones razonables:

* todo está concentrado en un notebook
* la arquitectura usa capas densas, no convolucionales
* no se incluyen métricas perceptuales más avanzadas
* no se compara con otros autoencoders
* no hay separación formal en módulos
* no se guardan pesos ni checkpoints
* no se analiza cuantitativamente el espacio latente
* la reconstrucción puede perder bastante detalle por la compresión extrema

* * *

## Posibles mejoras futuras

Algunas mejoras naturales para una siguiente versión:

* usar un autoencoder convolucional
* comparar distintos tamaños de espacio latente
* medir el efecto real de la compresión sobre la calidad visual
* añadir visualización del espacio latente
* guardar y recargar el modelo entrenado
* incorporar métricas como PSNR o SSIM
* comparar reconstrucción en CPU vs GPU
* refactorizar el notebook en una versión más limpia para portfolio

* * *

## Conclusión

Este proyecto muestra que un autoencoder puede comprimir imágenes de CIFAR-10 en una representación de solo 32 valores, lo que equivale aproximadamente a una **compresión del 99%**, y aun así reconstruir correctamente gran parte de la estructura visual de la imagen.

La conclusión principal es que, incluso con una compresión extremadamente agresiva, la red sigue siendo capaz de conservar patrones globales relevantes, aunque sacrifica nitidez y detalles finos. Es decir, el modelo no memoriza píxeles exactos: aprende una representación condensada de la información más importante.

Esto deja una idea bastante potente: cuando una red es capaz de reconstruir algo razonablemente bien después de destruir casi toda su dimensionalidad, significa que ha capturado estructura real de los datos y no solo ruido superficial.

* * *

## Motivación

Este proyecto está pensado como ejercicio práctico para entender de forma clara cómo funciona un autoencoder por dentro:

* cómo se comprime una imagen
* cómo se reconstruye desde un espacio latente
* qué información se conserva y cuál se pierde
* cómo influye el cuello de botella en la calidad final

Es un paso natural para estudiar representación aprendida, reducción de dimensionalidad, compresión neuronal y modelos generativos más avanzados.
