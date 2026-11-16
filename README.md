# Simulador de campaña de depósitos

Este repositorio es la **representación visual de un proyecto previo de análisis de datos y selección de modelos**. En aquel proyecto se evaluaron modelos para elegir uno con el que analizar los datos de una campaña bancaria. Aquí se utiliza el modelo seleccionado, ya entrenado, en una aplicación web que permite introducir los datos de un cliente y **visualizar una predicción** sobre la contratación de un depósito a plazo fijo.

La selección y el entrenamiento del modelo pertenecen al proyecto anterior. Esta aplicación se centra en preparar los datos introducidos, ejecutar la predicción y presentar el resultado de forma comprensible.

## Qué hace la aplicación

1. Recoge información del perfil del cliente, su situación financiera y el historial de la campaña.
2. Aplica a esos datos las transformaciones definidas en `mystreamlit.py` para construir las variables que recibe el modelo.
3. Carga el modelo SVM guardado en `modelo_final.pkl` y obtiene una predicción.
4. Muestra si el modelo estima que el cliente contratará el depósito y permite consultar los datos procesados utilizados.

La interfaz está construida con Streamlit. La predicción se genera para **un cliente cada vez** al pulsar «Realizar predicción»; no se muestra una probabilidad ni se procesan archivos por lotes.

## Estructura

| Archivo | Función |
| --- | --- |
| `mystreamlit.py` | Interfaz, preparación de datos, carga del modelo y presentación de la predicción. |
| `modelo_final.pkl` | Modelo entrenado procedente del proyecto de selección. |
| `requirements.txt` | Dependencias de Python para ejecutar la aplicación. |

## Ejecución local

Se necesita Python y el archivo `modelo_final.pkl` en la raíz del repositorio. Desde esa carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
streamlit run mystreamlit.py
```

Streamlit mostrará en la terminal la dirección local de la aplicación. Para desplegarla, configura `mystreamlit.py` como archivo principal y asegúrate de incluir el modelo y las dependencias.

## Datos y preparación

El formulario solicita edad, ocupación, estado civil, educación, saldo, hipoteca, préstamo, medio de contacto, mes y duración de la última llamada, número de contactos de la campaña, días desde la campaña anterior, contactos previos y resultado anterior. El valor `-1` en los días desde la campaña anterior indica que no hubo contacto previo.

Antes de llamar al modelo, la aplicación crea la variable `contactado_antes`, marca los medios de contacto desconocidos, convierte las respuestas de hipoteca y préstamo a valores binarios, transforma el saldo con una escala logarítmica y elimina la columna `day`. También trata ciertos valores ausentes de educación y del resultado de la campaña anterior. Estas transformaciones deben ser compatibles con las utilizadas al entrenar el modelo en el proyecto original.

## Alcance de la predicción

El resultado es una **estimación del modelo**, no una confirmación de que el cliente contratará el producto. Entre las entradas está la duración de la última llamada, por lo que una predicción que use ese dato presupone que la llamada ya se realizó. La calidad de la estimación depende de los datos introducidos y del comportamiento del modelo seleccionado en el proyecto de análisis.
