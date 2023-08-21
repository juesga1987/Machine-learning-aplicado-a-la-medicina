# Machine-learning-aplicado-a-la-medicina
# Descripción del caso de estudio:
Hemos sido contratados en el equipo de ciencia de datos en una consultora de renombre. Nos han asignado a un proyecto de estudio de atención en salud para un importante hospital. **Nuestro cliente desea saber las características más importantes que tienen los pacientes de cierto tipo de enfermedad que terminan en hospitalización.** Fue definido como caso aquel paciente que fue sometido a biopsia prostática y que en un periodo máximo de 30 días posteriores al procedimiento presentó fiebre, infección urinaria o sepsis; requiriendo manejo médico ambulatorio u hospitalizado para la resolución de la complicación y como control al paciente que fue sometido a biopsia prostática y que no presentó complicaciones infecciosas en el período de 30 días posteriores al procedimiento. Dado que tienen en su base de datos algunos datos referentes a los pacientes y resultados de exámenes diagnósticos, de pacientes hospitalizados y no hospitalizados, nos han entregado esta información.
# EDA:
1. Importamos las librerias que usaremos y el archivo de excel que nos brindaron para el análisis:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel(r'C:\Users\juesg\Desktop\DATA SCIENCE\Modulo No. 6\Proyecto Integrador\Propuesta 1\BBDD_Hospitalización.xlsx')
```
2. Eliminamos aquellas columnas que no son relevantes para nuestro modelo:
```df.drop(['DIAS HOSPITALIZACION MQ', 'DIAS HOSPITALIZACIÓN UPC'], axis=1, inplace=True)```
3. Revisamos valores nulos:
``` df.info()```
4. Al encontrar valores vacíos, procedemos a imputar estos faltantes con base en probabilidad y análisis cruzados
En primera instancia abordaremos las variables numéricas:
```
for ind, elemento in enumerate(df['AGENTE AISLADO']):
    if df['TIPO DE CULTIVO'][ind] == 'NO':
        df['AGENTE AISLADO'][ind] = 'NO'
    else:
        df['AGENTE AISLADO'][ind] = df['AGENTE AISLADO'][ind]

df['EDAD'].describe()
df['EDAD'].replace(to_replace=151, value=51, regex=True, inplace=True)
df['EDAD'].replace(to_replace=143, value=43, regex=True, inplace=True)

df['PSA'].describe()

df['PSA'] = df['PSA'].fillna(df['PSA'].mean())

df['NUMERO DE MUESTRAS TOMADAS'].describe()

sns.catplot(data=df)
```
Como se puede observar, decidí usar la media en el caso de PSA y # de muestras tomadas. Para el agente aislado, al hacer referencia a el tipo de infección es improbable que si no se realizo tipo de cultivo se haya identificado la infección.
En segunda instancia, revisaremos las variables categóricas:
```
sns.catplot(data=df, x="DIABETES", y= "EDAD", kind="box")
sns.catplot(data=df, x="HOSPITALIZACIÓN ULTIMO MES", y='EDAD', kind="box")
sns.catplot(data=df, x="BIOPSIAS PREVIAS", y='EDAD', kind="box")
sns.catplot(data=df, x="VOLUMEN PROSTATICO", y='EDAD', kind="box")
sns.catplot(data=df, x="ANTIBIOTICO UTILIAZADO EN LA PROFILAXIS", y='EDAD', kind="bar")
sns.catplot(data=df, x="CUP", y='EDAD', kind="bar")
sns.catplot(data=df, x="ENF. CRONICA PULMONAR OBSTRUCTIVA", y='EDAD', kind="bar")

df['BIOPSIA'].value_counts()

df['NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACIÓN INFECCIOSA'].value_counts()
df['NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACIÓN INFECCIOSA'].replace(to_replace='NO', value=0, inplace=True)
df['NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACIÓN INFECCIOSA'].astype(int)
df['NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACIÓN INFECCIOSA'].value_counts()

df['FIEBRE'].value_counts()
df['ITU'].value_counts()
df['TIPO DE CULTIVO'].value_counts()
df['AGENTE AISLADO'].value_counts()
df['PATRON DE RESISTENCIA'].value_counts()
```
Se identifico que para los nan en hospitalización los dias de hospitalización eran 0. Por lo que se procederá a imputar estos faltantes con NO.
```
df['HOSPITALIZACION'].fillna('NO', inplace=True)
```
Imputaremos los faltantes en biopsias previas por No. Dada la probabilidad de que sean NO.
```
df['BIOPSIAS PREVIAS'].value_counts()
df['BIOPSIAS PREVIAS'] = df['BIOPSIAS PREVIAS'].fillna('NO')
```
Imputaremos los faltantes de volumen prostático con SI dada la probabilidad.
```
df['VOLUMEN PROSTATICO'] = df['VOLUMEN PROSTATICO'].fillna('SI')
```
Imputaremos los faltantes de CUP prostático con NO dada la probabilidad
```
df['CUP'] = df['CUP'].fillna('NO')
```
Imputaremos los faltantes de enfermedad crónica pulmonar  con NO dada la probabilidad
```
df['ENF. CRONICA PULMONAR OBSTRUCTIVA'] = df['ENF. CRONICA PULMONAR OBSTRUCTIVA'].fillna('NO')
```
5. Dado que las variables numéricas se encuentran en escalas distintas, procedemos a normalizas:
```
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x = df['PSA']
x = np.array(x)
x = x.reshape(-1,1)
np.shape(x)
std_data = ss.fit_transform(x)
sns.scatterplot(data=std_data)
sns.scatterplot(data=x)
df['PSA_STD'] = std_data


x = df['NUMERO DE MUESTRAS TOMADAS']
x = np.array(x)
x = x.reshape(-1,1)
np.shape(x)
std_data2 = ss.fit_transform(x)
sns.scatterplot(data=std_data)
sns.scatterplot(data=x)
df['NUMERO DE MUESTRAS TOMADAS STD'] = std_data2
```
6. Dado que para usar el modelo de Machine learning nuestras variables categóricas deben ser legibles para el algoritmo, vamos a usa label encoder:
```
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df['DIABETES'])
df['DIABETES_L'] = le.transform(df['DIABETES'])

le = LabelEncoder()
le.fit(df['HOSPITALIZACIÓN ULTIMO MES'])
df['HOSPITALIZACIÓN_ULTIMO_MES_L'] = le.transform(df['HOSPITALIZACIÓN ULTIMO MES']) 

le = LabelEncoder()
le.fit(df['BIOPSIAS PREVIAS'])
df['BIOPSIAS_PREVIAS_L'] = le.transform(df['BIOPSIAS PREVIAS'])

le = LabelEncoder()
le.fit(df['VOLUMEN PROSTATICO'])
df['VOLUMEN_PROSTATICO_L'] = le.transform(df['VOLUMEN PROSTATICO'])

le = LabelEncoder()
le.fit(df['ANTIBIOTICO UTILIAZADO EN LA PROFILAXIS'])
df['ANTIBIOTICO_L'] = le.transform(df['ANTIBIOTICO UTILIAZADO EN LA PROFILAXIS'])

le = LabelEncoder()
le.fit(df['CUP'])
df['CUP_L'] = le.transform(df['CUP'])

le = LabelEncoder()
le.fit(df['ENF. CRONICA PULMONAR OBSTRUCTIVA'])
df['ENFERMEDAD_PULMONAR_L'] = le.transform(df['ENF. CRONICA PULMONAR OBSTRUCTIVA'])

le = LabelEncoder()
le.fit(df['BIOPSIA'])
df['BIOPSIA_L'] = le.transform(df['BIOPSIA'])

le = LabelEncoder()
le.fit(df['FIEBRE'])
df['FIEBRE_L'] = le.transform(df['FIEBRE'])

le = LabelEncoder()
le.fit(df['ITU'])
df['ITU_L'] = le.transform(df['ITU'])

le = LabelEncoder()
le.fit(df['TIPO DE CULTIVO'])
df['TIPO_DE_CULTIVO_L'] = le.transform(df['TIPO DE CULTIVO'])

le = LabelEncoder()
le.fit(df['AGENTE AISLADO'])
df['AGENTE_AISLADO_L'] = le.transform(df['AGENTE AISLADO'])

le = LabelEncoder()
le.fit(df['PATRON DE RESISTENCIA'])
df['PATRON_DE_RESISTENCIA_L'] = le.transform(df['PATRON DE RESISTENCIA'])

le = LabelEncoder()
le.fit(df['HOSPITALIZACION'])
df['HOSPITALIZACION_L'] = le.transform(df['HOSPITALIZACION'])
```
7. Con el fin de tener un backup de lo hecho hasta el momento y posteriormente eliminar las columnas no normalizadas y etiquetadas:
```
df.to_excel('df_sindrop.xlsx')
```
8. Eliminamos las columnas nombradas en el punta anterior:
```
df.drop(['DIABETES','HOSPITALIZACIÓN ULTIMO MES','BIOPSIAS PREVIAS','VOLUMEN PROSTATICO','ANTIBIOTICO UTILIAZADO EN LA PROFILAXIS','CUP','ENF. CRONICA PULMONAR OBSTRUCTIVA','BIOPSIA', 'FIEBRE', 'ITU', 'TIPO DE CULTIVO', 'AGENTE AISLADO', 'PATRON DE RESISTENCIA','PSA', 'NUMERO DE MUESTRAS TOMADAS', 'NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACIÓN INFECCIOSA','HOSPITALIZACION'], axis=1, inplace=True)
```
9. Realizamos una matriz de correlación con el fin de identificar la relación entre las variables y poder determinar cuales son mas importantes para el modelo de machine learning que haremos posteriormente:
```
def plot_corre_heatmap(corr):
    '''
    Definimos una función para ayudarnos a graficar un heatmap de correlación
    '''
    plt.figure(figsize=(20,20))
    sns.heatmap(corr, cbar = True,  square = False, annot=True, fmt= '.2f'
                ,annot_kws={'size': 10},cmap= 'coolwarm')
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    # Arreglamos un pequeño problema de visualización
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show()

corr = df.corr()
plot_corre_heatmap(corr)
```
10. Por ultimo en este proceso de EDA y ya con la base de datos lista, exportaremos a un archivo de tipo CSV:
```
df.to_csv('mlfile.csv')
```

