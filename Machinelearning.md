# Modelamiento -  Machine Learning
1. Importamos la base de datos del archivo CSV creado en el EDA:
```
df = pd.read_csv(r'C:\Users\juesg\Desktop\DATA SCIENCE\Modulo No. 6\Proyecto Integrador\Propuesta 1\mlfile.csv', sep=',')
```
2. Creamos las variables independientes y dependientes, X y Y respectivamente:
```
x = df[['FIEBRE_L','ITU_L', 'AGENTE_AISLADO_L', 'PSA_STD','EDAD']]
y = df['HOSPITALIZACION_L']
```
3. Separamos la muestra de entrenamiento y test. Usamos un valor arbitratio como profundidad del arbol:
```
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=4)

```
4. Entrenamos y predecimos:
```
dtc.fit(xtrain, ytrain)
ypred = dtc.predict(xtest)
```
5. Usamos la matriz de confusion(1) para comenzar a evaluar los resultados:
```
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

matrix = confusion_matrix(ypred, ytest)

cm_display = ConfusionMatrixDisplay(matrix)

fig, ax = plt.subplots(figsize=(4,5))

ax.matshow(matrix)
for (i, j), z in np.ndenumerate(matrix):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
plt.show()
```
6. Evaluamos el modelo a traves de las metricas accuracy, F1 Score y ROC AUC Score:
```
rom sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print('Accuracy score =', accuracy_score(ytest, ypred))
print('F1 score = ', f1_score(ypred, ytest))
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score


def multiclass_roc_auc_score(ytest, ypred, average= 'macro'):
    lb = LabelBinarizer()
    lb.fit(ytest)
    ytest = lb.transform(ytest)
    ypred = lb.transform(ypred)
    return roc_auc_score(ytest, ypred, average=average)

print('Curva ROC score = ',multiclass_roc_auc_score(ytest, ypred))
```
Estos son los resultados:
Accuracy score = 0.9941520467836257
F1 score =  0.9411764705882353
Curva ROC score =  0.9444444444444444

Podemos concluir:
a. Las tres metricas presentan resultados bastante buenos lo que nos indica que el modelo entrenamos esta en capacidad de hacer una buena clasificacion con base en los sintomas e informacion relevante de si el paciente sera o no hospitalizado.
b. Las variables seleccionadas para entrenar al modelo con base en la correlacion realizada durante el EDA permiten llegar a este grado de precision.
c. A pesar que el resultado es muy bueno, debemos ser cautelosos, pues en algunos casos esto es muestra de un overfitting que impediria al generalizar lograr un resultado correcto.

7. Con el fin de evitar Overfitting, usaremos la busqueda de grilla y validacion cruzada para encontrar los mejores los mejores valores para los hiperparametros:
```
from sklearn.model_selection import GridSearchCV

param_grid = {
              'criterion': ['gini', 'entropy'],
              'max_depth':[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]
              }

model = GridSearchCV(dtc, param_grid=param_grid, cv=10)
model.fit(xtrain, ytrain)

print(model.best_estimator_.get_params()['criterion'])
print(model.best_estimator_.get_params()['max_depth'])
```
Los resultados nos indican que debemos usar una profundidad de arbol igual 2 y el indice Gini como criterio de entrenamiento.
8. Precedemos a entrenar y predecir nuevamente el modelo con los valores para hiperparametros arrojados en el punto anterior:
```
dtc2 = DecisionTreeClassifier(max_depth=2, criterion='gini')
dtc2.fit(xtrain, ytrain)
ypred2 = dtc2.predict(xtest)
```
9. Realizamos la matriz de confusion(2):
```
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

matrix = confusion_matrix(ypred2, ytest)

cm_display = ConfusionMatrixDisplay(matrix)

fig, ax = plt.subplots(figsize=(4,5))

ax.matshow(matrix)
for (i, j), z in np.ndenumerate(matrix):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
plt.show()
```
10. Calculamos las metricas de evaluacion que seleccionamos al principio de este ejercicio:
```
print('Accuracy score =', accuracy_score(ytest, ypred2)) 
print('F1 score = ', f1_score(ypred2, ytest))
print('Curva ROC score = ',multiclass_roc_auc_score(ytest, ypred2))
```
Accuracy score = 0.9824561403508771
F1 score =  0.8
Curva ROC score =  0.8333333333333333

Como lo suponiamos cuando se entremo el modelo con una variable de profundidad arbitraria el modelo inicial estaba sobre estimado. Y a pesar que las metricas arrojadas al optimizar el modelo son aparentemente menores en terminos de eficacia de clasificacion a la hora de generalizar y evaluar nuevos datos el modelo sera mas preciso.
11. Con el fin de entrenar el modelo con otro algoritmo para comparar resultados y determinar cual es mejor, usaremos vecinos cercanos:
```
from sklearn.model_selection import GridSearchCV

param_grid = {
              'algorithm': ['auto', 'ball_tree','Kd_tree','brute'],
              'n_neighbors':[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]
              }


model = GridSearchCV(knn, param_grid=param_grid, cv=10)
model.fit(xtrain, ytrain)

print(model.best_estimator_.get_params()['algorithm'])
print(model.best_estimator_.get_params()['n_neighbors'])
```
Los resultados de optimizacion nos indican que debemos usar el hiperparametro algoritmo auto y un vecino.
12. Entrenamos y predecimos:
```
knn = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
knn.fit(xtrain, ytrain)
ypred4 = knn.predict(xtest)
```
13. Realizamos la matriz de confusion(3):
```
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

matrix = confusion_matrix(ypred4, ytest)

cm_display = ConfusionMatrixDisplay(matrix)

fig, ax = plt.subplots(figsize=(4,5))

ax.matshow(matrix)
for (i, j), z in np.ndenumerate(matrix):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
plt.show()
```
14. Evaluamos los resultados:
```
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


print('Accuracy score =', accuracy_score(ytest, ypred4)) 
print('F1 score = ', f1_score(ypred4, ytest))
print('Curva ROC score = ',multiclass_roc_auc_score(ytest, ypred4))
```
Los resultados son:
Accuracy score = 0.9590643274853801
F1 score =  0.5333333333333333
Curva ROC score =  0.7160493827160493
# Como conclusion general del modelo evaluado:
Habiendo evaluado los modelos del arbol de decision y vecinos mas cercanos y optimizandolos podermos concluir que el modelo de mejor desempeno fue el arbol de aprendizaje, a continuacion un breve resumen de las metricas de evaluacion:

Arbol de decision optimizado:

Accuracy = 0.9824
F1 score = 0.8
Curva ROC = 0.8333

Vecinos mas cercanos optimizado:

Accuracy = 0.9590
F1 score = 0.5333
Curva ROC = 0.7160

Como podemos observar el % de errores totales (Accuracy), precision y exhaustividad (F1) y capacidad de realizar una correcta clasificacion (Curva ROC) son mayores para el arbol de decision. Al ser este un modelo que busca entrenar y predecir de forma acertada
la posibilidad de que un pacientes sometido a biopsia de prostata tenga complicaciones y sea hospitalizado o no, debemos seleccionar aquel que tenga la mayor precision a la hora de realizar la clasificacion con los sintomas indicados fiebre, infeccion o ITU acompanados de la incidencia de la edad y el nivel de antigeno prostatico, por lo que nuestra recomendacion para el cliente es usar el modelo de arbol de decision con parametros optimizados.

