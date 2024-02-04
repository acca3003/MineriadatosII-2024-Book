``` r
library(genalg) library(MASS)
```

``` r
data(iris) set.seed(999) # Alas variables reales del dataset, les añadimos 10 variables más ficiticas (normal 0,1) # Ponemos estas variables al principio X <- cbind(scale(iris[,1:4]),matrix(rnorm(10*150), 150, 10)) Y <- iris[,5]
```

``` r
iris.evaluate <- function(indices) {
  ```
  
  ``` r
  result = 1 # Tiene que haber al menos 2 variables if (sum(indices) > 2) {
  ```
  
  ``` r
  # Creamos un modelo de clasificación con LDA usando sólo las variables que vienen
  # marcadas en la variablindices con valor 1
  # El LDA tiene el valor $posterior que devuelve la probabilidad de cada clase (tenemos tres clases en Y)
  # Podríamos usar el valor de $class y así tendríamos directamente la clase.
  modelo_lda <- lda(X[,indices==1], Y, CV=TRUE)$posterior
  
  # Cogemos la probabilidad más alta apply(modelo_lda, 1,function(x)which(x == max(x))), para cada fila (150 muestras)
  # Comprobamos cuantos hemos fallado y lo dividimos por el tamaño de Y (150 muestras)
  # El objetivo es que sea mínimo este número de fallos
  result = sum(Y != dimnames(modelo_lda)[[2]][apply(modelo_lda, 1,function(x)which(x == max(x)))]) / length(Y)}
```

``` r
result }
```

``` r
Ejecutamos el algoritmo evolutivo
```

``` r
system.time({ modelo_evolutivo <- rbga.bin(size=14, mutationChance=0.05, zeroToOneRatio=1,evalFunc=iris.evaluate, verbose=FALSE, iters = 50, popSize = 100) }) ## user system elapsed ## 17.05 0.72 22.03
```

``` r
Veamos como ha quedado el resultado de la población segúnla gráfica que nos da cuantas veces aparece cada variable en la población final. library(ggplot2)
```

``` r
Mostramos cual es el que ha aparecido más veces en la última iteración
```

``` r
uso_variables <- colSums(modelo_evolutivo$population)
```

``` r
Visualizamos cuanto se ha usado cada variable en esta última población
```

``` r
Construimos un dataframe con los datos
```

``` r
datos <- data.frame(variables=c(1:14),uso=uso_variables)
```

``` r
ggplot(datos,aes(x=variables,y=uso, fill=uso)) + geom_bar(stat="identity") + theme_minimal()
```

``` r
Obtenemos ahora cuales son las variables con los valores más altos. Primero vemos las suma de las columnas, que nos dará los valores de cuantas veces aparece cada variable, y luego hacemos una función que nos dice cuales son las variables más usadas pasándole como parámetro el vector de uso_variables y luego cuantas variables queremos quedarnos. # Creamos una función que nos da de un vector los índices de posición # donde están las X variables más usadas # Le pasamos como parámetro el vector y cuantas variables queremos que devuelva




```


  
  ``` r
  posicion_maximos <- function(datos, cuantos) { variables <- NULL if( cuantos>0) { for( i in 1:cuantos ) { variables[i] <- which.max(datos) datos[variables[i]] <- 0 }
    ```
    
    ``` r
  } variables
  ```
  
  ``` r
  }
  ```
  
  ``` r
  Obtenemos ahora cuales son las 6 variables más usados
  ```
  
  ``` r
  posicion_maximos(uso_variables, 6) ## [1] 2 4 1 3 8 13 
  ```