# Proyecto_MCF
Proyecto elaborado por:
Hernandez Zeferino Emmanuel 
Maldonado Morales Karla Sofia 
Chavarría Ramírez Andrea
Este proyecto es una aplicación web desarrollada en stramlit para análisis el riesgo de Tesla. Aquí se calculan métricas de riesgo como el Value Risk y Espected shortfall.
Utilizamos la Api de yahoo finance para obtener los precios históricos de tesla. 
En el proyecto se pedía asumir ciertas distrubuciones, sin embargo nos dimos a la tarea de hacer la prueba de shapiro wilk para determinar si se distribuia normal, sin embargo nuestro p value nos dio de o por lo que rechazamos a H0. 
Asi pues asumiendo sitribuciones normal y t student hacemos el calculo del VaR y CVAR, también utilizamos simulaciones Montecarlo asumineto normal y t student . 
Para validar la precisión utilizamos roling window, aquí utilizamos 252 dias para calular el VaR del día siguiente, aquí vemos si la perdida del día supera la estimación del VaR y CVAR, donde una buena estimación es si la violación es menor al 2.5%, finalmente la agregamos a una tabla donde si es menor es bueno, si no se tiene que ajustar .
Despues estimamos el VaR con una volatilidad móvil asumiendo una distribución normal y con una formula dada. Finqalmente mostramos los resultados en una tabla    
