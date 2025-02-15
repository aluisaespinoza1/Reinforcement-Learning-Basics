# Reinforcement-Learning-Basics
Este repositorio contiene una implementación de Deep Q-Learning (DQL) y LSTM (Red Neuronal Recurrente) para resolver el entorno clásico de Blackjack utilizando la biblioteca gymnasium. El agente utiliza una red neuronal profunda (DQN) para aprender a jugar al Blackjack óptimamente, buscando maximizar las recompensas a largo plazo.

## Estructura del código utilizada
1. Red neuronal: DuelingDQN_LSTM
Esta red se usa para estimar los valores de Q, la cual incluye:
  * Capa densa (fc1): x neuronas
  * Capa LSTM (lstm): x neuronas recurrentes
  * Capa densa (fc2): x neuronas
  * Flujo de valor y de ventaja:
    value_stream = estima el valor del estado actual
    advantage_stream = estima la ventaja de cada acción en ese estado

2. Agente de aprendizaje DQLAgent
Este agente es el que toma decisiones, almacena experiencias y ajusta sus pesos con base a las recompensas obtenidas.
Funciones:
init = inicializa el agente con los parámetros clave
remember(state, action, reward, next_state, done) = va a almacenar las experiencias en la memoria usando Prioritized Experience Replay
act(state, hidden) = selecciona una acción basada en epsilon
replay() = entrena la red con un minibatch de experiencias priorizadas
update_target_model = copia los pesos de la red principal en nuestra red objetivo

## ¿Cómo se realizó el entrenamiento?
Pusimos x episodios siguiendo esta lógica:
1. Inicializamos el entorno de BlackJack de gymnasium
2. Reseteamos el estado del juego y se estableció una variable hidden para la LSTM
3. El agente elige acciones hasta que el juego se termine
4. Se va almacenando la experiencia y se va entrenando la red
5. Cada x episodios se actualiza la red objetivo
6. Cada x episodios se imprime el progreso del entrenamiento

## ¿Cómo se optimizó el modelo?
Con DuelingDQN se separa la estimación del valor del estado  y la ventaja de cada acción. 
Prioritized Experience Replay se encarga de priorizar las experiencias con mayor error TD.
El uso de LSTM permite recordar las secuencias del juego y se utiliza Dropout para evitar un sobreajuste en el código.

## Posibles mejoras
Podemos ajustar la estructura de recompensas para seguir estrategias óptimas en el BlackJack, incluir estadísticas y encontrar mejores parámetros en las redes neuronales para lograr un mejor aprendizaje y pueda ganar más veces de las que pierde.

## Conclusión
Este código de Deep-Q-Learning con redes recurrentes nos permite mejorar el aprendizaje y hacer que aprenda sobre el juego de BlackJack y pueda ganar más cada vez que vuelve a jugar ya que guarda en la memoria los juegos pasados y va aprendiendo.
