# Reinforcement-Learning-Basics
Este repositorio contiene una implementación de Deep Q-Learning (DQL) para resolver el entorno clásico de Blackjack utilizando la biblioteca gymnasium. El agente utiliza una red neuronal profunda (DQN) para aprender a jugar al Blackjack óptimamente, buscando maximizar las recompensas a largo plazo.
El agente aprende a tomar decisiones óptimas para maximizar sus recompensas en el juego. 
El código utiliza la librería PyTorch para definir y entrenar la red neuronal que aproxima la función Q.

## Estructura del código utilizada
El código se divide en las siguientes partes principales:

1. Red neuronal: DuelingDQN
Esta red se usa para estimar los valores de Q, asigna un valor a cada par, representando la recompensa esperada al tomar una acción en un estado dado.

Arquitectura de la red:
La red neuronal está definida en la clase DQN, que hereda de nn.Module (la clase base de PyTorch para redes neuronales).

  * Capa densa (fc1): x neuronas <- Esta capa es una capa lineal que transforma el estado de entrada en un espacio de x dimensiones.
  * Capa densa (fc2): x neuronas <- Esta capa lineal transporta las x dimensiones en otro espacio de x dimensiones.
  * Capa densa (fc3): x neuronas <- Esta capa produce los valores Q para cada acción posible.

2. Función de activación ReLU
La red utiliza la función de activación ReLU (Rectified Linear Unit) después de las dos primeras capas lineales. ReLU se define como:
ReLU(x)=max(0,x)
ReLU introduce no linealidad en la red, permitiendo que el modelo aprenda patrones complejos en los datos.
Además, ReLU es computacionalmente eficiente y ayuda a evitar el problema del gradiente vanishing.

3. Agente de Deep-Q-Learning
El agente DQLAgent utiliza la red neuronal DQN para aprender la política óptima. A continuación, se describen los componentes clave del agente:
Hiperparámetros
* gamma: Factor de descuento (0.99). Controla la importancia de las recompensas futuras.
* epsilon: Tasa de exploración (1.0 inicialmente). Controla la probabilidad de que el agente elija una acción aleatoria.
* epsilon_decay: Tasa de decaimiento de epsilon (0.999). Reduce epsilon con el tiempo para equilibrar exploración y explotación.
* epsilon_min: Valor mínimo de epsilon (0.01).
* learning_rate: Tasa de aprendizaje (0.0015). Controla la magnitud de los ajustes en los pesos de la red.
* batch_size: Tamaño del minibatch (32). Número de experiencias utilizadas para entrenar la red en cada paso.

4. Memoria
El agente utiliza una memoria de repetición (deque) para almacenar experiencias (estado, acción, recompensa, siguiente estado, done). Esto permite reutilizar experiencias pasadas para entrenar la red.

El agente selecciona acciones utilizando una política ε-greedy:
Con probabilidad epsilon, elige una acción aleatoria (exploración).
De lo contrario, elige la acción con el mayor valor Q según la red neuronal (explotación).
python
Copy
def act(self, state):
    if np.random.rand() <= self.epsilon:
        return random.randrange(self.action_size)  # Exploración
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)  # Explotación
        return np.argmax(q_values.numpy())

5. Entrenamiento de la Red (Replay)
El agente entrena la red neuronal utilizando un minibatch de experiencias almacenadas en la memoria.
El proceso incluye:
* Calcular los valores Q objetivo.
* Calcular la pérdida entre los valores Q predichos y los valores Q objetivo.

6. Entrenamiento y evaluación
Entrenamiento:
 El agente se entrena durante 5000 episodios en el entorno de Blackjack.
 En cada episodio:
  El agente interactúa con el entorno, almacenando experiencias en la memoria.
  Después de cada episodio, se entrena la red neuronal utilizando la memoria de repetición.

Evaluación:
 Después del entrenamiento, el agente se evalúa en 1000 episodios de prueba. Se registran las victorias, derrotas y empates, y se calcula la  tasa de victorias.
 Ajustar los pesos de la red mediante retropropagación.

7. Optimización
Con DuelingDQN se separa la estimación del valor del estado  y la ventaja de cada acción. 
Prioritized Experience Replay se encarga de priorizar las experiencias con mayor error TD.

8. Resultados
El código incluye gráficos para visualizar:
Tasa de victorias a lo largo del tiempo.
Distribución final de resultados (victorias, derrotas, empates).

9. Conclusiones
Este proyecto demuestra cómo una red neuronal simple puede utilizarse para aproximar la función Q en un entorno de Blackjack. La combinación de Deep Q-Learning y redes neuronales permite que el agente aprenda una política óptima mediante la exploración y explotación del entorno.

Podemos ajustar la estructura de recompensas para seguir estrategias óptimas en el BlackJack, incluir estadísticas y encontrar mejores parámetros en las redes neuronales para lograr un mejor aprendizaje y pueda ganar más veces de las que pierde.
