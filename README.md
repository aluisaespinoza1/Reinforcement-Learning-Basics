# Reinforcement-Learning-Basics
Este repositorio contiene una implementación de Deep Q-Learning (DQL) para resolver el entorno clásico de Blackjack utilizando la biblioteca gymnasium. El agente utiliza una red neuronal profunda (DQN) para aprender a jugar al Blackjack óptimamente, buscando maximizar las recompensas a largo plazo.
El agente aprende a tomar decisiones óptimas para maximizar sus recompensas en el juego. 
El código utiliza la librería PyTorch para definir y entrenar la red neuronal que aproxima la función Q.

## Estructura del código utilizada
El código se divide en los siguientes componentes principales:

1. Agente (Agent):
Es el jugador que toma decisiones en el entorno. En este proyecto, el agente está representado por la clase DQLAgent.
Responsabilidades del Agente:
 * Tomar decisiones: Elige acciones (como "hit" o "stand") basándose en su política actual.
 * Aprender: Actualiza su conocimiento (la red neuronal) a partir de las recompensas y experiencias.
 * Explorar vs. Explotar: Decide si explorar (acciones aleatorias) o explotar (acciones basadas en lo aprendido).

2. Ambiente (Environment):
El entorno es el "mundo" en el que el agente interactúa.
En este caso, el entorno es el juego de Blackjack, proporcionado por la librería gymnasium.
Responsabilidades del Entorno:
 * Proporcionar estados: Devuelve el estado actual del juego (por ejemplo, la suma de cartas del jugador, la carta del dealer y si tiene un as usable).
 * Recibir acciones: Acepta la acción del agente (como "hit" o "stand").
 * Devolver recompensas: Proporciona una recompensa basada en la acción del agente (por ejemplo, +1 por ganar, -1 por perder, 0 por empate).
 * Indicar si el episodio ha terminado: Devuelve un indicador done que señala si el juego ha terminado.

3. Estado:
El estado representa la situación actual en el entorno.
En Blackjack, el estado es una tupla que contiene:
 * La suma de las cartas del jugador.
 * La carta visible del dealer.
 * Si el jugador tiene un as usable.

4. Acción (Action):
Las acciones son las decisiones que el agente puede tomar en cada estado. En Blackjack, las acciones son:
 * 0: "Hit" (pedir otra carta).
 * 1: "Stand" (quedarse con las cartas actuales).

5. Recompensa (Reward):
La recompensa es un valor numérico que el entorno devuelve después de que el agente toma una acción.
En Blackjack:
 * +1: Si el jugador gana.
 * -1: Si el jugador pierde.
 * 0: Si hay un empate.

6. Política:
La política es la estrategia que el agente utiliza para elegir acciones en cada estado.
En este caso, la política es ε-greedy:
 * Con probabilidad epsilon, elige una acción aleatoria (exploración).
 * De lo contrario, elige la acción con el mayor valor Q según la red neuronal (explotación).
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

7. Función del Valor
La función de valor (en este caso, la función Q) estima la recompensa esperada al tomar una acción en un estado dado. La red neuronal (DQN) se utiliza para aproximar esta función.

8. Memoria
La memoria de repetición almacena experiencias pasadas (estado, acción, recompensa, siguiente estado, done) para entrenar la red neuronal. Esto ayuda a romper la correlación entre experiencias consecutivas y estabiliza el entrenamiento.

9. Entrenamiento
El entrenamiento consiste en ajustar los pesos de la red neuronal para minimizar la diferencia entre los valores Q predichos y los valores Q objetivo. Esto se hace utilizando un minibatch de experiencias almacenadas en la memoria.

El agente se entrena durante 5000 episodios en el entorno de Blackjack.
En cada episodio:
  El agente interactúa con el entorno, almacenando experiencias en la memoria.
  Después de cada episodio, se entrena la red neuronal utilizando la memoria de repetición.

10. Evaluación
Después del entrenamiento, el agente se evalúa en 1000 episodios de prueba. Se registran las victorias, derrotas y empates, y se calcula la  tasa de victorias.
Ajustar los pesos de la red mediante retropropagación.

## Arquitectura de la red
Red neuronal: DuelingDQN
Esta red se usa para estimar los valores de Q, asigna un valor a cada par, representando la recompensa esperada al tomar una acción en un estado dado.
La red neuronal está definida en la clase DQN, que hereda de nn.Module (la clase base de PyTorch para redes neuronales).

  * Capa densa (fc1): x neuronas <- Esta capa es una capa lineal que transforma el estado de entrada en un espacio de x dimensiones.
  * Capa densa (fc2): x neuronas <- Esta capa lineal transporta las x dimensiones en otro espacio de x dimensiones.
  * Capa densa (fc3): x neuronas <- Esta capa produce los valores Q para cada acción posible.

Función de activación ReLU
La red utiliza la función de activación ReLU (Rectified Linear Unit) después de las dos primeras capas lineales. ReLU se define como:
ReLU(x)=max(0,x)
ReLU introduce no linealidad en la red, permitiendo que el modelo aprenda patrones complejos en los datos.
Además, ReLU es computacionalmente eficiente y ayuda a evitar el problema del gradiente vanishing.

Agente de Deep-Q-Learning
El agente DQLAgent utiliza la red neuronal DQN para aprender la política óptima. A continuación, se describen los componentes clave del agente:
Hiperparámetros
* gamma: Factor de descuento (0.99). Controla la importancia de las recompensas futuras.
* epsilon: Tasa de exploración (1.0 inicialmente). Controla la probabilidad de que el agente elija una acción aleatoria.
* epsilon_decay: Tasa de decaimiento de epsilon (0.999). Reduce epsilon con el tiempo para equilibrar exploración y explotación.
* epsilon_min: Valor mínimo de epsilon (0.01).
* learning_rate: Tasa de aprendizaje (0.0015). Controla la magnitud de los ajustes en los pesos de la red.
* batch_size: Tamaño del minibatch (32). Número de experiencias utilizadas para entrenar la red en cada paso.



## Optimización
Con DuelingDQN se separa la estimación del valor del estado  y la ventaja de cada acción. 
Prioritized Experience Replay se encarga de priorizar las experiencias con mayor error TD.

## Resultados
El código incluye gráficos para visualizar:
Tasa de victorias a lo largo del tiempo.
Distribución final de resultados (victorias, derrotas, empates).

## Conclusiones
Este proyecto demuestra cómo una red neuronal simple puede utilizarse para aproximar la función Q en un entorno de Blackjack. La combinación de Deep Q-Learning y redes neuronales permite que el agente aprenda una política óptima mediante la exploración y explotación del entorno.

Podemos ajustar la estructura de recompensas para seguir estrategias óptimas en el BlackJack, incluir estadísticas y encontrar mejores parámetros en las redes neuronales para lograr un mejor aprendizaje y pueda ganar más veces de las que pierde.

