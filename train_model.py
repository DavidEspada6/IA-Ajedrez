from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from chess_env import ChessEnv
import os

# Inicializar estadísticas
estadisticas = {
    "partidas": 0,
    "victorias_blancas": 0,
    "victorias_negras": 0,
    "tablas": 0,
    "movimientos_totales": 0
}

def main():
    # Envolver el entorno de ajedrez para que sea compatible con Stable Baselines3
    env = make_vec_env(lambda: ChessEnv(), n_envs=1)

    # Crear el modelo PPO
    model = PPO("MlpPolicy", env, verbose=1)

    # Entrenar el modelo
    model.learn(total_timesteps=50000)  # Puedes ajustar el número de pasos según sea necesario

    # Guardar el modelo entrenado
    model.save("ppo_chess_model")

    # Entrenamiento completado y modelo guardado
    model.save("ppo_chess_model")
    
    # Cargar el modelo entrenado para evaluación
    model = PPO.load("ppo_chess_model")
    
    # Llama a la función de evaluación con el modelo entrenado
    evaluar_modelo(model, num_partidas=100)  # Ajusta el número de partidas según necesites

    print("Entrenamiento completado y modelo guardado como ppo_chess_model.zip")

def evaluar_modelo(model, num_partidas=10):
    env = ChessEnv()  # Crea una instancia de tu entorno de ajedrez para la evaluación
    for partida in range(num_partidas):
        obs = env.reset()
        done = False
        movimientos = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            movimientos += 1
        env.render()  # Muestra el estado final del tablero
        resultado, movimientos_totales = env.get_game_result()
        registrar_partida(resultado, movimientos_totales)

    print(f"Estadísticas después de {num_partidas} partidas:")
    print(estadisticas)

def registrar_partida(resultado, movimientos):
    estadisticas["partidas"] += 1
    estadisticas["movimientos_totales"] += movimientos
    if resultado == "Blanco":
        estadisticas["victorias_blancas"] += 1
    elif resultado == "Negro":
        estadisticas["victorias_negras"] += 1
    else:
        estadisticas["tablas"] += 1

    with open("estadisticas_partidas.txt", "a") as f:
        f.write(f"Partida numero: {estadisticas['partidas']}\n")
        f.write(f"El ganador es: {resultado}\n")
        f.write(f"Los movimientos totales han sido: {movimientos}\n")
        f.write("------------------\n")

    print(f"Partida numero: {estadisticas['partidas']}")
    print(f"El ganador es: {resultado}")
    print(f"Los movimientos totales han sido: {movimientos}")
    print("------------------")

if __name__ == "__main__":
    main()
