import torch
import pandas as pd
from transformers import pipeline
from huggingface_hub import login
import re
login(token="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")
csv_files = [
    "/content/JMMLU/JMMLU/professional_medicine.csv",
    "/content/JMMLU/JMMLU/world_history.csv",
    "/content/JMMLU/JMMLU/college_computer_science.csv",
    "/content/JMMLU/JMMLU/management.csv"
]
total_questions = 0
correct_answers = 0
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-jpn-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",  # Cambiar a "mps" si usas un Mac con soporte MPS
)
# Cargar el archivo CSV y mostrar las columnas

def extract_option(generated_answer):
    # Buscar una opción entre A, B, C o D con regex
    match = re.search(r'\b([A-D]):', generated_answer)
    if match:
        return match.group(1)  # Retorna la letra encontrada (A, B, C o D)
    return None  # Si no encuentra una opción válida

def ask_question(row, dataset_name):
    global total_questions, correct_answers

    # Usamos índices para acceder a las columnas
    question = row.iloc[0]
    choice_a = row.iloc[1]
    choice_b = row.iloc[2]
    choice_c = row.iloc[3]
    choice_d = row.iloc[4]
    correct_answer = row.iloc[5]  # Respuesta correcta (A, B, C o D)

    # Crear el prompt con las opciones
    choices = f"A: {choice_a}, B: {choice_b}, C: {choice_c}, D: {choice_d}"
    prompt = f"[{dataset_name}] Pregunta: {question}\nOpciones: {choices}\n¿Qué opción es la correcta?"

    # Generar respuesta con Gemini 2
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(messages, return_full_text=False, max_new_tokens=256)
    generated_answer = outputs[0]["generated_text"].strip()

    # Extraer la opción seleccionada
    selected_option = extract_option(generated_answer)

    # Imprimir los resultados
    print(f"Dataset: {dataset_name}")
    print(f"Pregunta: {question}")
    print(f"Opciones: {choices}")
    #print(f"Respuesta generada: {generated_answer}")
    print(f"Respuesta seleccionada: {selected_option}")
    print(f"Respuesta correcta: {correct_answer}\n")

    # Comparar la opción seleccionada con la respuesta correcta
    total_questions += 1
    if selected_option == correct_answer:
        correct_answers += 1


# Paso 6: Iterar sobre los archivos y realizar el benchmark
for file_path in csv_files:
    dataset_name = file_path.split("/")[-1].replace(".csv", "")
    print(f"\nEvaluando el dataset: {dataset_name}\n{'-'*50}")

    # Cargar cada dataset y realizar preguntas
    data = pd.read_csv(file_path)
    for idx, row in data.iterrows():
        #if idx >= 5:  # Limitar a 5 preguntas por dataset para pruebas rápidas
        #    break
        ask_question(row, dataset_name)

# Paso 5: Calcular y mostrar el porcentaje de acierto
if total_questions > 0:
    accuracy = (correct_answers / total_questions) * 100
    print(f"\nPorcentaje de acierto: {accuracy:.2f}%")
else:
    print("No se realizaron preguntas.")