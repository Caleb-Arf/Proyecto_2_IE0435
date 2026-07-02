# =============================
# Librerías
# =============================
import pandas as pd
import matplotlib.pyplot as plt
import ast, io, contextlib, re, json, subprocess
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import ollama
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import new_features
plt.ioff()


def show_plots_and_wait():
    """
    Muestra las figuras de Matplotlib y bloquea el agente hasta que el usuario
    cierre todas las ventanas abiertas.
    """
    if plt.get_fignums():
        print("🖼️  Cierra la(s) figura(s) para continuar...")
        plt.show(block=True)
        plt.close('all')

# =============================
# Configuración del modelo
# =============================
llm = "qwen2.5:7b"
subprocess.run(["ollama", "pull", llm], check=False)
print(f"🚀 Modelo cargado: {llm}")

# =============================
# Carga directa del dataset
# =============================
DATA_PATH = "datos_limpios.csv"

try:
    dtf = pd.read_csv(DATA_PATH)
    datetime_col = next((col for col in dtf.columns if any(x in col.lower() for x in ["fecha", "date", "hora"])), None)
    if datetime_col is None:
        raise ValueError("No se encontró columna de fecha/hora.")

    dtf[datetime_col] = pd.to_datetime(dtf[datetime_col], errors="coerce")
    dtf.set_index(datetime_col, inplace=True)
    dtf.sort_index(inplace=True)
    dtf = dtf.asfreq("15T")  # frecuencia fija de 15 minutos

    print(f"✅ Dataset '{DATA_PATH}' cargado correctamente ({len(dtf)} filas)")
    print(f"   Índice temporal: {datetime_col} (frecuencia 15 min)")
    print(f"   Columnas disponibles: {list(dtf.columns)}\n")
    print("📋 Primeras filas del dataset:")
    print(dtf.head(), "\n")
    new_features.dtf = dtf

except Exception as e:
    print(f"❌ Error al cargar {DATA_PATH}: {e}")
    exit()

# =============================
# Herramientas
# =============================

def final_answer(text: str) -> str:
    return text


tool_final_answer = {
  'type': 'function',
  'function': {
    'name': 'final_answer',
    'description': 'Devuelve una respuesta en lenguaje natural al usuario',
    'parameters': {
      'type': 'object',
      'required': ['text'],
      'properties': {
        'text': {'type':'string', 'description':'respuesta en lenguaje natural'}
      }
    }
  }
}

def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def sanitize_code(code: str) -> str:
    code = code.replace("df[", "dtf[")
    if code.count("(") > code.count(")"):
        code += ")" * (code.count("(") - code.count(")"))
    if code.count("[") > code.count("]"):
        code += "]" * (code.count("[") - code.count("]"))
    if code.count('"') % 2 != 0:
        code += '"'
    if code.count("'") % 2 != 0:
        code += "'"
    return code

def code_exec(code: str) -> str:
    output = io.StringIO()
    code = sanitize_code(code.strip())
    if not is_valid_python(code):
        return "Error: código incompleto o inválido."
    with contextlib.redirect_stdout(output):
        try:
            tree = ast.parse(code, mode="exec")
            if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                expr = ast.Expression(tree.body[0].value)
                result = eval(compile(expr, "<code_exec>", "eval"), globals())
                if result is not None:
                    print(result)
            else:
                exec(code, globals())
        except Exception as e:
            print(f"Error: {e}")
    return output.getvalue()



tool_code_exec = {
  'type':'function',
  'function':{
    'name': 'code_exec',
    'description': 'Ejecuta código Python. Puede usar print(...) o una expresión simple como get_statistics(dtf["MW"]).',
    'parameters': {
      'type': 'object', 
      'required': ['code'],
      'properties': {
        'code': {'type':'string', 'description':'código Python a ejecutar'},
      }
    }
  }
}



def normalize_plot_args(t_inputs):
    """
    Normaliza los argumentos para plot_data y corrige errores comunes del modelo.
    """
    if not isinstance(t_inputs, dict):
        return t_inputs
    
    # Asegurar que "columns" sea lista
    if "columns" in t_inputs:
        if isinstance(t_inputs["columns"], str):
            try:
                # Convierte "['MW']" → ["MW"]
                t_inputs["columns"] = json.loads(t_inputs["columns"].replace("'", '"'))
            except Exception:
                # Si falla, convierte a lista simple
                t_inputs["columns"] = [t_inputs["columns"]]
        elif t_inputs["columns"] is None:
            t_inputs["columns"] = []
    
    return t_inputs

def plot_data(columns=None, start_date=None, end_date=None, title="Gráfico de datos"):
    try:
        df = dtf.copy()
        if columns:
            cols_validas = [c for c in columns if c in df.columns]
            df = df[cols_validas]
        if start_date and end_date:
            df = df.loc[start_date:end_date]
        elif start_date:
            df = df.loc[start_date]
        df.plot(figsize=(12, 5), linestyle="--")
        plt.title(title)
        plt.xlabel("Tiempo (15 min)")
        plt.ylabel("Potencia [MW]")
        plt.grid(True)
        show_plots_and_wait()
        return f"Gráfico generado con columnas {columns or list(df.columns)}."
    except Exception as e:
        return f"Error al graficar: {e}"


tool_plot_data = {
  'type': 'function',
  'function': {
    'name': 'plot_data',
    'description': 'Genera gráficos de una o varias columnas del dataset ya cargado, opcionalmente filtrando por fecha o rango de fechas.',
    'parameters': {
      'type': 'object',
      'properties': {
        'columns': {
          'type': 'array',
          'items': {'type': 'string'},
          'description': 'Lista de columnas a graficar, por ejemplo ["MW"] o ["MW", "MW_P"].'
        },
        'start_date': {
          'type': 'string',
          'description': 'Fecha inicial o fecha única en formato YYYY-MM-DD.'
        },
        'end_date': {
          'type': 'string',
          'description': 'Fecha final en formato YYYY-MM-DD.'
        },
        'title': {
          'type': 'string',
          'description': 'Título del gráfico.'
        }
      }
    }
  }
}



def predict_data(model="prophet", column=None, horizon=None, end_date=None):
    """
    Genera predicciones de series de tiempo usando Prophet o SARIMA.
    - model: "prophet" o "arima"
    - column: nombre de la columna a predecir (ej. "MW")
    - horizon: número de días o texto (ej. "2 días", "10 days")
    - end_date: fecha final (YYYY-MM-DD)
    El modelo ahora predice exactamente lo que el usuario pida, sin límite artificial.
    """
    try:
        df = dtf.copy()
        if column is None or column not in df.columns:
            return f"Error: debes especificar una columna válida. Columnas disponibles: {list(df.columns)}"

        # Datos base
        df = df[[column]].dropna().reset_index()
        df.columns = ["ds", "y"]
        freq = "15min"
        last_date = df["ds"].max()

        # -------------------------------
        # Determinar horizonte de predicción (pasos)
        # -------------------------------
        steps = 96  # default 1 día
        if horizon:
            if isinstance(horizon, str):
                nums = re.findall(r"\d+", horizon)
                days = int(nums[0]) if nums else 1
                steps = max(1, days * 96)
            elif isinstance(horizon, int):
                # Interpretamos como días
                steps = max(1, horizon * 96)
        if end_date and (not horizon or isinstance(horizon, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", horizon)):
            try:
                target_date = pd.to_datetime(end_date)
                delta = target_date - last_date
                if delta.total_seconds() <= 0:
                    return f"La fecha {end_date} ya está incluida en los datos."
                steps = max(1, int(delta.total_seconds() / (15 * 60)))
            except Exception as e:
                return f"Error interpretando end_date: {e}"

        # -------------------------------
        # Modelos
        # -------------------------------
        model = model.lower()

        if model == "prophet":
            m = Prophet(daily_seasonality=True)
            m.fit(df)

            future = m.make_future_dataframe(periods=steps, freq=freq)
            forecast = m.predict(future)
            forecast_pred = forecast[forecast["ds"] > last_date][["ds", "yhat"]]
            forecast_pred = forecast_pred.rename(columns={"ds": "fecha", "yhat": "MW_pred"})

            # Gráfico solo de predicciones
            plt.figure(figsize=(10, 5))
            plt.plot(forecast_pred["fecha"], forecast_pred["MW_pred"], "o-", label="Predicción (Prophet)")
            plt.title(f"Predicción Prophet para {column} ({steps} pasos, {steps//96} día(s))")
            plt.xlabel("Fecha")
            plt.ylabel(column)
            plt.grid(True)
            plt.legend()
            show_plots_and_wait()

            out_df = forecast_pred.copy()

        elif model == "arima":
            # Reducimos a últimos ~30 días si hay demasiados datos (memoria)
            df_use = df.tail(96 * 30) if len(df) > 10000 else df.copy()
            df_use.set_index("ds", inplace=True)

            sarimax = SARIMAX(df_use["y"], order=(2, 1, 2), seasonal_order=(1, 0, 1, 96)).fit(disp=False)

            future_dates = pd.date_range(last_date, periods=steps + 1, freq=freq)[1:]
            forecast_vals = sarimax.forecast(steps=steps)
            forecast_df = pd.DataFrame({"fecha": future_dates, "MW_pred": forecast_vals})

            # Gráfico
            plt.figure(figsize=(10, 5))
            plt.plot(forecast_df["fecha"], forecast_df["MW_pred"], "o-", label="Predicción (SARIMA)")
            plt.title(f"Predicción SARIMA para {column} ({steps} pasos, {steps//96} día(s))")
            plt.xlabel("Fecha")
            plt.ylabel(column)
            plt.grid(True)
            plt.legend()
            show_plots_and_wait()

            out_df = forecast_df.copy()

        else:
            return "Error: modelo no reconocido. Usa 'prophet' o 'arima'."

        # -------------------------------
        # Guardado automático a CSV
        # -------------------------------
        os.makedirs(save_dir, exist_ok=True)
        # Rango para nombre de archivo
        start_str = pd.to_datetime(out_df["fecha"].min()).strftime("%Y%m%d_%H%M")
        end_str   = pd.to_datetime(out_df["fecha"].max()).strftime("%Y%m%d_%H%M")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        filename = f"pred_{model}_{column}_{start_str}_to_{end_str}_{timestamp}.csv"
        out_path = os.path.join(save_dir, filename)

        # Guardar solo dos columnas solicitadas
        out_df[["fecha", "MW_pred"]].to_csv(out_path, index=False)

        # Mostrar tabla (en consola) y confirmar ruta
        pd.set_option("display.max_rows", None)
        print("\n📈 Predicciones futuras:\n")
        print(out_df[["fecha", "MW_pred"]].to_string(index=False))
        print(f"\n💾 Guardado en: {out_path}")

        return f"Predicción {model.upper()} completada ({len(out_df)} puntos). Archivo: {out_path}"

    except Exception as e:
        return f"Error durante la predicción: {e}"





tool_predict_data = {
  'type': 'function',
  'function': {
    'name': 'predict_data',
    'description': 'Genera predicciones de series de tiempo con Prophet o ARIMA. Siempre grafica los valores futuros junto con los datos históricos y devuelve un resumen de los últimos valores predichos.',
    'parameters': {
      'type': 'object',
      'properties': {
        'model': {
          'type': 'string',
          'description': 'Modelo de predicción a usar: "prophet" o "arima".'
        },
        'column': {
          'type': 'string',
          'description': 'Nombre de la columna a predecir'
        },
        'horizon': {
          'type': 'integer',
          'description': 'Horizonte de predicción en días.'
        }
      },
      'required': ['model', 'column']
    }
  }
}

# =============================
# Diccionario de herramientas
# =============================
dic_tools = {
    "final_answer": final_answer,
    "code_exec": code_exec,
    "plot_data": plot_data,
    "predict_data": predict_data
}
dic_tools.update(new_features.new_tools)
globals().update(new_features.new_tools)

# =============================
# Ejecutor de herramientas
# =============================
def normalize_tool_args(t_name, t_inputs):
    """
    Normaliza argumentos de herramientas base y de new_features antes de ejecutar.
    """
    if t_name == "plot_data":
        return normalize_plot_args(t_inputs)
    if t_name == "code_exec":
        if isinstance(t_inputs, dict):
            return {"code": t_inputs.get("code", "")}
        return t_inputs
    if t_name == "predict_data":
        t_inputs = normalize_predict_args(t_inputs)
        if isinstance(t_inputs, dict):
            t_inputs.setdefault("model", "prophet")
            t_inputs.setdefault("horizon", 7)
            if "column" not in t_inputs or t_inputs["column"] not in dtf.columns:
                t_inputs["column"] = list(dtf.columns)[0]
        return t_inputs
    if t_name == "final_answer" and isinstance(t_inputs, dict) and "final_answer" in t_inputs:
        return {"text": t_inputs["final_answer"]}

    normalizer = new_features.normalizers.get(t_name)
    if normalizer:
        return normalizer(t_inputs)
    return t_inputs


def normalize_predict_args(t_inputs):
    """
    Corrige y normaliza argumentos comunes en predict_data.
    - Si 'horizon' parece una fecha, lo convierte a 'end_date'.
    - Si no se especifica horizon, usa 96 pasos (1 día de 15 min).
    - Convierte 'horizon': '1 day' → 96.
    - Limita horizon a 288 (máx. 3 días).
    """
    if not isinstance(t_inputs, dict):
        return t_inputs

    # Ignora campos inventados por el modelo
    for k in ["forecast_type", "prediction_type", "days_ahead"]:
        t_inputs.pop(k, None)

    # horizon recibido como fecha → end_date
    if "horizon" in t_inputs and isinstance(t_inputs["horizon"], str):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", t_inputs["horizon"]):
            t_inputs["end_date"] = t_inputs.pop("horizon")
        else:
            # "1 day", "2 días" → convertir a pasos de 96 por día
            num = re.findall(r"\d+", t_inputs["horizon"])
            t_inputs["horizon"] = int(num[0]) * 96 if num else 96

    # date → end_date
    if "date" in t_inputs and "end_date" not in t_inputs:
        t_inputs["end_date"] = t_inputs.pop("date")

    # Si no se especifica horizon, usar 96 pasos (1 día)
    if "horizon" not in t_inputs:
        t_inputs["horizon"] = 96

    # Si horizon es entero pero muy grande, limitar
    if isinstance(t_inputs.get("horizon"), int) and t_inputs["horizon"] > 288:
        print(f"⚠️  Horizon demasiado grande ({t_inputs['horizon']} pasos). Se limitará a 288.")
        t_inputs["horizon"] = 288

    # Normaliza modelo
    if "model" in t_inputs and isinstance(t_inputs["model"], str):
        t_inputs["model"] = t_inputs["model"].lower()

    return t_inputs


def get_requested_column(query):
    """
    Extrae una columna del texto del usuario. Si no encuentra una, usa MW
    cuando exista o la primera columna numérica disponible.
    """
    query_lower = query.lower()
    for col in dtf.columns:
        if col.lower() in query_lower:
            return col

    numeric_cols = list(dtf.select_dtypes(include="number").columns)
    if "MW" in dtf.columns:
        return "MW"
    if numeric_cols:
        return numeric_cols[0]
    return list(dtf.columns)[0]


def is_stat_plot_request(query):
    """
    Detecta peticiones que deben ir a stat_plotting, no a historical_hourly
    ni a get_statistics.
    """
    query_lower = query.lower()
    graph_terms = [
        "grafica", "gráfica", "grafico", "gráfico", "graficar",
        "grafique", "plot", "plotear", "plotea", "visualiza"
    ]
    stat_plot_terms = [
        "histograma", "boxplot", "caja", "distribucion", "distribución",
        "promedio horario", "perfil horario", "por hora", "hora pico",
        "promedio diario", "perfil diario", "estadistica", "estadística"
    ]

    return any(term in query_lower for term in graph_terms) and any(term in query_lower for term in stat_plot_terms)


def infer_stat_plot_args(query, t_inputs=None):
    """
    Convierte una petición en argumentos para stat_plotting.
    """
    t_inputs = dict(t_inputs or {}) if isinstance(t_inputs, dict) else {}
    query_lower = query.lower()

    if "column" not in t_inputs or t_inputs["column"] not in dtf.columns:
        t_inputs["column"] = get_requested_column(query)

    if "plot_type" not in t_inputs:
        if "histograma" in query_lower:
            t_inputs["plot_type"] = "histogram"
        elif "boxplot" in query_lower or "caja" in query_lower:
            t_inputs["plot_type"] = "boxplot"
        elif "promedio diario" in query_lower or "perfil diario" in query_lower:
            t_inputs["plot_type"] = "daily_profile"
        elif "promedio horario" in query_lower or "perfil horario" in query_lower or "por hora" in query_lower or "hora pico" in query_lower:
            t_inputs["plot_type"] = "hourly_profile"
        else:
            t_inputs["plot_type"] = "distribution"

    return new_features.normalize_stat_plotting_args(t_inputs)


def maybe_reroute_to_stat_plotting(t_name, t_inputs, user_query):
    """
    Si el modelo elige una herramienta de cálculo para una petición de gráfico
    estadístico, redirige la llamada a stat_plotting.
    """
    if user_query and is_stat_plot_request(user_query) and t_name in ("get_statistics", "historical_hourly", "final_answer"):
        return "stat_plotting", infer_stat_plot_args(user_query, t_inputs)
    return t_name, t_inputs




def use_tool(agent_res: dict, dic_tools: dict, user_query: str = "") -> dict:
    """
    Ejecuta las herramientas solicitadas por el modelo.
    - Soporta tanto tool_calls formales como JSON plano en content.
    - Incluye normalización automática de argumentos.
    """
    msg = agent_res["message"]
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls is None and hasattr(msg, "get"):
        tool_calls = msg.get("tool_calls")
    content = msg.get("content", "") if hasattr(msg, "get") else getattr(msg, "content", "")
    res, t_name, t_inputs = "", "", ""

    # ✅ Caso 1: tool_calls formales (cuando Ollama estructura la llamada)
    if tool_calls:
        for tool in tool_calls:
            t_name = tool["function"]["name"]
            raw_args = tool["function"]["arguments"]

            # Parsear argumentos
            try:
                t_inputs = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                t_inputs = raw_args

            t_inputs = normalize_tool_args(t_name, t_inputs)
            t_name, t_inputs = maybe_reroute_to_stat_plotting(t_name, t_inputs, user_query)

            # Ejecutar herramienta
            if f := dic_tools.get(t_name):
                print(f"🔧 > {t_name} -> Inputs: {t_inputs}")
                try:
                    t_output = f(**t_inputs) if isinstance(t_inputs, dict) else f(t_inputs)
                except Exception as e:
                    cols = list(dtf.columns) if 'dtf' in globals() else 'No hay dataset cargado'
                    t_output = f"Error ejecutando {t_name}: {e}. Columnas disponibles: {cols}"
                show_plots_and_wait()
                print(f"📊 Resultado:\n{t_output}\n")
                res = t_output
            else:
                print(f"🤬 > {t_name} -> NotFound")

    # ✅ Caso 2: JSON plano en msg.content (ej: {"name": ..., "arguments": {...}})
    elif content and content.strip().startswith("{"):
        try:
            tool_call = json.loads(content)
            t_name = tool_call.get("name", "")
            t_inputs = tool_call.get("arguments", {})

            t_inputs = normalize_tool_args(t_name, t_inputs)
            t_name, t_inputs = maybe_reroute_to_stat_plotting(t_name, t_inputs, user_query)

            if f := dic_tools.get(t_name):
                print(f"🔧 > {t_name} -> Inputs: {t_inputs}")
                res = f(**t_inputs) if isinstance(t_inputs, dict) else f(t_inputs)
                show_plots_and_wait()
                print(f"📊 Resultado:\n{res}\n")
            else:
                res = f"Herramienta {t_name} no encontrada."
        except Exception as e:
            res = f"⚠️ Error al interpretar JSON: {e}\nContenido: {content}"

    # ✅ Caso 3: mensaje normal (texto plano sin tool)
    elif content:
        res = content
        print(f"💬 {res}")

    return {"res": res, "tool_used": t_name, "inputs_used": t_inputs}



def run_agent(llm, messages, available_tools):
    tool_used, local_memory = '', ''
    used_compute = False

    while tool_used != 'final_answer':
        try:
            agent_res = ollama.chat(
                model=llm, 
                messages=messages, 
                #format="json", 
                tools=[v for v in available_tools.values()]
            )

            dic_res = use_tool(agent_res, dic_tools, user_query=messages[-1]["content"])
            res, tool_used, inputs_used = dic_res["res"], dic_res["tool_used"], dic_res["inputs_used"]

          
            if tool_used in ("code_exec", "plot_data", "get_statistics", "historical_hourly", "stat_plotting"):
                used_compute = True

            user_query = messages[-1]["content"].lower()
            needs_compute = any(word in user_query for word in [
                "promedio", "media", "máximo", "mínimo", "suma", "resta",
                "gráfico", "grafica", "plot", "visualiza", "filtra",
                "porcentaje", "calcula", "valor", "estadística", "histograma", "error", 
            ])

            if tool_used == "final_answer" and needs_compute and not used_compute:
                print("⚠️ > El modelo intentó responder sin calcular. Reintentando...")
                messages.append({
                    "role": "user", 
                    "content": "Debes usar code_exec, plot_data, get_statistics, historical_hourly o stat_plotting antes de final_answer."
                })
                tool_used = ""
                continue

        except Exception as e:
            print("⚠️ >", e)
            res = f"Intenté usar {tool_used} pero falló. Intentaré otra cosa."
            messages.append({"role": "assistant", "content": res})

        if tool_used not in ['', 'final_answer']:
            # Agregar al historial de memoria
            local_memory += f"\nTool used: {tool_used}.\nInput used: {inputs_used}.\nOutput: {res}"
            messages.append({"role": "assistant", "content": f"Resultado: {res}"})
            available_tools.pop(tool_used, None)
            if len(available_tools) == 1:
                messages.append({"role": "user", "content": "ahora activa la herramienta final_answer."})

        if tool_used == '':
            break

    return res

# =============================
# Bucle principal
# =============================
prompt = """
Eres un Analista de Datos especializado en series de tiempo eléctricas.

Contexto del dataset:
- El archivo `datos_limpios.csv` ya está cargado en memoria como `dtf`.
- Representa datos reales de consumo eléctrico con frecuencia de 15 minutos.
- Columnas disponibles:
  • `MW`: Potencia eléctrica medida (en megavatios).
  • `MW_P`: Potencia predicha (en megavatios).
- El índice temporal (`fechaHora`) está en formato datetime y define los intervalos de 15 minutos.

Tu objetivo:
Ayudar al usuario a analizar, visualizar y predecir los datos de `dtf`.

Herramientas disponibles:
1. **code_exec** → Ejecuta código Python con `print(...)` (por ejemplo, cálculos, estadísticas, correlaciones).
2. **plot_data** → Genera gráficos (por ejemplo, tendencias diarias o comparaciones MW vs MW_P).
3. **predict_data** → Predice valores futuros usando Prophet o ARIMA (por ejemplo, los próximos días de MW).
4. **get_statistics** → Calcula estadísticas descriptivas de `MW` o `MW_P`. No genera gráficos.
5. **historical_hourly** → Calcula el promedio histórico por hora del día e identifica horas pico. No genera gráficos.
6. **stat_plotting** → Genera solo gráficos estadísticos: distribución, histograma, boxplot, promedio horario o promedio diario.
7. **final_answer** → Explica resultados o responde en lenguaje natural.

Restricciones:
- No cargues ni reemplaces el dataset (ya está cargado como `dtf`).
- No inventes columnas, datos o archivos.
- Siempre usa los nombres reales de columnas.
- Usa comillas dobles en nombres de columnas, ej: `dtf["MW"]`.
- Nunca uses `pd.read_csv()` dentro de `code_exec`.
- Si el usuario pide calcular estadísticas o promedio horario sin pedir explícitamente gráficos, NO uses herramientas de gráficos.
- Si el usuario pide graficar estadísticas, perfiles horarios, histograma, boxplot o distribución, usa `stat_plotting`.
- Si el usuario pide graficar la serie temporal real por fecha/rango, usa `plot_data`.
- Si el usuario pide una fecha o rango que no existe, informa el error con `final_answer`.

Ejemplos de comportamiento esperado:
- “¿Cuál es el promedio de MW?” → usa `code_exec` con `print(dtf["MW"].mean())`.
- “Grafica el 2024-09-06” → usa `plot_data` con `columns=["MW"]`, `start_date="2024-09-06"`, `end_date="2024-09-06"`.
- “Predice MW para los próximos 2 días con Prophet” → usa `predict_data` con `{"model": "prophet", "column": "MW", "horizon": 2}`.
- “Dame estadísticas de MW” → usa `get_statistics` con `{"column": "MW"}`.
- “Calcula el promedio histórico por hora de MW” → usa `historical_hourly` con `{"column": "MW"}`.
- “Grafica el histograma de MW” → usa `stat_plotting` con `{"column": "MW", "plot_type": "histogram"}`.
- “Grafica el promedio histórico por hora de MW” → usa `stat_plotting` con `{"column": "MW", "plot_type": "hourly_profile"}`.
- “¿Qué columnas tiene el archivo?” → usa solo `final_answer` describiendo las columnas.

Tu prioridad es responder de forma útil, precisa y sin inventar resultados.
"""


messages = [{"role": "system", "content": prompt}]
print("💬 Agente centrado en 'datos_limpios.csv'. Escribe tus consultas o 'quit' para salir.\n")

while True:
    q = input("🙂 > ")
    if q.lower() == "quit":
        break
    messages.append({"role": "user", "content": q})
    available_tools = {
        "final_answer": tool_final_answer,
        "code_exec": tool_code_exec,
        "plot_data": tool_plot_data,
        "predict_data": tool_predict_data
    }
    for tool_schema in new_features.new_tools_schemas:
        available_tools[tool_schema["function"]["name"]] = tool_schema

    if is_stat_plot_request(q):
        t_inputs = infer_stat_plot_args(q)
        print(f"🔧 > stat_plotting -> Inputs: {t_inputs}")
        try:
            dic_res = {
                "res": dic_tools["stat_plotting"](**t_inputs),
                "tool_used": "stat_plotting",
                "inputs_used": t_inputs
            }
        except Exception as e:
            dic_res = {
                "res": f"Error ejecutando stat_plotting: {e}. Columnas disponibles: {list(dtf.columns)}",
                "tool_used": "stat_plotting",
                "inputs_used": t_inputs
            }
        show_plots_and_wait()
        print("👽 >", dic_res["res"], "\n")
        messages.append({"role": "assistant", "content": dic_res["res"]})
        continue

    res = ollama.chat(model=llm, messages=messages, tools=[v for v in available_tools.values()], format="json")
    dic_res = use_tool(res, dic_tools, user_query=q)
    print("👽 >", dic_res["res"], "\n")
    messages.append({"role": "assistant", "content": dic_res["res"]})
