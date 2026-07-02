"""
Módulo de nuevas características para el agente de análisis eléctrico.
Contiene herramientas adicionales para estadísticas, anomalías y predicciones.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
plt.ioff()


def show_plots_and_wait():
    """
    Muestra las figuras de Matplotlib y bloquea el agente hasta que el usuario
    cierre todas las ventanas abiertas.
    """
    if plt.get_fignums():
        print("Cierra la(s) figura(s) para continuar...")
        plt.show(block=True)
        plt.close('all')


# Herramienta 1: Estadísticas
def get_statistics(column=None, by_hour=False, by_day=False):
    """
    Obtiene estadísticas descriptivas completas de una columna.
    
    Args:
        column: Nombre de la columna (ej: "MW")
        by_hour: Agrupar estadísticas por hora del día
        by_day: Agrupar estadísticas por día

    Returns:
        DataFrame con estadísticas o mensaje de error
    """
    try:
        # Validar que dtf existe
        if 'dtf' not in globals():
            return "Error: El dataset 'dtf' no está cargado en memoria."

        if isinstance(column, pd.Series):
            column = column.name
        
        # Validar columna
        if column is None:
            numeric_cols = dtf.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) == 0:
                return "Error: No hay columnas numéricas en el dataset."
            column = numeric_cols[0]
            print(f" Usando columna numérica: '{column}'")
        
        if column not in dtf.columns:
            return f"Error: Columna '{column}' no existe. Columnas disponibles: {list(dtf.columns)}"
        
        # Datos limpios
        df = dtf[column].dropna()
        
        if len(df) == 0:
            return f"Error: La columna '{column}' no tiene datos válidos"
        
        # Estadísticas básicas
        stats = {
            'Media': df.mean(),
            'Mediana': df.median(),
            'Desviación Estándar': df.std(),
            'Varianza': df.var(),
            'Mínimo': df.min(),
            'Máximo': df.max(),
            'Rango': df.max() - df.min(),
            'Q1 (25%)': df.quantile(0.25),
            'Q3 (75%)': df.quantile(0.75),
            'IQR': df.quantile(0.75) - df.quantile(0.25),
            'Asimetría': df.skew(),
            'Curtosis': df.kurtosis(),
            'Total Datos': len(df),
            'Datos Faltantes': dtf[column].isna().sum(),
            'Porcentaje Faltantes': (dtf[column].isna().sum() / len(dtf)) * 100
        }
        
        # Convertir a DataFrame para mejor presentación
        stats_df = pd.DataFrame([stats], index=[column]).round(2)
        
        # Mostrar en consola con formato bonito
        print("\n" + "="*70)
        print(f"ESTADÍSTICAS DE '{column}'")
        print("="*70)
        print(stats_df.to_string())
        print("="*70 + "\n")
        
        # Estadísticas por hora
        if by_hour:
            print("\nESTADÍSTICAS POR HORA:")
            stats_hour = df.groupby(df.index.hour).agg(['mean', 'std', 'min', 'max', 'count'])
            print(stats_hour.round(2))
            print("\n")
        
        # Estadísticas por día
        if by_day:
            print("\nESTADÍSTICAS POR DÍA:")
            stats_day = df.groupby(df.index.date).agg(['mean', 'std', 'min', 'max', 'count'])
            print(stats_day.round(2))
            print("\n")
        # Retornar resumen
        return f"Estadísticas calculadas para '{column}'. Media: {stats['Media']:.2f}, Mediana: {stats['Mediana']:.2f}, Desv.Std: {stats['Desviación Estándar']:.2f}"
        
    except Exception as e:
        return f"Error calculando estadísticas: {e}"


tool_get_statistics = {
    'type': 'function',
    'function': {
        'name': 'get_statistics',
        'description': 'Calcula estadísticas descriptivas completas de una columna numérica: media, mediana, desviación estándar, cuartiles, asimetría, curtosis, etc. No genera gráficos.',
        'parameters': {
            'type': 'object',
            'properties': {
                'column': {
                    'type': 'string',
                    'description': 'Nombre de la columna a analizar (ej: "MW", "MW_P")'
                },
                'by_hour': {
                    'type': 'boolean',
                    'description': 'Si es True, muestra estadísticas agrupadas por hora del día'
                },
                'by_day': {
                    'type': 'boolean',
                    'description': 'Si es True, muestra estadísticas agrupadas por día'
                }
            },
            'required': ['column']
        }
    }
}




# Herramienta 4: Históricos por hora

def historical_hourly(column=None):
    """
    Calcula el promedio histórico por hora del día para una columna.
    
    Args:
        column: Columna a analizar

    Returns:
        DataFrame con promedio por hora
    """
    try:
        if 'dtf' not in globals():
            return "Error: El dataset 'dtf' no está cargado en memoria."

        if isinstance(column, pd.Series):
            column = column.name
        
        if column is None or column not in dtf.columns:
            return f"Error: Columna '{column}' no existe. Columnas disponibles: {list(dtf.columns)}"
        
        # Calcular promedio por hora
        hourly_mean = dtf.groupby(dtf.index.hour)[column].mean()
        hourly_std = dtf.groupby(dtf.index.hour)[column].std()
        hourly_min = dtf.groupby(dtf.index.hour)[column].min()
        hourly_max = dtf.groupby(dtf.index.hour)[column].max()
        
        # Crear DataFrame de resultados
        results = pd.DataFrame({
            'Hora': hourly_mean.index,
            'Media': hourly_mean.values,
            'Desv_Std': hourly_std.values,
            'Mínimo': hourly_min.values,
            'Máximo': hourly_max.values
        })
        
        print("\nPERFIL HORARIO HISTÓRICO")
        print("="*60)
        print(results.round(2).to_string(index=False))
        print("="*60 + "\n")
        
        # Encontrar horas pico
        max_hour = results.loc[results['Media'].idxmax()]
        min_hour = results.loc[results['Media'].idxmin()]
        
        print(f"Hora de mayor demanda: {int(max_hour['Hora'])}:00 ({max_hour['Media']:.2f})")
        print(f"Hora de menor demanda: {int(min_hour['Hora'])}:00 ({min_hour['Media']:.2f})")
        return f"Perfil horario calculado para {column}. Hora pico: {int(max_hour['Hora'])}:00"
        
    except Exception as e:
        return f"Error calculando perfil horario: {e}"


tool_historical_hourly = {
    'type': 'function',
    'function': {
        'name': 'historical_hourly',
        'description': 'Calcula el promedio histórico por hora del día para una columna. Identifica horas pico y patrones diarios. No genera gráficos.',
        'parameters': {
            'type': 'object',
            'properties': {
                'column': {
                    'type': 'string',
                    'description': 'Columna a analizar (ej: "MW")'
                }
            },
            'required': ['column']
        }
    }
}


# Herramienta 3: Gráfico de Estadísticas

def stat_plotting(column=None, plot_type="distribution"):
    """
    Grafica estadísticas calculadas a partir del dataset.

    plot_type:
        - distribution: histograma y boxplot
        - histogram: solo histograma
        - boxplot: solo boxplot
        - hourly_profile: promedio histórico por hora
        - daily_profile: promedio histórico por día
    """
    try:
        if 'dtf' not in globals():
            return "Error: El dataset 'dtf' no está cargado en memoria."

        if column is None or column not in dtf.columns:
            return f"Error: Columna '{column}' no existe. Columnas disponibles: {list(dtf.columns)}"

        df = dtf[column].dropna()
        if len(df) == 0:
            return f"Error: La columna '{column}' no tiene datos válidos"

        plot_type = str(plot_type or "distribution").lower()

        if plot_type in ["distribution", "distribucion", "distribución"]:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            df.hist(bins=30, edgecolor='black', alpha=0.7)
            plt.title(f'Histograma de {column}')
            plt.xlabel(column)
            plt.ylabel('Frecuencia')
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            df.plot(kind='box')
            plt.title(f'Boxplot de {column}')
            plt.ylabel(column)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            message = f"Gráfico de distribución generado para {column}."

        elif plot_type in ["histogram", "histograma"]:
            plt.figure(figsize=(10, 5))
            df.hist(bins=30, edgecolor='black', alpha=0.7)
            plt.title(f'Histograma de {column}')
            plt.xlabel(column)
            plt.ylabel('Frecuencia')
            plt.grid(True, alpha=0.3)
            message = f"Histograma generado para {column}."

        elif plot_type == "boxplot":
            plt.figure(figsize=(7, 5))
            df.plot(kind='box')
            plt.title(f'Boxplot de {column}')
            plt.ylabel(column)
            plt.grid(True, alpha=0.3)
            message = f"Boxplot generado para {column}."

        elif plot_type in ["hourly_profile", "hourly", "promedio_horario", "perfil_horario"]:
            hourly_mean = dtf.groupby(dtf.index.hour)[column].mean()
            hourly_std = dtf.groupby(dtf.index.hour)[column].std()

            plt.figure(figsize=(12, 6))
            plt.plot(hourly_mean.index, hourly_mean.values, 'b-', linewidth=2, label='Media')
            plt.fill_between(
                hourly_mean.index,
                hourly_mean.values - hourly_std.values,
                hourly_mean.values + hourly_std.values,
                alpha=0.3,
                label='±1 Desv. Std.'
            )
            plt.title(f'Perfil Horario Histórico - {column}')
            plt.xlabel('Hora del día')
            plt.ylabel(column)
            plt.xticks(range(0, 24, 2))
            plt.legend()
            plt.grid(True, alpha=0.3)
            message = f"Gráfico de promedio horario generado para {column}."

        elif plot_type in ["daily_profile", "daily", "promedio_diario", "perfil_diario"]:
            daily_mean = dtf.groupby(dtf.index.date)[column].mean()

            plt.figure(figsize=(12, 5))
            plt.plot(pd.to_datetime(daily_mean.index), daily_mean.values, 'o-', linewidth=2)
            plt.title(f'Promedio Diario Histórico - {column}')
            plt.xlabel('Fecha')
            plt.ylabel(column)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            message = f"Gráfico de promedio diario generado para {column}."

        else:
            return "Error: plot_type no reconocido. Usa distribution, histogram, boxplot, hourly_profile o daily_profile."

        show_plots_and_wait()
        return message

    except Exception as e:
        return f"Error generando gráfico estadístico: {e}"


tool_stat_plotting = {
    'type': 'function',
    'function': {
        'name': 'stat_plotting',
        'description': 'Genera únicamente gráficos estadísticos: histograma, boxplot, distribución, promedio horario o promedio diario. Usar solo cuando el usuario pida graficar estadísticas; no reemplaza plot_data.',
        'parameters': {
            'type': 'object',
            'properties': {
                'column': {
                    'type': 'string',
                    'description': 'Columna numérica a graficar, por ejemplo "MW" o "MW_P".'
                },
                'plot_type': {
                    'type': 'string',
                    'description': 'Tipo de gráfico estadístico: distribution, histogram, boxplot, hourly_profile o daily_profile.'
                }
            },
            'required': ['column', 'plot_type']
        }
    }
}



# Normalizadores para entender texto

def normalize_stats_args(t_inputs):
    """Normaliza argumentos para get_statistics"""
    if not isinstance(t_inputs, dict):
        return t_inputs
    
    t_inputs.pop('show_plot', None)

    # Convertir strings a booleanos
    for key in ['by_hour', 'by_day']:
        if key in t_inputs:
            if isinstance(t_inputs[key], str):
                t_inputs[key] = t_inputs[key].lower() in ['true', '1', 'yes']
    
    return t_inputs


def normalize_historical_hourly_args(t_inputs):
    """Normaliza argumentos para historical_hourly"""
    if not isinstance(t_inputs, dict):
        return t_inputs

    t_inputs.pop('show_plot', None)
    return t_inputs


def normalize_stat_plotting_args(t_inputs):
    """Normaliza argumentos para stat_plotting"""
    if not isinstance(t_inputs, dict):
        return t_inputs

    aliases = {
        "distribucion": "distribution",
        "distribución": "distribution",
        "histograma": "histogram",
        "caja": "boxplot",
        "box": "boxplot",
        "promedio horario": "hourly_profile",
        "perfil horario": "hourly_profile",
        "horario": "hourly_profile",
        "promedio diario": "daily_profile",
        "perfil diario": "daily_profile",
        "diario": "daily_profile",
    }

    if 'plot_type' in t_inputs and isinstance(t_inputs['plot_type'], str):
        raw_plot_type = t_inputs['plot_type'].strip().lower()
        t_inputs['plot_type'] = aliases.get(raw_plot_type, raw_plot_type)

    return t_inputs


def normalize_anomaly_args(t_inputs):
    """Normaliza argumentos para detect_anomalies"""
    if not isinstance(t_inputs, dict):
        return t_inputs
    
    if 'threshold' in t_inputs and isinstance(t_inputs['threshold'], str):
        try:
            t_inputs['threshold'] = float(t_inputs['threshold'])
        except:
            t_inputs['threshold'] = 1.5
    
    if 'method' in t_inputs:
        t_inputs['method'] = t_inputs['method'].lower()
    
    return t_inputs


def normalize_ml_args(t_inputs):
    """Normaliza argumentos para predict_ml"""
    if not isinstance(t_inputs, dict):
        return t_inputs
    
    if 'model_type' in t_inputs:
        t_inputs['model_type'] = t_inputs['model_type'].lower()
    
    if 'test_size' in t_inputs and isinstance(t_inputs['test_size'], str):
        try:
            t_inputs['test_size'] = float(t_inputs['test_size'])
        except:
            t_inputs['test_size'] = 0.2
    
    return t_inputs


# Nuevos diccionarios de herramientas

# Este diccionario contiene todas las herramientas nuevas
new_tools = {
    "get_statistics": get_statistics,
    "historical_hourly": historical_hourly,
    "stat_plotting": stat_plotting
}

# Esquemas de las herramientas para el LLM
new_tools_schemas = [
    tool_get_statistics,
    tool_historical_hourly,
    tool_stat_plotting
]

# Normalizadores por herramienta
normalizers = {
    "get_statistics": normalize_stats_args,
    "historical_hourly": normalize_historical_hourly_args,
    "stat_plotting": normalize_stat_plotting_args
}
