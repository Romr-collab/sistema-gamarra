import sys
import collections
# Parche para simular cv2 en servidores sin gráficos
try:
    import cv2
except ImportError:
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from ultralytics import YOLO
import numpy as np
from scipy.spatial.distance import cdist
import os

# Configuración de la página web
st.set_page_config(page_title="Monitoreo Gamarra", layout="wide")

# Cabecera de la App
st.markdown("""
    <div style="background-color: #1E3A8A; padding: 20px; border-radius: 10px; text-align: center; color: white; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 28px;">RECOLECCIÓN INTELIGENTE GAMARRA</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 14px;">Plataforma de Optimización Logística en Tiempo Real • Municipalidad de La Victoria</p>
    </div>
""", unsafe_allow_html=True)

# Enlace de tu Google Sheets (Reemplázalo por el tuyo real)
URL_SHEETS = "https://docs.google.com/spreadsheets/d/1D-Rsf0oXo7DqGn1KnnbMqqTGI2xnFG9jEFfm0jVBVbI/export?format=csv"

@st.cache_data(ttl=60) # Se actualiza automáticamente cada 60 segundos si hay cambios
def cargar_datos():
    return pd.read_csv(URL_SHEETS)

try:
    df_dron = cargar_datos()
    
    # Matriz fija de validación (Tus 6 fotos limpias reales)
    FOTOS_LIMPIAS = ["foto3.jpg", "foto4.jpg", "foto6.jpg", "foto9.jpg", "foto12.jpg", "foto15.jpg"]
    
    puntos_criticos = []
    
    # Procesamos los datos simulando el escaneo de IA
    for index, fila in df_dron.iterrows():
        nombre_foto = fila['nombre_archivo'].strip()
        if nombre_foto not in FOTOS_LIMPIAS:
            puntos_criticos.append({
                'Punto': f"Contenedor {len(puntos_criticos)+1}",
                'latitud': fila['latitud'],
                'longitud': fila['longitud'],
                'Foto Origen': nombre_foto
            })
            
    df_puntos = pd.DataFrame(puntos_criticos)
    
    # --- MOSTRAR KPIS EN LA APP ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🚨 Puntos Críticos Detectados", value=len(df_puntos))
    with col2:
        st.metric(label="🟢 Estado del Sistema", value="ACTIVO / AUTOMÁTICO")
    with col3:
        st.metric(label="🚛 Unidad Asignada", value="Camión Victoria-1")
        
    st.markdown("---")
    
    # --- ALGORITMO DE OPTIMIZACIÓN (TSP) ---
    lat_base = -12.065200  
    lon_base = -77.030300  
    punto_inicio = (lat_base, lon_base)
    
    coordenadas_basura = df_puntos[['latitud', 'longitud']].values
    coordenadas_totales = np.vstack([punto_inicio, coordenadas_basura])
    
    visitados = [0]
    no_visitados = list(range(1, len(coordenadas_totales)))
    
    punto_actual = 0
    while no_visitados:
        distancias = cdist([coordenadas_totales[punto_actual]], coordenadas_totales[no_visitados])[0]
        mas_cercano = no_visitados[np.argmin(distancias)]
        visitados.append(mas_cercano)
        no_visitados.remove(mas_cercano)
        punto_actual = mas_cercano
    visitados.append(0)
    
    # --- DISEÑO DE LA INTERFAZ: MAPA Y TABLA LADO A LADO ---
    col_mapa, col_tabla = st.columns([2, 1])
    
    with col_mapa:
        st.subheader("🗺️ Ruta de Recolección Optimizada")
        mapa = folium.Map(location=[-12.068, -77.022], zoom_start=15)
        
        # Base Verde
        folium.Marker(punto_inicio, popup="COCHERA MUNICIPAL", icon=folium.Icon(color='green', icon='home')).add_to(mapa)
        
        orden_ruta = [punto_inicio]
        # Puntos Rojos
        for i, idx_nodo in enumerate(visitados[1:-1]):
            datos_p = df_puntos.iloc[idx_nodo - 1]
            orden_ruta.append([datos_p['latitud'], datos_p['longitud']])
            folium.Marker(
                [datos_p['latitud'], datos_p['longitud']], 
                popup=f"Parada {i+1}", 
                icon=folium.Icon(color='red', icon='trash')
            ).add_to(mapa)
            
        orden_ruta.append(punto_inicio)
        folium.PolyLine(orden_ruta, color="blue", weight=4, opacity=0.8).add_to(mapa)
        
        # Desplegar mapa en la web
        st_folium(mapa, width=700, height=450, returned_objects=[])
        
    with col_tabla:
        st.subheader("📋 Hoja de Ruta para el Chofer")
        
        # Construir la tabla ordenada paso a paso
        hoja_ruta_datos = []
        hoja_ruta_datos.append({"Orden": "🛫 SALIDA", "Ubicación / Coordenadas": "Av. Iquitos Cdra. 4 (Cochera)"})
        
        for i, idx_nodo in enumerate(visitados[1:-1]):
            datos_p = df_puntos.iloc[idx_nodo - 1]
            hoja_ruta_datos.append({
                "Orden": f"📍 Parada {i+1}",
                "Ubicación / Coordenadas": f"{datos_p['latitud']}, {datos_p['longitud']}"
            })
            
        hoja_ruta_datos.append({"Orden": "🏁 RETORNO", "Ubicación / Coordenadas": "Av. Iquitos Cdra. 4 (Cochera)"})
        
        df_hoja = pd.DataFrame(hoja_ruta_datos)
        st.dataframe(df_hoja, use_container_width=True, hide_index=True)
        
        if st.button("🔄 Forzar Actualización de Datos"):
            st.rerun()

except Exception as e:
    st.error(f"Falta configurar el enlace de Google Sheets o subir los archivos. Error: {e}")
