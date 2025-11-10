#!/bin/bash

echo "========================================="
echo "  ActionMiner Lite - Iniciando App"
echo "========================================="
echo ""

# Verificar que existe el modelo
if [ ! -d "models/best_baseline" ]; then
    echo "❌ ERROR: No se encontró el modelo entrenado"
    echo "Por favor, ejecuta primero:"
    echo "  python src/experiments/exp01_embeddings_logreg.py"
    exit 1
fi

echo "✓ Modelo encontrado"
echo ""

# Verificar dependencias
if ! python -c "import streamlit" 2>/dev/null; then
    echo "❌ ERROR: Streamlit no está instalado"
    echo "Instala las dependencias con:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "✓ Dependencias verificadas"
echo ""

echo "🚀 Iniciando aplicación Streamlit..."
echo ""
echo "La aplicación se abrirá en tu navegador en:"
echo "  👉 http://localhost:8501"
echo ""
echo "Presiona Ctrl+C para detener la aplicación"
echo ""
echo "========================================="
echo ""

streamlit run app/streamlit_app.py
