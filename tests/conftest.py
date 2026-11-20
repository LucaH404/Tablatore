import sys
from pathlib import Path

# Rende importabile il pacchetto "app" dalla cartella del progetto
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
