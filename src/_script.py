import os

# Mude para o diretório onde o script está
src_dir = os.path.dirname(os.path.abspath(__file__))  # Obtém o diretório do _script.py
os.chdir(src_dir)  # Muda o diretório de trabalho para 'src'

# Executa o script e salva toda a saída em um arquivo txt
os.system("python Test_FairnessGroup_MovieLens_Gender_Novelty-1000.py > Test_FairnessGroup_MovieLens_Gender_Novelty-1000-01.txt 2>&1")