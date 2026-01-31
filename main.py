#!/usr/bin/env python3
"""
Sistema de Question Answering - Implementação Modular
Autor: Seu Nome
Data: 2024
"""

import argparse
import sys
import os
from datetime import datetime

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import Config
from data.dataloader import DataLoader
from models import DistilBERTModel, RoBERTaModel
from utils.metrics import MetricsCalculator
from utils.helpers import HelperFunctions

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Sistema de Question Answering - Análise Comparativa',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py --interval 55 --models both --output-dir ./results
  python main.py --interval shard_055.csv --models distilbert --max-examples 500
  python main.py --list-intervals
        """
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        required=False,
        help='Número do intervalo ou nome do arquivo CSV (ex: 55 ou shard_055.csv)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        choices=['distilbert', 'roberta', 'both'],
        default='both',
        help='Modelos a serem avaliados (default: both)'
    )
    
    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help='Número máximo de exemplos a processar (default: todos)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Diretório para salvar resultados (default: ./output)'
    )
    
    parser.add_argument(
        '--list-intervals',
        action='store_true',
        help='Listar intervalos disponíveis e sair'
    )
    
    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Não exportar resultados para CSV'
    )
    
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Não criar visualizações'
    )
    
    return parser.parse_args()

def main():
    """Função principal"""
    print("=" * 70)
    print("🚀 SISTEMA DE QUESTION ANSWERING - ANÁLISE COMPARATIVA")
    print("=" * 70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup configurações
    Config.setup_dirs()
    
    # Inicializar carregador de dados
    dataloader = DataLoader()
    
    # Listar intervalos se solicitado
    if args.list_intervals:
        print("\n📁 INTERVALOS DISPONÍVEIS:")
        print("-" * 50)
        intervals = dataloader.list_intervals()
        for i, interval in enumerate(intervals, 1):
            print(f"{i:3d}. {interval}")
        print(f"\nTotal: {len(intervals)} intervalos")
        sys.exit(0)
    
    # Verificar se intervalo foi especificado
    if not args.interval:
        print("\n❌ ERRO: É necessário especificar um intervalo com --interval")
        print("   Use --list-intervals para ver intervalos disponíveis")
        sys.exit(1)
    
    # Carregar intervalo
    try:
        print(f"\n📂 Carregando intervalo: {args.interval}")
        df = dataloader.load_interval(args.interval)
        
        # Limitar número de exemplos se especificado
        if args.max_examples and args.max_examples < len(df):
            print(f"   Limitando a {args.max_examples} exemplos...")
            df = df.head(args.max_examples)
        
        print(f"   Total de exemplos: {len(df)}")
        print(f"   Colunas disponíveis: {