"""
Command Line Interface
======================

CLI for hospital stock prediction using Prophet.

Usage:
    python -m src.cli predict --product "Poulet Frais" --days 30
    python -m src.cli analyze --dataset enriched
    python -m src.cli list-products
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

# Suppress Prophet/plotly warning
warnings.filterwarnings("ignore", message="Importing plotly failed")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    settings, 
    load_yaml_config, 
    get_dataset_path,
    create_results_directory,
    DATASETS,
    REGRESSORS
)
from src.data_loader import load_dataset, prepare_prophet_data, train_test_split
from src.metrics import calculate_metrics, print_metrics, interpret_mape
from src.model import train_prophet_model, predict, predict_future, check_prophet_available


def interactive_select(options: list, prompt: str = "Sélectionnez une option") -> str:
    """
    Interactive selection menu using keyboard.
    
    Args:
        options: List of (key, label, exists) tuples
        prompt: Prompt message
        
    Returns:
        Selected key
    """
    try:
        # Try to use keyboard navigation on Windows
        if sys.platform == 'win32':
            import msvcrt
            import os
            
            selected = 0
            
            def draw_menu():
                os.system('cls')
                print("=" * 70)
                print("📊 SÉLECTION DU DATASET - Clinique du Mont Vert")
                print("=" * 70)
                print(f"\n{prompt}:")
                print("-" * 50)
                
                for i, (key, label, exists) in enumerate(options):
                    status = "✅" if exists else "❌"
                    prefix = "→ " if i == selected else "  "
                    print(f"{prefix}{key:<12} {label:<30} {status}")
                
                print("-" * 50)
                print("↑↓: Naviguer | Entrée: Sélectionner | q: Quitter")
            
            draw_menu()
            
            while True:
                # Get key press
                key_press = msvcrt.getch()
                
                if key_press == b'\xe0':  # Arrow key prefix
                    arrow = msvcrt.getch()
                    if arrow == b'H':  # Up
                        selected = (selected - 1) % len(options)
                    elif arrow == b'P':  # Down
                        selected = (selected + 1) % len(options)
                    draw_menu()
                elif key_press == b'\r':  # Enter
                    os.system('cls')
                    return options[selected][0]
                elif key_press in (b'q', b'Q', b'\x1b'):  # q or Escape
                    os.system('cls')
                    return None
        else:
            # Fallback for non-Windows: simple numbered selection
            raise ImportError("Use fallback")
            
    except (ImportError, Exception):
        # Fallback: simple numbered menu
        print(f"\n{prompt}:")
        print("-" * 50)
        
        for i, (key, label, exists) in enumerate(options, 1):
            status = "✅" if exists else "❌"
            print(f"  {i}. {key:<12} {label:<30} {status}")
        
        print("-" * 50)
        
        while True:
            try:
                choice = input("Votre choix (numéro ou 'q' pour quitter): ").strip()
                if choice.lower() == 'q':
                    return None
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx][0]
                print("❌ Choix invalide")
            except ValueError:
                # Maybe they typed the key directly
                for key, _, _ in options:
                    if choice.lower() == key.lower():
                        return key
                print("❌ Entrez un numéro valide")


def select_dataset_interactive() -> str:
    """
    Interactive dataset selection.
    
    Returns:
        Selected dataset key or None if cancelled
    """
    options = []
    for key, filename in DATASETS.items():
        path = get_dataset_path(filename)
        exists = path.exists()
        
        # Add description
        if key == "base":
            label = "Dataset de base (51k lignes)"
        elif key == "realistic":
            label = "Dataset réaliste FIFO (24k)"
        elif key == "enriched":
            label = "Dataset enrichi ⭐ (85k)"
        else:
            label = filename
            
        options.append((key, label, exists))
    
    return interactive_select(options, "Choisissez un dataset")


def select_product_interactive(dataset_key: str = "enriched") -> str:
    """
    Interactive product selection from dataset.
    
    Args:
        dataset_key: Dataset to load products from
        
    Returns:
        Selected product name or None if cancelled
    """
    # Load dataset to get products
    if dataset_key in DATASETS:
        dataset_file = DATASETS[dataset_key]
    else:
        dataset_file = dataset_key
    
    dataset_path = get_dataset_path(dataset_file)
    
    if not dataset_path.exists():
        print(f"❌ Dataset non trouvé: {dataset_path}")
        return None
    
    df = load_dataset(str(dataset_path))
    
    # Detect product column
    if 'nom_produit' in df.columns:
        product_col = 'nom_produit'
    elif 'produit' in df.columns:
        product_col = 'produit'
    else:
        print("❌ Aucune colonne de produit trouvée")
        return None
    
    # Get unique products sorted
    products = sorted(df[product_col].unique())
    
    # Build options with line count
    options = []
    for product in products:
        count = len(df[df[product_col] == product])
        label = f"{count:,} lignes"
        options.append((product, label, True))
    
    return interactive_select(options, "Choisissez un produit")


def cmd_predict(args):
    """
    Run prediction for a specific product.
    
    Args:
        args: Parsed command line arguments
    """
    if not check_prophet_available():
        print("=" * 70)
        print("🏥 PRÉDICTION DE STOCK - Clinique du Mont Vert")
        print("=" * 70)
        print("❌ Erreur: Prophet n'est pas installé")
        print("   Installez-le avec: pip install prophet")
        return 1
    
    # Get dataset (interactive if not specified)
    dataset_key = args.dataset
    if not dataset_key:
        dataset_key = select_dataset_interactive()
        if dataset_key is None:
            print("\n❌ Sélection annulée.")
            return 1
    
    # Get product (interactive if not specified)
    product_name = args.product
    if not product_name:
        product_name = select_product_interactive(dataset_key)
        if product_name is None:
            print("\n❌ Sélection annulée.")
            return 1
    
    # Get days (interactive if not specified)
    days = args.days
    if days is None:
        print("=" * 70)
        print("📅 HORIZON DE PRÉDICTION")
        print("=" * 70)
        while True:
            try:
                days_input = input("\nNombre de jours à prédire (1-365): ").strip()
                if days_input.lower() == 'q':
                    print("\n❌ Sélection annulée.")
                    return 1
                days = int(days_input)
                if 1 <= days <= 365:
                    break
                print("❌ Veuillez entrer un nombre entre 1 et 365")
            except ValueError:
                print("❌ Veuillez entrer un nombre valide")
    
    print("=" * 70)
    print("🏥 PRÉDICTION DE STOCK - Clinique du Mont Vert")
    print("=" * 70)
    print(f"\n📦 Produit: {product_name}")
    print(f"📅 Horizon: {days} jours")
    print(f"📊 Dataset: {dataset_key}")
    
    # Get dataset path
    if dataset_key in DATASETS:
        dataset_file = DATASETS[dataset_key]
    else:
        dataset_file = dataset_key
    
    dataset_path = get_dataset_path(dataset_file)
    
    if not dataset_path.exists():
        print(f"\n❌ Erreur: Dataset non trouvé: {dataset_path}")
        return 1
    
    # Load and prepare data
    print("\n📥 Chargement des données...")
    df = load_dataset(str(dataset_path))
    
    # Detect target column (quantite_consommee or quantite)
    if 'quantite_consommee' in df.columns:
        target_col = 'quantite_consommee'
    elif 'quantite' in df.columns:
        target_col = 'quantite'
    else:
        print("❌ Erreur: Aucune colonne de quantité trouvée")
        return 1
    
    # Detect product column
    if 'nom_produit' in df.columns:
        product_col = 'nom_produit'
    elif 'produit' in df.columns:
        product_col = 'produit'
    else:
        print("❌ Erreur: Aucune colonne de produit trouvée")
        return 1
    
    prophet_df = prepare_prophet_data(
        df,
        target_column=target_col,
        product_filter=product_name,
        product_column=product_col
    )
    
    if len(prophet_df) == 0:
        print(f"\n❌ Erreur: Aucune donnée pour le produit '{product_name}'")
        return 1
    
    # Split data
    train, test = train_test_split(prophet_df, test_ratio=0.2)
    
    # Train model
    print("\n🤖 Entraînement du modèle...")
    model = train_prophet_model(
        train,
        daily_seasonality=settings.prophet.daily_seasonality,
        weekly_seasonality=settings.prophet.weekly_seasonality,
        yearly_seasonality=settings.prophet.yearly_seasonality,
        seasonality_mode=settings.prophet.seasonality_mode,
        changepoint_prior_scale=settings.prophet.changepoint_prior_scale
    )
    
    # Evaluate on test set
    print("\n📊 Évaluation sur données de test...")
    predictions_test = predict(model, test)
    
    y_true = test['y'].values
    y_pred = predictions_test['yhat'].values
    
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, unit="kg")
    
    # Future predictions
    print(f"\n🔮 Prédictions pour les {days} prochains jours...")
    future_preds = predict_future(model, periods=days)
    
    # Display predictions
    print("\n" + "=" * 70)
    print("📅 PRÉDICTIONS FUTURES")
    print("=" * 70)
    print(f"{'Date':<15} {'Prédit (kg)':<15} {'Min':<12} {'Max':<12}")
    print("-" * 54)
    
    for _, row in future_preds.tail(min(days, 10)).iterrows():
        date_str = row['ds'].strftime('%Y-%m-%d')
        print(f"{date_str:<15} {row['yhat']:>10.1f}    {row['yhat_lower']:>8.1f}    {row['yhat_upper']:>8.1f}")
    
    if days > 10:
        print(f"... ({days - 10} jours supplémentaires)")
    
    # Save results if requested
    if args.save:
        results_dir = create_results_directory(product_name)
        
        # Save predictions CSV
        future_preds[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
            results_dir / 'predictions.csv', index=False
        )
        
        # Save metrics JSON
        import json
        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés: {results_dir}")
    
    print("\n✅ Prédiction terminée!")
    return 0


def cmd_analyze(args):
    """
    Analyze a dataset and show statistics.
    
    Args:
        args: Parsed command line arguments
    """
    print("=" * 70)
    print("📊 ANALYSE DE DATASET - Clinique du Mont Vert")
    print("=" * 70)
    
    # Interactive dataset selection if none provided
    if not args.dataset:
        dataset_key = select_dataset_interactive()
        if dataset_key is None:
            print("\n❌ Sélection annulée.")
            return 1
    else:
        dataset_key = args.dataset
    
    # Get dataset path
    if dataset_key in DATASETS:
        dataset_file = DATASETS[dataset_key]
    else:
        dataset_file = dataset_key
    
    dataset_path = get_dataset_path(dataset_file)
    
    if not dataset_path.exists():
        print(f"\n❌ Erreur: Dataset non trouvé: {dataset_path}")
        return 1
    
    # Load data
    print(f"\n📥 Chargement: {dataset_path.name}")
    df = load_dataset(str(dataset_path))
    
    # Basic info
    print("\n" + "=" * 70)
    print("📋 INFORMATIONS GÉNÉRALES")
    print("=" * 70)
    print(f"Lignes:    {len(df):,}")
    print(f"Colonnes:  {len(df.columns)}")
    print(f"Mémoire:   {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Columns
    print("\n📊 Colonnes:")
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        null_pct = nulls / len(df) * 100
        print(f"   - {col:<25} ({dtype}) - {null_pct:.1f}% manquants")
    
    # Date range
    date_cols = ['date', 'ds', 'Date', 'DATE']
    date_col = next((c for c in date_cols if c in df.columns), None)
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        print(f"\n📅 Période: {df[date_col].min().date()} → {df[date_col].max().date()}")
        print(f"   Durée:   {(df[date_col].max() - df[date_col].min()).days} jours")
    
    # Products if available
    product_col = next((c for c in ['produit', 'product', 'Produit'] if c in df.columns), None)
    
    if product_col:
        products = df[product_col].unique()
        print(f"\n📦 Produits ({len(products)}):")
        for p in products[:10]:
            count = len(df[df[product_col] == p])
            print(f"   - {p}: {count:,} lignes")
        if len(products) > 10:
            print(f"   ... et {len(products) - 10} autres")
    
    # Numeric summary
    print("\n📈 Statistiques numériques:")
    numeric_cols = df.select_dtypes(include=['number']).columns[:5]
    for col in numeric_cols:
        print(f"\n   {col}:")
        print(f"      Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
        print(f"      Moyenne: {df[col].mean():.2f}, Médiane: {df[col].median():.2f}")
    
    print("\n✅ Analyse terminée!")
    return 0


def cmd_list_products(args):
    """
    List configured products.
    
    Args:
        args: Parsed command line arguments
    """
    print("=" * 70)
    print("📦 PRODUITS CONFIGURÉS - Clinique du Mont Vert")
    print("=" * 70)
    
    try:
        products_config = load_yaml_config("products")
        products = products_config.get('products', {})
        
        print(f"\n{'Produit':<20} {'Catégorie':<12} {'DLC':<8} {'Priorité':<10}")
        print("-" * 54)
        
        for key, prod in products.items():
            name = prod.get('name', key)
            category = prod.get('category', '-')
            dlc = f"{prod.get('dlc_days', '?')}j"
            priority = prod.get('priority', '-')
            print(f"{name:<20} {category:<12} {dlc:<8} {priority:<10}")
        
        print(f"\nTotal: {len(products)} produits configurés")
        
    except FileNotFoundError:
        print("\n⚠️ Fichier products.yaml non trouvé")
        print("   Créez le fichier config/products.yaml")
    
    return 0


def cmd_list_datasets(args):
    """
    List available datasets.
    
    Args:
        args: Parsed command line arguments
    """
    print("=" * 70)
    print("📊 DATASETS DISPONIBLES")
    print("=" * 70)
    
    print(f"\n{'Clé':<15} {'Fichier':<35} {'Existe':<10}")
    print("-" * 60)
    
    for key, filename in DATASETS.items():
        path = get_dataset_path(filename)
        exists = "✅" if path.exists() else "❌"
        print(f"{key:<15} {filename:<35} {exists:<10}")
    
    return 0


def cmd_train(args):
    """
    Train and save a Prophet model.
    
    Args:
        args: Parsed command line arguments
    """
    print("=" * 70)
    print("🤖 ENTRAÎNEMENT DE MODÈLE")
    print("=" * 70)
    
    if not check_prophet_available():
        print("❌ Erreur: Prophet n'est pas installé")
        return 1
    
    product_name = args.product
    dataset_key = args.dataset
    
    # Get dataset
    if dataset_key in DATASETS:
        dataset_file = DATASETS[dataset_key]
    else:
        dataset_file = dataset_key
    
    dataset_path = get_dataset_path(dataset_file)
    
    if not dataset_path.exists():
        print(f"❌ Dataset non trouvé: {dataset_path}")
        return 1
    
    # Load and prepare
    print(f"\n📥 Chargement du dataset {dataset_key}...")
    df = load_dataset(str(dataset_path))
    
    prophet_df = prepare_prophet_data(
        df,
        product_filter=product_name,
        product_column='produit'
    )
    
    # Train on all data
    print(f"\n🤖 Entraînement sur {len(prophet_df)} jours...")
    model = train_prophet_model(prophet_df, verbose=True)
    
    # Save model
    if args.output:
        import pickle
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\n💾 Modèle sauvegardé: {output_path}")
    
    print("\n✅ Entraînement terminé!")
    return 0


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog='stock-predict',
        description='Prédiction de stock hospitalier avec Prophet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  %(prog)s predict --product "Poulet Frais" --days 30
  %(prog)s analyze --dataset enriched
  %(prog)s list-products
  %(prog)s list-datasets
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Faire des prédictions')
    predict_parser.add_argument(
        '--product', '-p',
        default=None,
        help='Nom du produit à prédire (interactif si non spécifié)'
    )
    predict_parser.add_argument(
        '--days', '-d',
        type=int,
        default=None,
        help='Nombre de jours à prédire (interactif si non spécifié)'
    )
    predict_parser.add_argument(
        '--dataset', '-D',
        default=None,
        help='Dataset à utiliser (interactif si non spécifié)'
    )
    predict_parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Sauvegarder les résultats'
    )
    predict_parser.set_defaults(func=cmd_predict)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyser un dataset')
    analyze_parser.add_argument(
        '--dataset', '-D',
        default=None,
        help='Dataset à analyser (interactif si non spécifié)'
    )
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # List products command
    list_products_parser = subparsers.add_parser('list-products', help='Lister les produits')
    list_products_parser.set_defaults(func=cmd_list_products)
    
    # List datasets command
    list_datasets_parser = subparsers.add_parser('list-datasets', help='Lister les datasets')
    list_datasets_parser.set_defaults(func=cmd_list_datasets)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Entraîner un modèle')
    train_parser.add_argument(
        '--product', '-p',
        required=True,
        help='Produit pour lequel entraîner'
    )
    train_parser.add_argument(
        '--dataset', '-D',
        default='enriched',
        help='Dataset à utiliser'
    )
    train_parser.add_argument(
        '--output', '-o',
        help='Chemin pour sauvegarder le modèle (.pkl)'
    )
    train_parser.set_defaults(func=cmd_train)
    
    return parser


def main():
    """Main entry point."""
    # Import pandas here to avoid slow startup for help
    global pd
    import pandas as pd
    
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
