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
from pathlib import Path
from datetime import datetime

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


def cmd_predict(args):
    """
    Run prediction for a specific product.
    
    Args:
        args: Parsed command line arguments
    """
    print("=" * 70)
    print("🏥 PRÉDICTION DE STOCK - Clinique du Mont Vert")
    print("=" * 70)
    
    if not check_prophet_available():
        print("❌ Erreur: Prophet n'est pas installé")
        print("   Installez-le avec: pip install prophet")
        return 1
    
    # Load configuration
    product_name = args.product
    days = args.days
    dataset_key = args.dataset
    
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
    
    prophet_df = prepare_prophet_data(
        df,
        product_filter=product_name,
        product_column='produit'
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
        required=True,
        help='Nom du produit à prédire'
    )
    predict_parser.add_argument(
        '--days', '-d',
        type=int,
        default=30,
        help='Nombre de jours à prédire (défaut: 30)'
    )
    predict_parser.add_argument(
        '--dataset', '-D',
        default='enriched',
        help='Dataset à utiliser: base, realistic, enriched (défaut: enriched)'
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
        default='enriched',
        help='Dataset à analyser'
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
