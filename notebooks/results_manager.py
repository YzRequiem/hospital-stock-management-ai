"""
Results Manager - Gestion intelligente des résultats d'analyse
Crée automatiquement des dossiers horodatés pour stocker les résultats
"""

import os
import shutil
from datetime import datetime
from pathlib import Path


class ResultsManager:
    """Gestionnaire de résultats avec organisation par timestamp"""

    def __init__(self, base_results_dir="../results"):
        """
        Initialise le gestionnaire de résultats

        Args:
            base_results_dir: Chemin vers le dossier de résultats de base
        """
        self.base_results_dir = Path(base_results_dir)
        self.current_run_dir = None
        self.timestamp = None

    def create_run_directory(self):
        """
        Crée un nouveau dossier pour cette exécution avec timestamp
        Format: YYYYMMDD_HHMMSS

        Returns:
            Path: Chemin vers le dossier créé
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = self.base_results_dir / self.timestamp

        # Créer le dossier et ses sous-dossiers
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        (self.current_run_dir / "graphs").mkdir(exist_ok=True)

        print(f"📁 Dossier de résultats créé : {self.current_run_dir}")
        return self.current_run_dir

    def save_file(self, source_path, subdirectory=None):
        """
        Sauvegarde un fichier dans le dossier de résultats

        Args:
            source_path: Chemin du fichier source
            subdirectory: Sous-dossier optionnel (ex: "graphs")
        """
        if self.current_run_dir is None:
            self.create_run_directory()

        source = Path(source_path)
        if not source.exists():
            print(f"⚠️  Fichier introuvable : {source_path}")
            return

        # Déterminer la destination
        if subdirectory:
            dest_dir = self.current_run_dir / subdirectory
            dest_dir.mkdir(exist_ok=True)
        else:
            dest_dir = self.current_run_dir

        dest = dest_dir / source.name

        # Copier le fichier
        shutil.copy2(source, dest)
        print(f"✅ Sauvegardé : {dest.relative_to(self.base_results_dir.parent)}")

    def save_graph(self, graph_path):
        """Sauvegarde un graphique dans le sous-dossier graphs"""
        self.save_file(graph_path, subdirectory="graphs")

    def save_data(self, data_path):
        """Sauvegarde un fichier de données dans le dossier principal"""
        self.save_file(data_path)

    def create_summary_file(self, product_name, summary_dict):
        """
        Crée un fichier résumé de l'exécution

        Args:
            product_name: Nom du produit analysé
            summary_dict: Dictionnaire contenant les informations de résumé
        """
        if self.current_run_dir is None:
            self.create_run_directory()

        summary_file = self.current_run_dir / "README.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("RÉSUMÉ DE L'ANALYSE\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Produit analysé : {product_name}\n")
            f.write(f"Date d'exécution : {self.timestamp}\n\n")

            if summary_dict:
                f.write("Résultats clés :\n")
                for key, value in summary_dict.items():
                    f.write(f"  - {key}: {value}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("FICHIERS GÉNÉRÉS\n")
            f.write("=" * 70 + "\n\n")

            # Lister les fichiers
            f.write("Graphiques (dans graphs/) :\n")
            graphs_dir = self.current_run_dir / "graphs"
            if graphs_dir.exists():
                for file in sorted(graphs_dir.glob("*.png")):
                    f.write(f"  - {file.name}\n")

            f.write("\nDonnées :\n")
            for file in sorted(self.current_run_dir.glob("*.csv")):
                f.write(f"  - {file.name}\n")
            for file in sorted(self.current_run_dir.glob("*.json")):
                f.write(f"  - {file.name}\n")

        print(f"✅ Fichier résumé créé : {summary_file.relative_to(self.base_results_dir.parent)}")

    def get_run_path(self):
        """Retourne le chemin du dossier d'exécution actuel"""
        if self.current_run_dir is None:
            self.create_run_directory()
        return self.current_run_dir

    def list_previous_runs(self, limit=10):
        """
        Liste les exécutions précédentes

        Args:
            limit: Nombre maximum d'exécutions à afficher

        Returns:
            List[Path]: Liste des dossiers d'exécutions
        """
        if not self.base_results_dir.exists():
            return []

        runs = sorted(
            [d for d in self.base_results_dir.iterdir() if d.is_dir()],
            reverse=True
        )
        return runs[:limit]

    def print_previous_runs(self, limit=10):
        """Affiche les exécutions précédentes"""
        runs = self.list_previous_runs(limit)

        if not runs:
            print("📁 Aucune exécution précédente trouvée")
            return

        print(f"\n📁 {len(runs)} exécution(s) précédente(s) :")
        print("=" * 70)

        for i, run_dir in enumerate(runs, 1):
            # Compter les fichiers
            num_graphs = len(list((run_dir / "graphs").glob("*.png"))) if (run_dir / "graphs").exists() else 0
            num_csv = len(list(run_dir.glob("*.csv")))
            num_json = len(list(run_dir.glob("*.json")))

            print(f"\n{i}. {run_dir.name}")
            print(f"   📊 {num_graphs} graphiques, {num_csv} CSV, {num_json} JSON")

            # Afficher le README s'il existe
            readme = run_dir / "README.txt"
            if readme.exists():
                with open(readme, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if "Produit analysé" in line:
                            print(f"   {line.strip()}")
                            break


# Fonction utilitaire pour faciliter l'utilisation
def setup_results_manager():
    """
    Configure et retourne un gestionnaire de résultats
    À utiliser au début de votre notebook
    """
    manager = ResultsManager()
    manager.create_run_directory()
    return manager


# Exemple d'utilisation dans le notebook
if __name__ == "__main__":
    print("🔧 Test du Results Manager\n")

    # Créer un gestionnaire
    manager = ResultsManager()

    # Afficher les exécutions précédentes
    manager.print_previous_runs()

    # Créer un nouveau dossier
    run_dir = manager.create_run_directory()

    print(f"\n✅ Nouveau dossier créé : {run_dir}")
    print(f"📂 Chemin complet : {run_dir.absolute()}")
