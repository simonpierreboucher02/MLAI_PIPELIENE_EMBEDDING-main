# main.py

import logging
from pathlib import Path
import yaml
from embedding_processor import EmbeddingProcessor


def setup_logging(log_config):
    """
    Configure le système de logging en fonction de la configuration YAML.

    Args:
        log_config (dict): Configuration du logging.
    """
    log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_format = log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
    log_file = log_config.get("file", "embedding_processor.log")
    log_stream = log_config.get("stream", True)

    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    if log_stream:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )


def load_config(config_path):
    """
    Charge la configuration depuis un fichier YAML.

    Args:
        config_path (str ou Path): Chemin vers le fichier de configuration YAML.

    Returns:
        dict: Configuration chargée.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"🚫 Erreur lors du chargement de la configuration : {str(e)}")
        exit(1)


def main():
    # Chemin vers le fichier de configuration
    config_path = "config.yaml"

    # Charger la configuration
    config = load_config(config_path)

    # Configurer le logging
    setup_logging(config.get("logging", {}))
    logger = logging.getLogger('EmbeddingProcessor')

    # Extraire les paramètres de configuration
    input_dir = config.get("input_dir", "chemin/vers/dossier_txt")
    output_dir = config.get("output_dir", "chemin/vers/dossier_embeddings")
    api_keys = config.get("openai_api_keys", [])

    # Vérifiez que les clés API sont fournies
    if not api_keys:
        # Si les clés ne sont pas directement dans config.yaml, vérifier le fichier
        api_keys_file = config.get("api_keys_file", "api_keys.txt")
        if not Path(api_keys_file).is_file():
            logger.error(f"🚫 Le fichier des clés API '{api_keys_file}' n'existe pas.")
            return

        # Lire les clés API depuis le fichier
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            api_keys = [line.strip() for line in f if line.strip()]

        if not api_keys:
            logger.error("🚫 Aucune clé API OpenAI trouvée dans le fichier.")
            return

    # Paramètres additionnels
    verbose = config.get("verbose", True)
    chunk_size = config.get("chunk_size", 1200)
    overlap_size = config.get("overlap_size", 100)
    embedding_model = config.get("embedding_model", "text-embedding-ada-002")
    system_prompt = config.get("system_prompt", None)
    llm_max_tokens = config.get("llm_max_tokens", 200)
    max_workers = config.get("max_workers", 10)

    # Initialiser le processeur d'embeddings
    processor = EmbeddingProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        openai_api_keys=api_keys,
        verbose=verbose,
        logger=logger,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        embedding_model=embedding_model,
        system_prompt=system_prompt,
        llm_max_tokens=llm_max_tokens
    )

    # Exécuter le traitement des fichiers
    processor.process_all_files(max_workers=max_workers)
    logger.info("🎉 Génération des embeddings terminée.")


if __name__ == "__main__":
    main()
