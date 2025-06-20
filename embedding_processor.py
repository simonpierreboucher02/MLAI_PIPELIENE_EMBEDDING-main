# embedding_processor.py

import re
import json
import logging
import threading
import requests
import numpy as np
from pathlib import Path
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, as_completed


class EmbeddingProcessor:
    """
    Une classe pour traiter les embeddings de texte en utilisant l'API d'OpenAI.

    Ce processeur gère le découpage du texte, la contextualisation avec GPT-4,
    et la génération des embeddings. Il gère les limites de taux d'API en alternant
    plusieurs clés API et supporte le traitement concurrent pour plus d'efficacité.
    """

    def __init__(self, input_dir, output_dir, openai_api_keys, verbose=False, logger=None,
                 chunk_size=1200, overlap_size=100, embedding_model="text-embedding-ada-002",
                 system_prompt=None, llm_max_tokens=200):
        """
        Initialise l'EmbeddingProcessor avec les configurations nécessaires.

        Args:
            input_dir (str ou Path): Dossier contenant les fichiers texte d'entrée.
            output_dir (str ou Path): Dossier où les embeddings et les métadonnées sont sauvegardés.
            openai_api_keys (list): Liste des clés API OpenAI pour l'alternance.
            verbose (bool, optional): Flag pour activer le logging détaillé. Defaults à False.
            logger (logging.Logger, optional): Logger pour enregistrer les informations et les erreurs. Defaults à None.
            chunk_size (int, optional): Taille maximale des chunks en tokens. Defaults à 1200.
            overlap_size (int, optional): Nombre de tokens de chevauchement entre les chunks. Defaults à 100.
            embedding_model (str, optional): Modèle d'embedding à utiliser. Defaults à "text-embedding-ada-002".
            system_prompt (str, optional): Prompt système pour la contextualisation via LLM. Defaults à None.
            llm_max_tokens (int, optional): Nombre maximal de tokens pour la réponse du LLM. Defaults à 200.
        """
        # Configurer les chemins
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialiser les listes globales pour tous les fichiers
        self.all_embeddings = []
        self.all_metadata = []

        # Configurer l'API OpenAI
        self.openai_api_keys = openai_api_keys
        self.headers_cycle = cycle([
            {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            } for key in self.openai_api_keys
        ])
        self.lock = threading.Lock()

        # Configurer le logging
        self.logger = logger or logging.getLogger(__name__)
        self.verbose = verbose

        # Configurer le découpage du texte et le modèle d'embedding
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.embedding_model = embedding_model

        # Configurer la contextualisation
        self.system_prompt = system_prompt or (
            "You are an expert analyst. The following text is an excerpt from a larger document. "
            "Your task is to provide context for the next section by referencing the overall document content. "
            "Ensure the context helps in better understanding the excerpt."
        )
        self.llm_max_tokens = llm_max_tokens

    def chunk_text(self, text):
        """
        Divise le texte en chunks plus petits avec des régions de chevauchement.

        Returns:
            list: Liste des chunks de texte.
        """
        try:
            tokens = text.split()

            # Si le texte est plus court que la taille du chunk, le traiter comme un seul chunk
            if len(tokens) <= self.chunk_size:
                return [text]

            chunks = []
            for i in range(0, len(tokens), self.chunk_size - self.overlap_size):
                chunk = ' '.join(tokens[i:i + self.chunk_size])
                chunks.append(chunk)

            # Assurer que le dernier chunk n'est pas trop petit
            if len(chunks) > 1 and len(tokens[-(self.chunk_size - self.overlap_size):]) < self.chunk_size // 2:
                # Fusionner le dernier chunk avec le précédent s'il est trop petit
                last_chunk = ' '.join(tokens[-self.chunk_size:])
                chunks[-1] = last_chunk

            return chunks

        except Exception as e:
            self.logger.error(f"Erreur lors du découpage du texte : {str(e)}")
            return [text]  # Retourner le texte complet en cas d'erreur

    def get_contextualized_chunk(self, chunk, full_text, headers, document_name, page_num, chunk_id):
        """
        Obtient des informations contextuelles pour un chunk de texte en utilisant GPT-4.

        Args:
            chunk (str): Le chunk de texte à contextualiser.
            full_text (str): Le texte complet du document pour référence.
            headers (dict): En-têtes HTTP contenant l'autorisation de l'API.
            document_name (str): Nom du document en cours de traitement.
            page_num (int): Numéro de page dans le document.
            chunk_id (int): Identifiant du chunk dans la page.

        Returns:
            str ou None: Les informations contextuelles générées par GPT-4 si réussi, None sinon.
        """
        try:
            system_prompt = {
                "role": "system",
                "content": self.system_prompt
            }
            user_prompt = {
                "role": "user",
                "content": f"Document: {full_text}\n\nChunk: {chunk}\n\nPlease provide context for this excerpt in French."
            }

            payload = {
                "model": "gpt-4",
                "messages": [system_prompt, user_prompt],
                "temperature": 0,
                "max_tokens": self.llm_max_tokens,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }

            if self.verbose:
                self.logger.info(f"Appel du LLM pour {document_name} page {page_num}, chunk {chunk_id}")

            self.logger.info(f"🔗 Appel de l'API GPT-4 pour {document_name} page {page_num} chunk {chunk_id}")

            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']

        except Exception as e:
            self.logger.error(f"Erreur lors de la contextualisation : {str(e)}")
            return None

    def get_embedding(self, text, headers, document_name, page_num, chunk_id):
        """
        Obtient un vecteur d'embedding pour un texte donné en utilisant l'API d'OpenAI.

        Args:
            text (str): Le texte à embeder.
            headers (dict): En-têtes HTTP contenant l'autorisation de l'API.
            document_name (str): Nom du document en cours de traitement.
            page_num (int): Numéro de page dans le document.
            chunk_id (int): Identifiant du chunk dans la page.

        Returns:
            list ou None: Le vecteur d'embedding si réussi, None sinon.
        """
        try:
            payload = {
                "input": text,
                "model": self.embedding_model,
                "encoding_format": "float"
            }

            if self.verbose:
                self.logger.info(f"Appel de l'API Embedding pour {document_name} page {page_num} chunk {chunk_id}")

            self.logger.info(f"🔗 Appel de l'API Embedding pour {document_name} page {page_num} chunk {chunk_id}")

            response = requests.post(
                'https://api.openai.com/v1/embeddings',
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()['data'][0]['embedding']

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de l'embedding : {str(e)}")
            return None

        finally:
            # Remettre la clé API dans le cycle
            if headers.get("Authorization"):
                api_key = headers["Authorization"].split("Bearer ")[-1]
                self.headers_cycle = cycle([
                    {
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json"
                    } for key in self.openai_api_keys
                ])

    def process_chunk(self, chunk_info):
        """
        Traite un chunk de texte spécifique en le contextualisant et en générant son embedding.

        Args:
            chunk_info (tuple): Un tuple contenant :
                - txt_file_path (Path): Chemin vers le fichier texte.
                - chunk_id (int): Identifiant du chunk.
                - chunk (str): Contenu du chunk.
                - full_text (str): Texte complet du document.
                - document_name (str): Nom du document.
                - page_num (int): Numéro de page dans le document.

        Returns:
            tuple: (embedding, metadata) si réussi, (None, None) sinon.
        """
        try:
            txt_file_path, chunk_id, chunk, full_text, document_name, page_num = chunk_info

            with self.lock:
                headers = next(self.headers_cycle)

            # Obtenir le contexte via GPT-4
            context = self.get_contextualized_chunk(chunk, full_text, headers, document_name, page_num, chunk_id)
            if not context:
                return None, None

            # Combiner le contexte avec le chunk original
            combined_text = f"{context}\n\nContext:\n{chunk}"

            # Obtenir l'embedding du texte combiné
            embedding = self.get_embedding(combined_text, headers, document_name, page_num, chunk_id)

            if embedding:
                metadata = {
                    "filename": txt_file_path.name,
                    "chunk_id": chunk_id,
                    "page_num": page_num,
                    "text_raw": chunk,
                    "context": context,
                    "text": combined_text
                }
                return embedding, metadata

            return None, None

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du chunk : {str(e)}")
            return None, None

    def process_file(self, txt_file_path):
        """
        Prépare les informations des chunks à partir d'un fichier texte pour l'embedding.

        Args:
            txt_file_path (Path): Chemin vers le fichier texte.

        Returns:
            list: Liste de tuples contenant les informations des chunks pour le traitement.
        """
        try:
            self.logger.info(f"📂 Traitement du fichier : {txt_file_path}")

            with open(txt_file_path, 'r', encoding='utf-8') as file:
                full_text = file.read()

            chunks = self.chunk_text(full_text)

            # Extraire le numéro de page depuis le nom du fichier en utilisant regex (à ajuster si nécessaire)
            # Exemple de nom de fichier : "document_page_1.txt"
            match = re.search(r'_page_(\d+)', txt_file_path.stem)
            if match:
                page_num = int(match.group(1))
            else:
                page_num = 1  # Par défaut si non trouvé

            chunk_infos = [
                (txt_file_path, i, chunk, full_text, txt_file_path.stem, page_num)
                for i, chunk in enumerate(chunks, 1)
            ]

            return chunk_infos

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du fichier {txt_file_path} : {str(e)}")
            return []

    def process_all_files(self, max_workers=10):
        """
        Traite tous les fichiers textes dans le répertoire d'entrée pour générer les embeddings.

        Args:
            max_workers (int, optional): Nombre maximal de threads pour le traitement concurrent. Defaults à 10.
        """
        try:
            txt_files = list(self.input_dir.glob('*.txt'))
            total_files = len(txt_files)
            self.logger.info(f"📢 Début du traitement de {total_files} fichiers dans '{self.input_dir}'")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for txt_file_path in txt_files:
                    chunk_infos = self.process_file(txt_file_path)
                    for chunk_info in chunk_infos:
                        futures.append(executor.submit(self.process_chunk, chunk_info))

                for future in as_completed(futures):
                    embedding, metadata = future.result()
                    if embedding and metadata:
                        self.all_embeddings.append(embedding)
                        self.all_metadata.append(metadata)

            if self.all_embeddings:
                # Sauvegarder les résultats
                chunks_json_path = self.output_dir / "chunks.json"
                with open(chunks_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump({"metadata": self.all_metadata}, json_file, ensure_ascii=False, indent=4)

                embeddings_npy_path = self.output_dir / "embeddings.npy"
                np.save(embeddings_npy_path, np.array(self.all_embeddings))

                self.logger.info(f"✅ Fichiers créés : {chunks_json_path} et {embeddings_npy_path}")

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des fichiers : {str(e)}")
            raise
