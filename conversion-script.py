#!/usr/bin/env python3
# convert_podgpt_legal.py
# Precise conversion of actual PodGPT repo to Legal AI Intake System
# Based on real structure: https://github.com/vkola-lab/PodGPT/tree/main

import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PodGPTLegalConverter:
    """
    Converts the real PodGPT medical system to Legal AI Intake
    Uses actual PodGPT repository structure
    """
    
    def __init__(self):
        self.podgpt_path = Path("./PodGPT")
        self.legal_path = Path("./ai-legal-intake")
        
        # Real PodGPT structure (from GitHub)
        self.podgpt_structure = {
            "core_files": [
                "rag_pipeline/download_model.py",
                "rag_pipeline/pipeline.py", 
                "rag_pipeline/config.py",
                "rag_pipeline/embeddings.py",
                "rag_pipeline/retrieval.py",
                "rag_pipeline/generation.py",
                "utils/data_processing.py",
                "utils/evaluation.py",
                "api/app.py",
                "requirements.txt"
            ],
            "config_files": [
                "config/model_config.yaml",
                "config/data_config.yaml", 
                "config/api_config.yaml"
            ],
            "medical_specific": [
                "data/medical_qa/",
                "data/pubmed/",  
                "data/clinical_notes/",
                "models/medical_bert/",
                "prompts/medical_prompts.json"
            ]
        }

    def clone_and_convert(self):
        """Clone PodGPT and convert to legal system"""
        
        print("🔄 Converting PodGPT Medical → Legal AI Intake System")
        
        # 1. Clone PodGPT if not exists
        self._clone_podgpt()
        
        # 2. Copy core architecture (unchanged)
        self._copy_core_architecture()
        
        # 3. Replace medical-specific components
        self._replace_medical_components()
        
        # 4. Create legal-specific additions
        self._create_legal_additions()
        
        # 5. Update configurations
        self._update_configurations()
        
        # 6. Create legal datasets
        self._setup_legal_datasets()
        
        # 7. Update requirements
        self._update_requirements()
        
        print("✅ Conversion completed! Legal AI Intake system ready.")

    def _clone_podgpt(self):
        """Clone PodGPT repository"""
        
        if self.podgpt_path.exists():
            logger.info("📁 PodGPT already exists, pulling latest...")
            subprocess.run(["git", "pull"], cwd=self.podgpt_path, check=False)
        else:
            logger.info("📥 Cloning PodGPT repository...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/vkola-lab/PodGPT.git",
                str(self.podgpt_path)
            ], check=True)
            
        logger.info("✅ PodGPT repository ready")

    def _copy_core_architecture(self):
        """Copy PodGPT's core RAG architecture (keep unchanged)"""
        
        logger.info("📋 Copying core RAG architecture...")
        
        # Create target directory
        self.legal_path.mkdir(exist_ok=True)
        
        # Copy core RAG pipeline files (these work for any domain)
        core_mappings = {
            # RAG Pipeline (core logic - domain agnostic)
            "rag_pipeline/pipeline.py": "src/rag_pipeline/pipeline.py",
            "rag_pipeline/embeddings.py": "src/rag_pipeline/embeddings.py", 
            "rag_pipeline/retrieval.py": "src/rag_pipeline/retrieval.py",
            "rag_pipeline/generation.py": "src/rag_pipeline/generation.py",
            "rag_pipeline/config.py": "src/rag_pipeline/config.py",
            
            # Utilities (domain agnostic)
            "utils/data_processing.py": "src/utils/data_processing.py",
            "utils/evaluation.py": "src/utils/evaluation.py",
            
            # API structure (modify for legal)
            "api/app.py": "src/api/legal_app.py",
            
            # Base requirements
            "requirements.txt": "requirements_base.txt"
        }
        
        for src_file, dst_file in core_mappings.items():
            src_path = self.podgpt_path / src_file
            dst_path = self.legal_path / dst_file
            
            if src_path.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                logger.info(f"📄 Copied: {src_file} → {dst_file}")
            else:
                logger.warning(f"⚠️ Source not found: {src_file}")

    def _replace_medical_components(self):
        """Replace medical-specific components with legal equivalents"""
        
        logger.info("🔄 Replacing medical components with legal equivalents...")
        
        # Replace download_model.py with our legal version
        legal_download_model = '''#!/usr/bin/env python3
# Legal AI Intake - Model Downloader (adapted from PodGPT)

import os
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm

class LegalModelDownloader:
    """Download legal-specific models (adapted from PodGPT's medical downloader)"""
    
    def __init__(self, models_dir="./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Legal models (replacing PodGPT's medical models)
        self.models = {
            # Speech Processing (same as PodGPT but for legal conversations)
            "whisper-large-v3": "openai/whisper-large-v3",
            
            # Legal Embeddings (replacing medical embeddings)
            "legal-bert": "nlpaueb/legal-bert-base-uncased", 
            "legal-roberta": "joelito/legal-xlm-roberta-base",
            
            # Legal Entity Recognition (replacing medical NER)
            "legal-ner": "law-ai/InLegalBERT",
            
            # Multilingual Legal Models
            "legal-multilingual": "joelito/legal-xlm-roberta-base"
        }
    
    def download_all(self):
        """Download all legal models"""
        for model_name, model_path in self.models.items():
            print(f"📥 Downloading {model_name}...")
            self.down
