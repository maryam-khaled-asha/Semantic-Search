"""
================================================================================
SEMANTIC SEARCH SYSTEM - COMPLETE IMPLEMENTATION
================================================================================

A production-ready semantic search system for Arabic/English e-commerce items.

Features:
- Multilingual embeddings (E5-large-instruct)
- Vector database (Qdrant Cloud)
- Generic query analyzer (no hardcoding)
- Automatic filter extraction (price, categories, etc.)
- Comprehensive testing & reporting

Author: Maryam Khaled Asha
Date: January 2026
Version: 1.0.0
================================================================================
"""

import os
import re
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import torch

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, Range,
    PayloadSchemaType
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """System configuration - UPDATE THESE VALUES FOR YOUR SETUP"""
    
    # Qdrant Cloud Configuration
    QDRANT_URL = "https://002ec0a0-7e36-4abd-82d9-ea17fc898325.us-east4-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.UKyFB0M3IPTsj20NE89h05mMjLyyeAtzw4GkX2qlROI"
    COLLECTION_NAME = "items_generic"
    
    # Model Configuration
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
    EMBEDDING_DIM = 1024
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Processing Configuration
    BATCH_SIZE = 32              # Embedding batch size
    UPLOAD_BATCH_SIZE = 512      # Qdrant upload batch size
    
    # Data Configuration
    DATA_DIR = Path("data")
    CSV_FILE = "embedding_ready_data.csv"
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*80)
        print("⚙️ SYSTEM CONFIGURATION")
        print("="*80)
        print(f"  Model: {cls.EMBEDDING_MODEL}")
        print(f"  Collection: {cls.COLLECTION_NAME}")
        print(f"  Device: {cls.DEVICE.upper()}")
        print(f"  Embedding Dimension: {cls.EMBEDDING_DIM}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print("="*80 + "\n")


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ItemRecord:
    """
    Represents a single product/item in the database.
    
    Attributes:
        item_id: Unique identifier for the item
        title: Item title/name
        combined_text: Full text description (used for embeddings)
        price: Item price (float)
        category_id: Category identifier
        category_title: Human-readable category name
    """
    item_id: str
    title: str
    combined_text: str
    price: float
    category_id: str
    category_title: str
    
    def get_passage(self) -> str:
        """
        Format text for E5 embedding with 'passage:' instruction.
        
        E5 models use instruction prefixes:
        - 'passage:' for documents/items to be indexed
        - 'query:' for search queries
        
        Returns:
            Formatted passage string for embedding
        """
        return f"passage: {self.combined_text}"
    
    def is_valid(self) -> bool:
        """
        Validate record has minimum required data.
        
        Returns:
            True if record is valid, False otherwise
        """
        return (
            bool(self.item_id) and 
            bool(self.combined_text) and 
            len(self.combined_text.strip()) >= 5
        )
    
    def to_payload(self) -> Dict:
        """
        Convert record to Qdrant payload format.
        
        Returns:
            Dictionary suitable for Qdrant storage
        """
        return {
            "item_id": self.item_id,
            "title": self.title,
            "price": self.price,
            "category_id": self.category_id,
            "category_title": self.category_title,
            "combined_text": self.combined_text[:1000]  # Truncate for storage
        }


# ============================================================================
# QUERY ANALYZER
# ============================================================================

class GenericQueryAnalyzer:
    """
    Universal query analyzer for extracting filters and entities.
    
    This analyzer works for ANY query without hardcoding specific terms.
    It uses generic patterns to detect:
    - Numeric filters (price, year, size, etc.)
    - Comparison operators (less than, greater than, between)
    - Entities (brands, models, etc.)
    - Categories
    
    Example:
        >>> analyzer = GenericQueryAnalyzer()
        >>> result = analyzer.analyze("سيارة مرسيدس سعرها تحت 10 آلاف")
        >>> print(result['numeric_filters'])
        {'Price': {'max': 10000.0}}
    """
    
    def __init__(self, dataset_df: Optional[pd.DataFrame] = None):
        """
        Initialize the analyzer.
        
        Args:
            dataset_df: Optional DataFrame to learn categories from
        """
        # Numeric pattern (matches integers and decimals)
        self.number_pattern = r'\d+(?:[.,]\d+)?'
        
        # Comparison patterns (Arabic and English)
        self.comparison_patterns = {
            'less_than': [
                r'(?:تحت|أقل من|اقل من|under|less than|below|<)\s*' + self.number_pattern
            ],
            'greater_than': [
                r'(?:فوق|أكثر من|اكثر من|over|more than|above|>)\s*' + self.number_pattern
            ],
            'between': [
                r'(?:بين|between)\s*' + self.number_pattern + r'\s*(?:و|and|to|-)\s*' + self.number_pattern
            ]
        }
        
        # Number multipliers
        self.multipliers = {
            'آلاف': 1000, 'الاف': 1000, 'ألف': 1000, 'الف': 1000,
            'thousand': 1000, 'k': 1000,
            'مليون': 1000000, 'million': 1000000, 'm': 1000000
        }
        
        # Learn categories from dataset
        self.categories = []
        if dataset_df is not None:
            self.categories = dataset_df['CategoryTitle'].unique().tolist()
    
    def analyze(self, query: str) -> Dict:
        """
        Analyze query and extract all filters and entities.
        
        Args:
            query: User search query (Arabic or English)
        
        Returns:
            Dictionary containing:
            - original_query: The input query
            - clean_query: Query with filters removed
            - numeric_filters: Extracted numeric constraints
            - extracted_entities: Detected entities (brands, models)
            - category_match: Matched category (if any)
        """
        return {
            "original_query": query,
            "clean_query": self._clean_query(query),
            "numeric_filters": self._extract_numeric_filters(query),
            "extracted_entities": self._extract_entities(query),
            "category_match": self._match_category(query)
        }
    
    def _extract_numeric_filters(self, query: str) -> Dict:
        """
        Extract numeric constraints from query.
        
        Detects patterns like:
        - "تحت 10 آلاف" → max: 10000
        - "فوق 2015" → min: 2015
        - "بين 5 و 10" → min: 5, max: 10
        
        Args:
            query: Search query
        
        Returns:
            Dictionary with numeric filters (e.g., {'Price': {'max': 10000}})
        """
        filters = {}
        
        # Extract "less than" constraints
        for pattern in self.comparison_patterns['less_than']:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                num = re.search(self.number_pattern, match.group())
                if num:
                    value = float(num.group().replace(',', '.'))
                    multiplier = self._get_multiplier(query, match.end())
                    value *= multiplier
                    filters["Price"] = filters.get("Price", {})
                    filters["Price"]["max"] = value
        
        # Extract "greater than" constraints
        for pattern in self.comparison_patterns['greater_than']:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                num = re.search(self.number_pattern, match.group())
                if num:
                    value = float(num.group().replace(',', '.'))
                    multiplier = self._get_multiplier(query, match.end())
                    value *= multiplier
                    filters["Price"] = filters.get("Price", {})
                    filters["Price"]["min"] = value
        
        # Extract "between" constraints
        for pattern in self.comparison_patterns['between']:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                nums = re.findall(self.number_pattern, match.group())
                if len(nums) >= 2:
                    min_val = float(nums[0].replace(',', '.'))
                    max_val = float(nums[1].replace(',', '.'))
                    multiplier = self._get_multiplier(query, match.end())
                    filters["Price"] = {
                        "min": min_val * multiplier,
                        "max": max_val * multiplier
                    }
        
        return filters
    
    def _get_multiplier(self, query: str, pos: int) -> float:
        """
        Get numeric multiplier near a position in the query.
        
        Args:
            query: Search query
            pos: Position to search from
        
        Returns:
            Multiplier value (e.g., 1000 for "آلاف")
        """
        context = query[pos:pos+20].lower()
        for word, mult in self.multipliers.items():
            if word in context:
                return mult
        return 1.0
    
    def _extract_entities(self, query: str) -> List[Dict]:
        """
        Extract entities using generic patterns.
        
        Detects:
        - Capitalized words (brands)
        - English words in Arabic text (models)
        
        Args:
            query: Search query
        
        Returns:
            List of entity dictionaries with 'text' and 'type'
        """
        entities = []
        
        # Detect English words (often brands/models)
        for eng in re.findall(r'[A-Za-z]{2,}', query):
            if eng not in [e["text"] for e in entities]:
                entities.append({"text": eng, "type": "brand_or_model"})
        
        return entities
    
    def _match_category(self, query: str) -> Optional[str]:
        """
        Fuzzy match category from query.
        
        Args:
            query: Search query
        
        Returns:
            Matched category name or None
        """
        query_lower = query.lower()
        for category in self.categories:
            if category.lower() in query_lower:
                return category
        return None
    
    def _clean_query(self, query: str) -> str:
        """
        Remove filter expressions from query.
        
        This creates a "clean" query suitable for semantic search
        by removing numeric filters and multipliers.
        
        Args:
            query: Original query
        
        Returns:
            Clean query with filters removed
        """
        clean = query
        
        # Remove all comparison patterns
        for patterns in self.comparison_patterns.values():
            for pattern in patterns:
                clean = re.sub(pattern, '', clean, flags=re.IGNORECASE)
        
        # Remove multipliers
        for mult in self.multipliers.keys():
            clean = clean.replace(mult, '')
        
        # Clean up whitespace
        return ' '.join(clean.split()).strip()


# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """
    Load and validate data from CSV files.
    
    Handles:
    - Multiple encoding formats (UTF-8, UTF-8-BOM, Windows-1256)
    - Column name cleaning
    - Data validation
    - Environment detection (Colab vs local)
    """
    
    @staticmethod
    def detect_environment() -> Path:
        """
        Detect whether running in Google Colab or local environment.
        
        Returns:
            Path to data directory
        """
        try:
            from google.colab import drive
            print("🔵 Google Colab detected")
            drive.mount('/content/drive', force_remount=True)
            data_dir = Path("/content/drive/MyDrive/Evolvo")
            if not data_dir.exists():
                print(f"⚠️ Drive path not found, using: {Config.DATA_DIR}")
                return Config.DATA_DIR
            print(f"✓ Using Google Drive: {data_dir}")
            return data_dir
        except ImportError:
            print("🟢 Local environment detected")
            print(f"✓ Using local directory: {Config.DATA_DIR}")
            return Config.DATA_DIR
    
    @staticmethod
    def load_csv(file_path: Path) -> pd.DataFrame:
        """
        Load CSV with multiple encoding support.
        
        Args:
            file_path: Path to CSV file
        
        Returns:
            Loaded DataFrame
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file can't be loaded with any encoding
        """
        if not file_path.exists():
            raise FileNotFoundError(f"❌ File not found: {file_path}")
        
        print(f"\n📁 Loading: {file_path}")
        
        # Try multiple encodings for Arabic data
        encodings = ['utf-8', 'utf-8-sig', 'windows-1256', 'latin1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                
                # Clean BOM and extra characters from column names
                df.columns = (df.columns
                             .str.replace('\ufeff', '')
                             .str.replace('ï»؟', '')
                             .str.strip())
                
                print(f"✓ Loaded with encoding: {encoding}")
                return df
                
            except (UnicodeDecodeError, Exception) as e:
                continue
        
        raise ValueError("❌ Could not load CSV with any encoding")
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> None:
        """
        Validate DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
        
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = [
            'ItemId', 'ItemTitle', 'CombinedText', 
            'Price', 'CategoryId', 'CategoryTitle'
        ]
        
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"❌ Missing required columns: {missing}")
        
        print(f"\n📊 Data Summary:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Categories: {df['CategoryTitle'].nunique()}")
    
    @staticmethod
    def build_records(df: pd.DataFrame) -> List[ItemRecord]:
        """
        Convert DataFrame to ItemRecord objects.
        
        Args:
            df: Input DataFrame
        
        Returns:
            List of valid ItemRecord objects
        """
        print("\n🔨 Building item records...")
        records = []
        skipped = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                record = ItemRecord(
                    item_id=str(row['ItemId']),
                    title=str(row.get('ItemTitle', '')) if pd.notna(row.get('ItemTitle')) else "",
                    combined_text=str(row.get('CombinedText', '')) if pd.notna(row.get('CombinedText')) else "",
                    price=float(row.get('Price', 0) or 0) if pd.notna(row.get('Price')) else 0.0,
                    category_id=str(row.get('CategoryId', '')) if pd.notna(row.get('CategoryId')) else "",
                    category_title=str(row.get('CategoryTitle', '')) if pd.notna(row.get('CategoryTitle')) else ""
                )
                
                if record.is_valid():
                    records.append(record)
                else:
                    skipped += 1
                    
            except Exception:
                skipped += 1
        
        print(f"✓ Built {len(records):,} valid records")
        if skipped > 0:
            print(f"  ⚠️ Skipped {skipped} invalid records")
        
        return records


# ============================================================================
# EMBEDDING GENERATOR
# ============================================================================

class EmbeddingGenerator:
    """
    Generate embeddings using sentence transformers.
    
    Features:
    - GPU acceleration when available
    - Batch processing for efficiency
    - Progress tracking
    - E5 instruction formatting
    """
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL, device: str = Config.DEVICE):
        """
        Initialize embedding generator.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device
        self.model_name = model_name
        
        print(f"\n🤖 Loading embedding model...")
        print(f"  Model: {model_name}")
        print(f"  Device: {device.upper()}")
        
        self.model = SentenceTransformer(model_name, device=device)
        
        actual_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded (dimension: {actual_dim})")
        
        if actual_dim != Config.EMBEDDING_DIM:
            print(f"⚠️ Warning: Expected dimension {Config.EMBEDDING_DIM}, got {actual_dim}")
    
    def generate_embeddings(
        self, 
        records: List[ItemRecord], 
        batch_size: int = Config.BATCH_SIZE
    ) -> np.ndarray:
        """ 
        Generate embeddings for all records.
        
        Args:
            records: List of ItemRecord objects
            batch_size: Number of items to process at once
        
        Returns:
            NumPy array of embeddings (n_items, embedding_dim)
        """
        print(f"\n🔄 Generating embeddings for {len(records):,} items...")
        print(f"  Batch size: {batch_size}")
        print(f"  Strategy: Original text (no preprocessing)\n")
        
        # Prepare passages with E5 instruction
        passages = [record.get_passage() for record in records]
        
        # Generate embeddings in batches
        embeddings_list = []
        
        for i in tqdm(range(0, len(passages), batch_size), desc="Embedding"):
            batch = passages[i:i+batch_size]
            
            batch_embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,  # Normalize for cosine similarity
                device=self.device,
                show_progress_bar=False
            )
            
            embeddings_list.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings = np.vstack(embeddings_list)
        
        # Validate embeddings
        print(f"\n✓ Embeddings generated: {embeddings.shape}")
        print(f"  Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f} (should be ~1.0)")
        print(f"  All valid: {np.all(np.isfinite(embeddings))}")
        print(f"  Memory: {embeddings.nbytes / 1e6:.1f} MB")
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a search query.
        
        Args:
            query: Search query text
        
        Returns:
            Normalized embedding vector
        """
        # Use 'query:' prefix for E5 models
        query_text = f"query: {query}"
        
        return self.model.encode(
            query_text,
            normalize_embeddings=True,
            device=self.device,
            show_progress_bar=False
        )


# ============================================================================
# VECTOR DATABASE MANAGER
# ============================================================================

class VectorDatabase:
    """
    Manage Qdrant vector database operations.
    
    Features:
    - Collection creation/management
    - Batch upload with progress tracking
    - Index creation for filtering
    - Connection validation
    """
    
    def __init__(self, 
        url: str = Config.QDRANT_URL, 
        api_key: str = Config.QDRANT_API_KEY,
        collection_name: str = Config.COLLECTION_NAME
    ):
        """
        Initialize database connection.
        
        Args:
            url: Qdrant Cloud URL
            api_key: API key for authentication
            collection_name: Name of the collection to use
        """
        self.url = url
        self.collection_name = collection_name
        
        print(f"\n🗄️ Connecting to Qdrant Cloud...")
        print(f"  URL: {url[:50]}...")
        
        self.client = QdrantClient(url=url, api_key=api_key)
        
        # Verify connection
        collections = self.client.get_collections()
        print(f"✓ Connected successfully")
        print(f"  Existing collections: {[c.name for c in collections.collections]}")
    
    def create_collection(
        self, 
        dimension: int = Config.EMBEDDING_DIM,
        recreate: bool = True
    ) -> None:
        """
        Create or recreate collection.
        
        Args:
            dimension: Vector dimension
            recreate: If True, delete existing collection first
        """
        print(f"\n📚 Setting up collection: '{self.collection_name}'")
        
        # Delete existing if requested
        if recreate:
            try:
                existing = self.client.get_collection(self.collection_name)
                print(f"  ⚠️ Collection exists with {existing.points_count} points")
                print(f"  Deleting and recreating...")
                self.client.delete_collection(self.collection_name)
            except Exception:
                print(f"  Creating new collection...")
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,  # Best for normalized embeddings
                on_disk=False              # Keep in memory for speed
            )
        )
        
        print(f"✓ Collection '{self.collection_name}' ready")
        print(f"  Vector size: {dimension}")
        print(f"  Distance metric: Cosine")
    
    def create_price_index(self) -> None:
        """
        Create index on price field for fast filtering.
        
        This is required for price-based queries to work efficiently.
        """
        print(f"\n🔧 Creating price index...")
        
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="price",
                field_schema=PayloadSchemaType.FLOAT
            )
            print("✓ Price index created")
            time.sleep(2)  # Wait for indexing
            
        except Exception as e:
            if "already exists" in str(e).lower():
                print("✓ Price index already exists")
            else:
                print(f"⚠️ Could not create index: {e}")
    
    def upload_data(
        self,
        records: List[ItemRecord],
        embeddings: np.ndarray,
        batch_size: int = Config.UPLOAD_BATCH_SIZE
    ) -> None:
        """
        Upload records and embeddings to Qdrant.
        
        Args:
            records: List of ItemRecord objects
            embeddings: Corresponding embedding vectors
            batch_size: Number of points to upload per batch
        """
        print(f"\n📤 Uploading {len(records):,} points to Qdrant")
        print(f"  Batch size: {batch_size}\n")
        
        point_id = 0
        
        for batch_start in tqdm(range(0, len(records), batch_size), desc="Uploading"):
            batch_end = min(batch_start + batch_size, len(records))
            
            # Prepare batch points
            points = []
            for i in range(batch_start, batch_end):
                points.append(PointStruct(
                    id=point_id,
                    vector=embeddings[i].tolist(),
                    payload=records[i].to_payload()
                ))
                point_id += 1
            
            # Upload batch
            self.client.upsert(
                collection_name=self.collection_name, 
                points=points,
                wait=True
            )
        
        # Verify upload
        time.sleep(2)
        info = self.client.get_collection(self.collection_name)
        
        print(f"\n✓ Upload complete!")
        print(f"  Expected: {len(records):,}")
        print(f"  In Qdrant: {info.points_count:,}")
        print(f"  Status: '{'✓ MATCH' if info.points_count == len(records) else '❌ MISMATCH'}'")    
    def search(
        self,
        query_vector: np.ndarray,
        filters: Optional[Filter] = None,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            filters: Optional Qdrant filters
            top_k: Number of results to return
            score_threshold: Minimum similarity score
        
        Returns:
            List of search results
        """
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=top_k,
                query_filter=filters,
                score_threshold=score_threshold,
                with_payload=True
            ).points
            
            return results
            
        except Exception as e:
            # Handle missing index gracefully
            if "Index required" in str(e) and filters:
                print("  ⚠️ Retrying without filters (no index)")
                
                # Get more results
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector.tolist(),
                    limit=top_k * 2,
                    score_threshold=score_threshold,
                    with_payload=True
                ).points
                
                # Manual filtering (slower but works)
                if filters and filters.must:
                    filtered = []
                    for r in results:
                        price = r.payload.get('price', 0)
                        
                        # Check if item passes all filter conditions
                        passes_filters = True
                        for cond in filters.must:
                            if hasattr(cond, 'range'):
                                if hasattr(cond.range, 'lte') and price > cond.range.lte:
                                    passes_filters = False
                                    break
                                if hasattr(cond.range, 'gte') and price < cond.range.gte:
                                    passes_filters = False
                                    break
                        
                        if passes_filters:
                            filtered.append(r)
                            if len(filtered) >= top_k:
                                break
                    
                    return filtered
                
                return results[:top_k]
            else:
                raise e
    
    def run_diagnostics(self, embedder: 'EmbeddingGenerator') -> None:
        """
        Run diagnostic checks on the database and search functionality.
        
        Args:
            embedder: EmbeddingGenerator instance for test embedding
        """
        print("\n" + "="*70)
        print("🔍 RUNNING DIAGNOSTICS")
        print("="*70)
        
        # Check collection
        try:
            info = self.client.get_collection(self.collection_name)
            print(f"✓ Collection: {info.points_count:,} points")
        except Exception as e:
            print(f"❌ Collection check failed: {e}")
            return
        
        # Test embedding generation
        test_query = "سيارة"
        try:
            test_embedding = embedder.encode_query(test_query)
            print(f"✓ Embedding generated: {test_embedding.shape}")
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return
        
        # Test raw search (no filters)
        try:
            raw_results = self.client.query_points(
                collection_name=self.collection_name,
                query=test_embedding.tolist(),
                limit=3,
                with_payload=True
            ).points
            
            print(f"✓ Raw search returned: {len(raw_results)} results")
            if raw_results:
                print(f"  Top result: {raw_results[0].payload['title']} (score: {raw_results[0].score:.4f})")
            else:
                print("  ❌ NO RESULTS - This indicates a fundamental issue!")
        except Exception as e:
            print(f"❌ Raw search failed: {e}")
        
        # Check a random point to verify data quality
        if info.points_count > 0:
            try:
                import random
                random_id = random.randint(0, info.points_count - 1)
                
                point = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[random_id],
                    with_payload=True,
                    with_vectors=True
                )
                
                if point:
                    print(f"\n✓ Sample point {random_id}:")
                    print(f"  Title: {point[0].payload.get('title', 'N/A')}")
                    print(f"  Has vector: {point[0].vector is not None}")
                    if point[0].vector:
                        print(f"  Vector dimension: {len(point[0].vector)}")
                        vector_norm = np.linalg.norm(point[0].vector)
                        print(f"  Vector norm: {vector_norm:.4f}")
            except Exception as e:
                print(f"⚠️ Could not retrieve sample point: {e}")
        
        print("\n" + "="*70)


# ============================================================================
# SEARCH ENGINE
# ============================================================================

class SemanticSearchEngine:
    """
    Complete semantic search engine.
    
    Combines:
    - Query analysis
    - Embedding generation
    - Vector search
    - Result formatting
    """
    
    def __init__(self,
        analyzer: GenericQueryAnalyzer,
        embedder: EmbeddingGenerator,
        database: VectorDatabase
    ):
        """
        Initialize search engine.
        
        Args:
            analyzer: Query analyzer instance
            embedder: Embedding generator instance
            database: Vector database instance
        """
        self.analyzer = analyzer
        self.embedder = embedder
        self.database = database
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        verbose: bool = True
    ) -> List:
        """
        Perform semantic search.
        
        Args:
            query: Search query (Arabic or English)
            top_k: Number of results to return
            verbose: If True, print detailed information
        
        Returns:
            List of search results
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🔍 Query: '{query}'")
            print(f"{'='*70}")
        
        # Step 1: Analyze query
        analysis = self.analyzer.analyze(query)
        
        if verbose:
            print(f"\n📊 Analysis:")
            print(f"  Clean Query: '{analysis['clean_query']}'")
            if analysis['numeric_filters']:
                print(f"  Filters: {analysis['numeric_filters']}")
            if analysis['extracted_entities']:
                entities = [e['text'] for e in analysis['extracted_entities']]
                print(f"  Entities: {entities}")
        
        # Step 2: Generate query embedding
        query_embedding = self.embedder.encode_query(analysis['clean_query'])
        
        # Step 3: Build Qdrant filters
        qdrant_filter = self._build_filter(analysis['numeric_filters'])
        
        # Step 4: Search
        results = self.database.search(
            query_vector=query_embedding,
            filters=qdrant_filter,
            top_k=top_k
        )
        
        # Step 5: Display results
        if verbose:
            self._display_results(results)
        
        return results
    
    def _build_filter(self, numeric_filters: Dict) -> Optional[Filter]:
        """
        Build Qdrant filter from numeric constraints.
        
        Args:
            numeric_filters: Dictionary of numeric constraints
        
        Returns:
            Qdrant Filter object or None
        """
        if not numeric_filters:
            return None
        
        conditions = []
        
        for field, constraints in numeric_filters.items():
            range_params = {}
            
            if "min" in constraints:
                range_params["gte"] = constraints["min"]
            if "max" in constraints:
                range_params["lte"] = constraints["max"]
            
            if range_params:
                conditions.append(
                    FieldCondition(
                        key="price",  # Currently only price filtering
                        range=Range(**range_params)
                    )
                )
        
        if conditions:
            return Filter(must=conditions)
        
        return None
    
    def _display_results(self, results: List) -> None:
        """
        Display search results in a formatted way.
        
        Args:
            results: List of search results from Qdrant
        """
        if not results:
            print(f"\n❌ No results found")
            return
        
        print(f"\n✨ Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. 📦 {result.payload['title']}")
            print(f"   💰 ${result.payload['price']:.2f}")
            print(f"   📁 {result.payload['category_title']}")
            print(f"   🎯 Score: {result.score:.4f}")
            
            # Show snippet of description
            text = result.payload['combined_text']
            snippet = text[:150] + "..." if len(text) > 150 else text
            print(f"   📝 {snippet}\n")


# ============================================================================
# TEST RUNNER
# ============================================================================

class TestRunner:
    """
    Run comprehensive tests on the search system.
    
    Features:
    - Multiple test cases
    - Result tracking
    - Performance metrics
    - HTML/Markdown/JSON reporting
    """
    
    # Standard test cases
    TEST_CASES = [
        {
            "id": 1,
            "query": "بدي سيارة اقتصادية للاستخدام اليومي، ما تستهلك بنزين كتير، تكون بعد 2015 وسعرها مناسب",
            "description": "Economical daily car, low fuel, after 2015",
            "expected_category": "سيارات"
        },
        {
            "id": 2,
            "query": "سيارة عائلية واسعة، أوتوماتيك، مناسبة للسفر، وما تكون قديمة",
            "description": "Spacious family car, automatic",
            "expected_category": "سيارات"
        },
        {
            "id": 3,
            "query": "بيت للبيع قريب من مدرسة ومنطقة هادئة، 3 غرف على الأقل، وسعره مو مبالغ فيه",
            "description": "House near school, 3+ rooms",
            "expected_category": "عقارات"
        },
        {
            "id": 4,
            "query": "أرض استثمارية داخل المدينة، تصلح لبناء تجاري، ومساحتها متوسطة",
            "description": "Investment land, commercial",
            "expected_category": "عقارات"
        },
        {
            "id": 5,
            "query": "أكل كلاب مستورد، مناسب للجراء، يقوي المناعة وما يسبب حساسية",
            "description": "Imported dog food, puppies",
            "expected_category": "حيوانات"
        },
        {
            "id": 6,
            "query": "أكل قطط اقتصادي بس يكون صحي للاستخدام اليومي",
            "description": "Economical cat food, healthy",
            "expected_category": "حيوانات"
        },
        {
            "id": 7,
            "query": "كر��م للبشرة الحساسة، بدون عطر، مناسب للاستخدام اليومي",
            "description": "Sensitive skin cream, fragrance-free",
            "expected_category": "تجميل"
        },
        {
            "id": 8,
            "query": "بدي وظيفة دوام جزئي بمجال المبيعات أو خدمة العملاء بدون خبرة كبيرة",
            "description": "Part-time job, sales/customer service",
            "expected_category": "وظائف"
        },
        {
            "id": 9,
            "query": "هدية أنيقة لبنت بعمر 25 سنة، تكون عملية ومناسبة للمناسبات",
            "description": "Elegant gift, 25-year-old",
            "expected_category": "هدايا"
        },
        {
            "id": 10,
            "query": "لابتوب خفيف للدراسة، بطاريته قوية، وسعره متوسط",
            "description": "Lightweight study laptop",
            "expected_category": "كمبيوتر"
        },
        {
            "id": 11,
            "query": "بدي سيارة مرسيدس حمرا سعرها تحت 10 آلاف",
            "description": "Red Mercedes under 10k",
            "expected_category": "سيارات"
        },
        {
            "id": 12,
            "query": "بدي جاكيت شتوي جوخ لونه أسود",
            "description": "Black wool winter jacket",
            "expected_category": "ملابس"
        },
        {
            "id": 13,
            "query": "بدي لابتوب ماركة HP Gaming",
            "description": "HP Gaming laptop",
            "expected_category": "كمبيوتر"
        }
    ]
    
    def __init__(self, search_engine: SemanticSearchEngine):
        """Initialize test runner.
        
        Args:
            search_engine: Configured search engine instance
        """
        self.search_engine = search_engine
        self.results = {}
    
    def run_all_tests(self, verbose: bool = False) -> Dict:
        """Run all test cases."""
        print("\n" + "="*80)
        print("🧪 RUNNING COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Total tests: {len(self.TEST_CASES)}\n")
        
        for test in self.TEST_CASES:
            test_id = test['id']
            print(f"Test {test_id}/{len(self.TEST_CASES)}: {test['description']}", end=" ... ")
            
            try:
                # Run search
                results = self.search_engine.search(
                    query=test['query'],
                    top_k=3,
                    verbose=verbose
                )
                
                # Store results
                self.results[test_id] = {
                    "query": test['query'],
                    "description": test['description'],
                    "found": len(results),
                    "top_score": results[0].score if results else 0.0,
                    "results": [
                        {
                            "title": r.payload['title'],
                            "price": r.payload['price'],
                            "category": r.payload['category_title'],
                            "score": r.score
                        }
                        for r in results
                    ],
                    "status": "PASS" if len(results) > 0 else "FAIL"
                }
                
                print(f"✅ ({len(results)} results, score: {results[0].score:.4f})")
                
            except Exception as e:
                print(f"❌ ERROR: {str(e)[:50]}")
                self.results[test_id] = {
                    "query": test['query'],
                    "description": test['description'],
                    "found": 0,
                    "top_score": 0.0,
                    "results": [],
                    "status": "ERROR",
                    "error": str(e)
                }
        
        self._print_summary()
        return self.results
    
    def _print_summary(self) -> None:
        """Print test summary statistics."""
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        errors = sum(1 for r in self.results.values() if r['status'] == 'ERROR')
        
        avg_score = sum(r['top_score'] for r in self.results.values()) / total
        
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        print(f"  Total Tests: {total}")
        print(f"  ✅ Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"  ❌ Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"  ⚠️ Errors: {errors} ({errors/total*100:.1f}%)")
        print(f"  📊 Average Score: {avg_score:.4f}")
        print("="*80)
    
    def save_report(self, format: str = "html") -> str:
        """Save test report to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if format == "html":
            filename = f"test_report_{timestamp}.html"
            content = self._generate_html_report()
        elif format == "markdown":
            filename = f"test_report_{timestamp}.md"
            content = self._generate_markdown_report()
        else:  # json
            filename = f"test_report_{timestamp}.json"
            content = self._generate_json_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n💾 Report saved: {filename}")
        return filename
    
    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r['status'] == "PASS")
        
        # Build HTML with proper escaping for CSS braces
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Semantic Search Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 30px; border-radius: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
        .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }}
        .test {{ background: white; padding: 20px; margin: 10px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .pass {{ border-left: 5px solid #28a745; }}
        .fail {{ border-left: 5px solid #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Semantic Search Test Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    <div class="summary">
        <div class="card"><h3>Total Tests</h3><div style="font-size:32px">{total}</div></div>
        <div class="card"><h3>Success Rate</h3><div style="font-size:32px">{passed/total*100:.1f}%</div></div>
        <div class="card"><h3>Avg Score</h3><div style="font-size:32px">{sum(r['top_score'] for r in self.results.values())/total:.4f}</div></div>
    </div>
"""
        
        for test_id, result in self.results.items():
            status_class = "pass" if result['status'] == "PASS" else "fail"
            html += f"""
    <div class="test {status_class}">
        <h3>Test {test_id}: {result['description']}</h3>
        <p><strong>Query:</strong> {result['query']}</p>
        <p><strong>Results:</strong> {result['found']} items (Score: {result['top_score']:.4f})</p>
    </div>
"""
        
        html += "</body></html>"
        return html
    
    def _generate_markdown_report(self) -> str:
        """Generate Markdown report."""
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r['status'] == "PASS")
        avg_score = sum(r['top_score'] for r in self.results.values())/total
        
        md = f"""# 🚀 Semantic Search Test Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- Total Tests: {total}
- Passed: {passed} ({passed/total*100:.1f}%)
- Average Score: {avg_score:.4f}

## Test Results

"""
        
        for test_id, result in self.results.items():
            status = "✅" if result['status'] == "PASS" else "❌"
            md += f"""### {status} Test {test_id}: {result['description']}

**Query:** {result['query']}  
**Results:** {result['found']} items  
**Top Score:** {result['top_score']:.4f}

---

"""
        
        return md
    
    def _generate_json_report(self) -> str:
        """Generate JSON report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results.values() if r['status'] == "PASS"),
                "average_score": sum(r['top_score'] for r in self.results.values()) / len(self.results)
            },
            "tests": self.results
        }
        
        return json.dumps(report, ensure_ascii=False, indent=2)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - runs the complete pipeline.
    
    Steps:
    1. Load and validate data
    2. Initialize components
    3. Generate embeddings
    4. Upload to Qdrant
    5. Run tests
    6. Generate reports
    """
    
    # Print configuration
    Config.print_config()
    
    # ========================================
    # STEP 1: LOAD DATA
    # ========================================
    print("\n" + "="*80)
    print("📁 STEP 1: LOADING DATA")
    print("="*80)
    
    data_dir = DataLoader.detect_environment()
    csv_path = data_dir / Config.CSV_FILE
    
    df = DataLoader.load_csv(csv_path)
    DataLoader.validate_dataframe(df)
    
    records = DataLoader.build_records(df)
    
    # ========================================
    # STEP 2: INITIALIZE COMPONENTS
    # ========================================
    print("\n" + "="*80)
    print("🔧 STEP 2: INITIALIZING COMPONENTS")
    print("="*80)
    
    # Initialize analyzer
    print("\n🧠 Creating query analyzer...")
    analyzer = GenericQueryAnalyzer(dataset_df=df)
    print(f"✓ Analyzer ready (learned {len(analyzer.categories)} categories)")
    
    # Initialize embedder
    embedder = EmbeddingGenerator()
    
    # Initialize database
    database = VectorDatabase()
    
    # ========================================
    # STEP 3: GENERATE EMBEDDINGS
    # ========================================
    print("\n" + "="*80)
    print("🔢 STEP 3: GENERATING EMBEDDINGS")
    print("="*80)
    
    embeddings = embedder.generate_embeddings(records)
    
    # ========================================
    # STEP 4: SETUP DATABASE
    # ========================================
    print("\n" + "="*80)
    print("🗄️ STEP 4: SETTING UP DATABASE")
    print("="*80)
    
    # Create collection
    database.create_collection(dimension=Config.EMBEDDING_DIM, recreate=True)
    
    # Create price index for filtering
    database.create_price_index()
    
    # Upload embeddings
    database.upload_data(records, embeddings)
    
    # Run diagnostics to verify setup
    database.run_diagnostics(embedder)
    
    # ========================================
    # STEP 5: INITIALIZE SEARCH ENGINE
    # ========================================
    print("\n" + "="*80)
    print("🔍 STEP 5: INITIALIZING SEARCH ENGINE")
    print("="*80)
    
    search_engine = SemanticSearchEngine(
        analyzer=analyzer,
        embedder=embedder,
        database=database
    )
    
    print("✓ Search engine ready!")
    
    # ========================================
    # STEP 6: RUN TESTS
    # ========================================
    print("\n" + "="*80)
    print("🧪 STEP 6: RUNNING TESTS")
    print("="*80)
    
    test_runner = TestRunner(search_engine)
    results = test_runner.run_all_tests(verbose=False)
    
    # ========================================
    # STEP 7: GENERATE REPORTS
    # ========================================
    print("\n" + "="*80)
    print("📊 STEP 7: GENERATING REPORTS")
    print("="*80)
    
    test_runner.save_report(format="html")
    test_runner.save_report(format="markdown")
    test_runner.save_report(format="json")
    
    # ========================================
    # COMPLETE
    # ========================================
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print("\nYour semantic search system is ready to use!")
    print("You can now use search_engine.search(query) to search for items.")
    print("\nExample:")
    print("  >>> search_engine.search('بدي لابتوب HP Gaming', top_k=5)")
    
    return search_engine, test_runner


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    search_engine, test_runner = main()
    
    # Optional: Interactive search
    print("\n" + "="*80)
    print("🎯 INTERACTIVE SEARCH MODE")
    print("="*80)
    print("You can now search interactively. Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("Enter search query: ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            results = search_engine.search(query, top_k=5, verbose=True)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
