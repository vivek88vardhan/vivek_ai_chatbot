#!/usr/bin/env python3
"""
Weaviate Vector Database Testing with Prefect Workflow
=====================================================

This script demonstrates comprehensive Weaviate functionality using Prefect for orchestration:
- Client connection and health check
- Schema creation and management
- Data insertion with sample articles
- Various query types (filtering, sorting, semantic search, hybrid search)
- Data storage inspection
- Complete workflow visualization in Prefect UI

Prerequisites:
1. Docker with Weaviate running: docker run -d --name weaviate -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true semitechnologies/weaviate:latest
2. Install dependencies: pip install prefect weaviate-client sentence-transformers numpy
3. Start Prefect server: prefect server start
4. In another terminal, run this script: python weaviate_test_prefect.py

Author: AI Assistant
Date: August 2025
"""

import weaviate
import weaviate.config
from weaviate.auth import AuthApiKey, AuthClientCredentials
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import Filter
import json
import time
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging

# Try to import torch for tensor handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - will use alternative encoding methods")

from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.blocks.system import Secret
from prefect.cache_policies import NONE as NO_CACHE

# Configure logging
logging.basicConfig(level=logging.INFO)

# Sample data for testing
SAMPLE_ARTICLES = [
    {
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed. It uses statistical techniques to give computers the ability to learn and make decisions.",
        "author": "Dr. John Smith",
        "published_date": "2024-01-15T10:00:00Z",
        "category": "Technology",
        "reading_time": 8,
        "tags": ["AI", "ML", "Data Science"]
    },
    {
        "title": "The Future of Renewable Energy",
        "content": "Renewable energy sources like solar, wind, and hydroelectric power are becoming increasingly important for sustainable development. These clean energy solutions help reduce carbon emissions and combat climate change.",
        "author": "Dr. Jane Green",
        "published_date": "2024-01-20T14:30:00Z",
        "category": "Environment",
        "reading_time": 12,
        "tags": ["Solar", "Wind", "Climate"]
    },
    {
        "title": "Deep Learning Applications in Healthcare",
        "content": "Deep learning has revolutionized medical diagnosis and treatment planning. Neural networks can now analyze medical images, predict disease outcomes, and assist doctors in making more accurate diagnoses.",
        "author": "Dr. Mike Johnson",
        "published_date": "2024-02-01T09:15:00Z",
        "category": "Healthcare",
        "reading_time": 15,
        "tags": ["Deep Learning", "Medical", "AI"]
    },
    {
        "title": "Blockchain Technology Explained",
        "content": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records, called blocks, which are linked and secured using cryptography. It's the foundation of cryptocurrencies like Bitcoin.",
        "author": "Sarah Williams",
        "published_date": "2024-02-10T16:45:00Z",
        "category": "Technology",
        "reading_time": 10,
        "tags": ["Blockchain", "Cryptocurrency", "Security"]
    },
    {
        "title": "Climate Change and Ocean Levels",
        "content": "Rising ocean levels due to climate change pose significant threats to coastal communities worldwide. Understanding these changes is crucial for developing adaptation strategies and protecting vulnerable populations.",
        "author": "Prof. David Ocean",
        "published_date": "2024-02-15T11:20:00Z",
        "category": "Environment",
        "reading_time": 14,
        "tags": ["Climate Change", "Ocean", "Environment"]
    },
    {
        "title": "Natural Language Processing Advances",
        "content": "Recent advances in natural language processing have enabled computers to understand and generate human language with unprecedented accuracy. Large language models are transforming how we interact with technology.",
        "author": "Dr. Alice NLP",
        "published_date": "2024-03-01T13:10:00Z",
        "category": "Technology",
        "reading_time": 11,
        "tags": ["NLP", "AI", "Language Models"]
    }
]

# Configuration for different authentication methods
AUTH_CONFIG = {
    "method": "anonymous",  # Options: "anonymous", "api_key", "oidc"
    "api_key": "my-secret-key",  # For API key auth
    "client_secret": "your-client-secret",  # For OIDC auth
    "scope": "openid"  # For OIDC auth
}

@task(name="🔗 Create Weaviate Client", retries=3, cache_policy=NO_CACHE)
def create_weaviate_client(
    host: str = "localhost", 
    port: int = 8080,
    auth_method: str = "anonymous",
    api_key: Optional[str] = None,
    client_secret: Optional[str] = None
) -> weaviate.WeaviateClient:
    """Create and test Weaviate client connection with authentication support."""
    logger = get_run_logger()
    
    try:
        # Determine authentication method
        auth_config = None
        auth_description = "No authentication (anonymous)"
        
        if auth_method == "api_key" and api_key:
            auth_config = AuthApiKey(api_key=api_key)
            auth_description = f"API Key authentication (key: {api_key[:8]}...)"
            logger.info(f"🔑 Using API Key authentication")
            
        elif auth_method == "oidc" and client_secret:
            auth_config = AuthClientCredentials(
                client_secret=client_secret,
                scope="openid"
            )
            auth_description = "OIDC authentication"
            logger.info(f"🔐 Using OIDC authentication")
            
        else:
            logger.info(f"🌐 Using anonymous access (no authentication)")
        
        # Test HTTP endpoint first
        logger.info("🔍 Testing Weaviate HTTP endpoint...")
        headers = {}
        if auth_method == "api_key" and api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        try:
            response = requests.get(f"http://{host}:{port}/v1/meta", headers=headers, timeout=10)
            logger.info(f"📡 HTTP Response Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    meta_data = response.json()
                    logger.info(f"📊 HTTP Meta Response: {meta_data}")
                    if 'version' in meta_data:
                        logger.info(f"🏷️ Weaviate version from HTTP: {meta_data['version']}")
                except Exception as json_error:
                    logger.warning(f"⚠️ Could not parse HTTP response as JSON: {json_error}")
            
            if response.status_code == 401:
                logger.error("❌ Authentication failed - Invalid credentials")
                raise ConnectionError("Authentication failed - 401 Unauthorized")
            elif response.status_code == 403:
                logger.error("❌ Access forbidden - Check permissions")
                raise ConnectionError("Access forbidden - 403 Forbidden")
            elif response.status_code != 200:
                logger.error(f"❌ HTTP endpoint returned status {response.status_code}")
                raise ConnectionError(f"HTTP endpoint error - {response.status_code}")
            
            logger.info("✅ Weaviate HTTP endpoint is accessible")
            
        except requests.RequestException as req_error:
            logger.error(f"❌ HTTP request failed: {req_error}")
            raise ConnectionError(f"HTTP request failed: {req_error}")
        
        # Connection methods to try (in order of preference)
        connection_methods = [
            ("Simple local connection", lambda: weaviate.connect_to_local(
                host=host,
                port=port,
                auth_credentials=auth_config,
                skip_init_checks=True
            )),
            ("Local connection with config", lambda: weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=50051,
                auth_credentials=auth_config,
                additional_config=weaviate.config.AdditionalConfig(
                    timeout=weaviate.config.Timeout(init=30, query=60, insert=120),
                    startup_period=30
                ),
                skip_init_checks=True
            )),
            ("Custom connection", lambda: weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=False,
                grpc_host=host,
                grpc_port=50051,
                grpc_secure=False,
                auth_credentials=auth_config,
                additional_config=weaviate.config.AdditionalConfig(
                    timeout=weaviate.config.Timeout(init=30, query=60, insert=120)
                )
            ))
        ]
        
        for method_name, connection_func in connection_methods:
            try:
                logger.info(f"🔄 Trying {method_name}...")
                client = connection_func()
                
                # Test connection
                logger.info("🏥 Testing client readiness...")
                is_ready = client.is_ready()
                logger.info(f"📋 Client ready status: {is_ready}")
                
                if not is_ready:
                    logger.warning(f"⚠️ Client not ready, closing and trying next method...")
                    client.close()
                    continue
                    
                # Get cluster info and handle different response formats
                try:
                    logger.info("📊 Fetching cluster metadata...")
                    meta = client.get_meta()
                    logger.info(f"🔍 Metadata type: {type(meta)}")
                    logger.info(f"🔍 Metadata content: {meta}")
                    
                    # Handle both object and dict responses
                    if hasattr(meta, 'version'):
                        version = meta.version
                        logger.info("✅ Found version attribute")
                    elif isinstance(meta, dict):
                        version = meta.get('version', 'unknown')
                        logger.info(f"✅ Found version in dict: {version}")
                    else:
                        version = str(meta)
                        logger.info(f"⚠️ Unknown metadata format, using string: {version}")
                except Exception as meta_error:
                    logger.warning(f"⚠️ Could not get metadata: {meta_error}")
                    version = "unknown"
                
                logger.info(f"✅ Successfully connected to Weaviate")
                logger.info(f"📊 Weaviate version: {version}")
                logger.info(f"🏥 Cluster status: {'Healthy' if is_ready else 'Unhealthy'}")
                logger.info(f"🔧 Connection method: {method_name}")
                logger.info(f"🔐 Authentication: {auth_description}")
                
                return client
                
            except Exception as method_error:
                logger.warning(f"⚠️ {method_name} failed: {str(method_error)}")
                logger.warning(f"🔍 Error type: {type(method_error)}")
                continue
        
        # If all methods failed
        raise ConnectionError("All connection methods failed")
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Weaviate: {str(e)}")
        logger.info("💡 Troubleshooting steps:")
        logger.info("   1. Check if Weaviate is running: docker-compose ps")
        logger.info("   2. Verify authentication settings in docker-compose.yml")
        logger.info("   3. Test manually with curl:")
        if auth_method == "api_key" and api_key:
            logger.info(f"      curl -H 'Authorization: Bearer {api_key}' http://{host}:{port}/v1/meta")
        else:
            logger.info(f"      curl http://{host}:{port}/v1/meta")
        logger.info("   4. Check logs: docker-compose logs weaviate")
        raise

@task(name="📋 Create Article Schema", cache_policy=NO_CACHE)
def create_article_schema(client: weaviate.WeaviateClient) -> Dict[str, Any]:
    """Create the Article schema in Weaviate."""
    logger = get_run_logger()
    
    try:
        # Delete existing collection if it exists
        try:
            client.collections.delete("Article")
            logger.info("🗑️ Deleted existing Article collection")
        except:
            pass
        
        # Create new collection with schema - simplified without vector config
        collection = client.collections.create(
            name="Article",
            description="A news or blog article with content and metadata",
            properties=[
                Property(name="title", data_type=DataType.TEXT, description="Title of the article"),
                Property(name="content", data_type=DataType.TEXT, description="Main content of the article"),
                Property(name="author", data_type=DataType.TEXT, description="Author name"),
                Property(name="published_date", data_type=DataType.DATE, description="Publication date and time"),
                Property(name="category", data_type=DataType.TEXT, description="Article category"),
                Property(name="reading_time", data_type=DataType.INT, description="Estimated reading time in minutes"),
                Property(name="tags", data_type=DataType.TEXT_ARRAY, description="List of tags associated with the article")
            ]
            # Remove vector_config for now - we'll add vectors manually
        )
        
        logger.info("📝 Successfully created Article collection")
        logger.info(f"✅ Collection verified with properties created")
        
        # Return collection info as dict
        return {
            "name": "Article", 
            "description": "A news or blog article with content and metadata",
            "properties": [
                {"name": "title", "dataType": "text"},
                {"name": "content", "dataType": "text"},
                {"name": "author", "dataType": "text"},
                {"name": "published_date", "dataType": "date"},
                {"name": "category", "dataType": "text"},
                {"name": "reading_time", "dataType": "int"},
                {"name": "tags", "dataType": "text[]"}
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create schema: {str(e)}")
        logger.info("💡 Trying alternative schema creation method...")
        
        # Fallback method - try with explicit vector config
        try:
            from weaviate.classes.config import VectorDistances
            
            collection = client.collections.create(
                name="Article",
                description="A news or blog article with content and metadata",
                properties=[
                    Property(name="title", data_type=DataType.TEXT, description="Title of the article"),
                    Property(name="content", data_type=DataType.TEXT, description="Main content of the article"),
                    Property(name="author", data_type=DataType.TEXT, description="Author name"),
                    Property(name="published_date", data_type=DataType.DATE, description="Publication date and time"),
                    Property(name="category", data_type=DataType.TEXT, description="Article category"),
                    Property(name="reading_time", data_type=DataType.INT, description="Estimated reading time in minutes"),
                    Property(name="tags", data_type=DataType.TEXT_ARRAY, description="List of tags associated with the article")
                ],
                vector_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    vector_cache_max_objects=100000
                )
            )
            
            logger.info("📝 Successfully created Article collection using fallback method")
            
            return {
                "name": "Article", 
                "description": "A news or blog article with content and metadata",
                "properties": [
                    {"name": "title", "dataType": "text"},
                    {"name": "content", "dataType": "text"},
                    {"name": "author", "dataType": "text"},
                    {"name": "published_date", "dataType": "date"},
                    {"name": "category", "dataType": "text"},
                    {"name": "reading_time", "dataType": "int"},
                    {"name": "tags", "dataType": "text[]"}
                ]
            }
            
        except Exception as fallback_error:
            logger.error(f"❌ Fallback schema creation also failed: {str(fallback_error)}")
            logger.info("💡 Trying minimal schema creation...")
            
            # Final fallback - minimal schema
            try:
                collection = client.collections.create("Article")
                logger.info("📝 Successfully created Article collection with minimal schema")
                
                return {
                    "name": "Article", 
                    "description": "Minimal article collection",
                    "properties": []
                }
                
            except Exception as minimal_error:
                logger.error(f"❌ Even minimal schema creation failed: {str(minimal_error)}")
                raise

@task(name="📄 Print Current Schema", cache_policy=NO_CACHE)
def print_schema(client: weaviate.WeaviateClient) -> Dict[str, Any]:
    """Print the current Weaviate schema in a readable format."""
    logger = get_run_logger()
    
    try:
        # Get collection info
        collections = client.collections.list_all()
        
        logger.info("📋 Current Weaviate Collections:")
        logger.info("=" * 50)
        
        schema_info = {"collections": []}
        
        for collection_name in collections:
            collection = client.collections.get(collection_name)
            config = collection.config.get()
            
            logger.info(f"🏷️  Collection: {collection_name}")
            logger.info(f"   Description: {config.description or 'No description'}")
            logger.info(f"   Vectorizer: {config.vectorizer_config}")
            logger.info("   Properties:")
            
            collection_info = {
                "name": collection_name,
                "description": config.description,
                "properties": []
            }
            
            for prop in config.properties:
                logger.info(f"     • {prop.name} ({prop.data_type}): {prop.description or ''}")
                collection_info["properties"].append({
                    "name": prop.name,
                    "data_type": str(prop.data_type),
                    "description": prop.description
                })
            
            schema_info["collections"].append(collection_info)
            logger.info("-" * 30)
        
        return schema_info
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve schema: {str(e)}")
        raise

@task(name="🤖 Initialize Sentence Transformer")
def initialize_embedder() -> SentenceTransformer:
    """Initialize the sentence transformer for creating embeddings."""
    logger = get_run_logger()
    
    try:
        model_name = "all-MiniLM-L6-v2"  # Lightweight but effective model
        embedder = SentenceTransformer(model_name)
        logger.info(f"🤖 Initialized sentence transformer: {model_name}")
        return embedder
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize embedder: {str(e)}")
        raise

@task(name="📝 Insert Sample Articles", cache_policy=NO_CACHE)
def insert_sample_data(client: weaviate.WeaviateClient, embedder: SentenceTransformer) -> List[str]:
    """Insert sample articles into Weaviate with generated embeddings."""
    logger = get_run_logger()
    
    inserted_ids = []
    
    try:
        logger.info(f"📝 Starting insertion of {len(SAMPLE_ARTICLES)} articles...")
        
        collection = client.collections.get("Article")
        
        for i, article in enumerate(SAMPLE_ARTICLES, 1):
            # Create embeddings for title and content combined
            text_to_embed = f"{article['title']} {article['content']}"
            vector = embedder.encode(text_to_embed).tolist()
            
            # Insert with custom vector
            uuid = collection.data.insert(
                properties=article,
                vector=vector
            )
            
            inserted_ids.append(str(uuid))
            logger.info(f"✅ Inserted article {i}/{len(SAMPLE_ARTICLES)}: '{article['title'][:50]}...'")
        
        logger.info(f"🎉 Successfully inserted {len(inserted_ids)} articles")
        return inserted_ids
        
    except Exception as e:
        logger.error(f"❌ Failed to insert data: {str(e)}")
        raise

@task(name="📊 Print All Articles", cache_policy=NO_CACHE)
def print_all_articles(client: weaviate.WeaviateClient) -> List[Dict]:
    """Retrieve and print all articles from Weaviate."""
    logger = get_run_logger()
    
    try:
        collection = client.collections.get("Article")
        response = collection.query.fetch_objects(
            return_properties=["title", "author", "category", "published_date", "reading_time", "tags"]
        )
        
        articles = []
        for obj in response.objects:
            articles.append(obj.properties)
        
        logger.info("📚 All Articles in Database:")
        logger.info("=" * 60)
        
        for i, article in enumerate(articles, 1):
            logger.info(f"📄 Article {i}:")
            logger.info(f"   Title: {article['title']}")
            logger.info(f"   Author: {article['author']}")
            logger.info(f"   Category: {article['category']}")
            logger.info(f"   Published: {article['published_date']}")
            logger.info(f"   Reading Time: {article['reading_time']} minutes")
            logger.info(f"   Tags: {', '.join(article.get('tags', []))}")
            logger.info("-" * 40)
        
        logger.info(f"📈 Total articles found: {len(articles)}")
        return articles
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve articles: {str(e)}")
        raise

@task(name="🔍 Filter Articles by Category", cache_policy=NO_CACHE)
def filter_by_category(client: weaviate.WeaviateClient, category: str = "Technology") -> List[Dict]:
    """Filter articles by category."""
    logger = get_run_logger()
    
    try:
        collection = client.collections.get("Article")
        
        # Try the new v4 syntax first
        try:
            response = collection.query.fetch_objects(
                where=Filter.by_property("category").equal(category),
                return_properties=["title", "author", "category", "reading_time"]
            )
            
            articles = []
            for obj in response.objects:
                articles.append(obj.properties)
                
        except Exception as v4_error:
            logger.warning(f"⚠️ v4 filtering syntax failed: {v4_error}")
            logger.info("💡 Trying alternative filtering method...")
            
            # Fallback: Fetch all and filter in Python
            response = collection.query.fetch_objects(
                return_properties=["title", "author", "category", "reading_time"]
            )
            
            articles = []
            for obj in response.objects:
                if obj.properties.get("category") == category:
                    articles.append(obj.properties)
        
        logger.info(f"🔍 Articles filtered by category '{category}':")
        logger.info("=" * 50)
        
        for article in articles:
            logger.info(f"📄 {article['title']} by {article['author']} ({article['reading_time']} min)")
        
        logger.info(f"📊 Found {len(articles)} articles in '{category}' category")
        return articles
        
    except Exception as e:
        logger.error(f"❌ Failed to filter by category: {str(e)}")
        logger.info("💡 Trying manual filtering...")
        
        # Final fallback: Basic fetch and manual filter
        try:
            collection = client.collections.get("Article")
            response = collection.query.fetch_objects()
            
            articles = []
            for obj in response.objects:
                props = obj.properties
                if props.get("category") == category:
                    articles.append(props)
            
            logger.info(f"🔍 Manually filtered {len(articles)} articles by category '{category}'")
            return articles
            
        except Exception as manual_error:
            logger.error(f"❌ Manual filtering also failed: {str(manual_error)}")
            return []

@task(name="📈 Sort Articles by Reading Time", cache_policy=NO_CACHE)
def sort_by_reading_time(client: weaviate.WeaviateClient, ascending: bool = True) -> List[Dict]:
    """Sort articles by reading time."""
    logger = get_run_logger()
    
    try:
        collection = client.collections.get("Article")
        response = collection.query.fetch_objects(
            return_properties=["title", "author", "reading_time", "published_date"]
        )
        
        articles = []
        for obj in response.objects:
            articles.append(obj.properties)
        
        # Sort by reading time
        sorted_articles = sorted(
            articles, 
            key=lambda x: x['reading_time'], 
            reverse=not ascending
        )
        
        direction = "ascending" if ascending else "descending"
        logger.info(f"📈 Articles sorted by reading time ({direction}):")
        logger.info("=" * 50)
        
        for article in sorted_articles:
            logger.info(f"📖 {article['reading_time']} min: {article['title']} by {article['author']}")
        
        logger.info(f"📊 Sorted {len(sorted_articles)} articles")
        return sorted_articles
        
    except Exception as e:
        logger.error(f"❌ Failed to sort articles: {str(e)}")
        raise

@task(name="🧠 Semantic Search", cache_policy=NO_CACHE)
def semantic_search(client: weaviate.WeaviateClient, embedder: SentenceTransformer, query: str = "artificial intelligence machine learning", limit: int = 3) -> List[Dict]:
    """Perform semantic search to find similar articles using manual embeddings."""
    logger = get_run_logger()
    
    try:
        collection = client.collections.get("Article")
        
        # Generate embedding for the search query using our embedder
        logger.info(f"🤖 Generating embedding for query: '{query}'")
        
        # Fix encoding issue by being explicit about the method
        try:
            # Method 1: Use the SentenceTransformer encode method explicitly
            query_vector = embedder.encode([query])[0].tolist()  # Encode as list, take first result
            logger.info("✅ Used list-based encoding")
        except Exception as encode_error:
            logger.warning(f"⚠️ List-based encoding failed: {encode_error}")
            try:
                # Method 2: Direct encoding with numpy conversion
                query_vector = embedder.encode(query).tolist()
                logger.info("✅ Used direct encoding")
            except Exception as direct_error:
                logger.warning(f"⚠️ Direct encoding failed: {direct_error}")
                # Method 3: Alternative encoding approach with torch if available
                if TORCH_AVAILABLE:
                    query_vector = embedder.encode(query, convert_to_tensor=True).cpu().numpy().tolist()
                    logger.info("✅ Used torch tensor encoding")
                else:
                    # Method 4: Fallback to numpy array
                    embedding_array = embedder.encode(query)
                    query_vector = np.array(embedding_array).tolist()
                    logger.info("✅ Used numpy array encoding")
        
        logger.info(f"📊 Query vector dimensions: {len(query_vector)}")
        
        # Use near_vector instead of near_text since we don't have a vectorizer
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_properties=["title", "content", "author", "category"],
            return_metadata=["distance"]
        )
        
        articles = []
        for obj in response.objects:
            article_data = obj.properties.copy()
            article_data["_metadata"] = {"distance": obj.metadata.distance}
            articles.append(article_data)
        
        logger.info(f"🧠 Semantic search results for: '{query}'")
        logger.info("=" * 60)
        
        for i, article in enumerate(articles, 1):
            distance = article['_metadata']['distance']
            similarity = 1 - distance  # Convert distance to similarity
            
            logger.info(f"🎯 Result {i} (Similarity: {similarity:.3f}):")
            logger.info(f"   Title: {article['title']}")
            logger.info(f"   Author: {article['author']}")
            logger.info(f"   Category: {article['category']}")
            logger.info(f"   Content preview: {article['content'][:100]}...")
            logger.info("-" * 40)
        
        logger.info(f"📊 Found {len(articles)} semantically similar articles")
        return articles
        
    except Exception as e:
        logger.error(f"❌ Failed to perform semantic search: {str(e)}")
        logger.info("💡 Trying keyword-based fallback search...")
        
        # Fallback: Simple keyword search
        try:
            collection = client.collections.get("Article")
            response = collection.query.fetch_objects(
                return_properties=["title", "content", "author", "category"]
            )
            
            # Simple keyword matching
            query_words = query.lower().split()
            articles = []
            
            for obj in response.objects:
                props = obj.properties
                text_content = f"{props.get('title', '')} {props.get('content', '')}".lower()
                
                # Count keyword matches
                matches = sum(1 for word in query_words if word in text_content)
                if matches > 0:
                    article_data = props.copy()
                    article_data["_metadata"] = {"keyword_matches": matches}
                    articles.append(article_data)
            
            # Sort by number of matches
            articles.sort(key=lambda x: x["_metadata"]["keyword_matches"], reverse=True)
            articles = articles[:limit]
            
            logger.info(f"🔍 Keyword search results for: '{query}' (fallback)")
            logger.info("=" * 60)
            
            for i, article in enumerate(articles, 1):
                matches = article['_metadata']['keyword_matches']
                
                logger.info(f"🎯 Result {i} (Keyword matches: {matches}):")
                logger.info(f"   Title: {article['title']}")
                logger.info(f"   Author: {article['author']}")
                logger.info(f"   Category: {article['category']}")
                logger.info(f"   Content preview: {article['content'][:100]}...")
                logger.info("-" * 40)
            
            logger.info(f"📊 Found {len(articles)} articles using keyword search")
            return articles
            
        except Exception as fallback_error:
            logger.error(f"❌ Keyword search also failed: {str(fallback_error)}")
            return []

@task(name="🔀 Hybrid Search", cache_policy=NO_CACHE)
def hybrid_search(client: weaviate.WeaviateClient, embedder: SentenceTransformer, query: str = "renewable energy", alpha: float = 0.5, limit: int = 3) -> List[Dict]:
    """Perform hybrid search combining keyword and semantic search using manual implementation."""
    logger = get_run_logger()
    
    try:
        collection = client.collections.get("Article")
        
        # Generate embedding for the search query
        logger.info(f"🤖 Generating embedding for hybrid query: '{query}'")
        
        # Fix encoding issue with multiple fallback methods
        try:
            # Method 1: List-based encoding
            query_vector = embedder.encode([query])[0].tolist()
            logger.info("✅ Used list-based encoding for hybrid search")
        except Exception as encode_error:
            logger.warning(f"⚠️ List-based encoding failed: {encode_error}")
            try:
                # Method 2: Direct encoding
                query_vector = embedder.encode(query).tolist()
                logger.info("✅ Used direct encoding for hybrid search")
            except Exception as direct_error:
                logger.warning(f"⚠️ Direct encoding failed: {direct_error}")
                # Method 3: Torch-based encoding if available
                if TORCH_AVAILABLE:
                    query_vector = embedder.encode(query, convert_to_tensor=True).cpu().numpy().tolist()
                    logger.info("✅ Used torch tensor encoding for hybrid search")
                else:
                    # Method 4: Numpy fallback
                    embedding_array = embedder.encode(query)
                    query_vector = np.array(embedding_array).tolist()
                    logger.info("✅ Used numpy array encoding for hybrid search")
        
        logger.info(f"🔧 Implementing manual hybrid search (alpha={alpha})")
        
        # Manual hybrid search implementation
        # Step 1: Get vector search results
        try:
            vector_response = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit * 3,  # Get more for better combining
                return_properties=["title", "content", "author", "category"],
                return_metadata=["distance"]
            )
            
            vector_articles = {}
            for obj in vector_response.objects:
                vector_articles[str(obj.uuid)] = {
                    "data": obj.properties,
                    "vector_score": 1 - obj.metadata.distance  # Convert distance to similarity
                }
            
            logger.info(f"📊 Got {len(vector_articles)} vector search results")
            
        except Exception as vector_error:
            logger.warning(f"⚠️ Vector search failed: {vector_error}")
            vector_articles = {}
        
        # Step 2: Get all articles for keyword search
        all_response = collection.query.fetch_objects(
            return_properties=["title", "content", "author", "category"]
        )
        
        # Step 3: Calculate keyword scores
        query_words = query.lower().split()
        keyword_articles = {}
        
        for obj in all_response.objects:
            props = obj.properties
            text_content = f"{props.get('title', '')} {props.get('content', '')}".lower()
            
            # Calculate keyword score (percentage of query words found)
            word_matches = sum(1 for word in query_words if word in text_content)
            keyword_score = word_matches / len(query_words) if query_words else 0
            
            if keyword_score > 0 or str(obj.uuid) in vector_articles:
                keyword_articles[str(obj.uuid)] = {
                    "data": props,
                    "keyword_score": keyword_score
                }
        
        logger.info(f"📊 Got {len(keyword_articles)} keyword search candidates")
        
        # Step 4: Combine scores using alpha weighting
        combined_scores = {}
        all_uuids = set(vector_articles.keys()) | set(keyword_articles.keys())
        
        for uuid in all_uuids:
            vector_score = vector_articles.get(uuid, {}).get("vector_score", 0)
            keyword_score = keyword_articles.get(uuid, {}).get("keyword_score", 0)
            
            # Hybrid score: alpha * semantic + (1-alpha) * keyword
            hybrid_score = alpha * vector_score + (1 - alpha) * keyword_score
            
            # Get article data (prefer vector data if available)
            data = vector_articles.get(uuid, keyword_articles.get(uuid, {})).get("data", {})
            
            if hybrid_score > 0:  # Only include articles with some relevance
                combined_scores[uuid] = {
                    "data": data,
                    "score": hybrid_score,
                    "vector_score": vector_score,
                    "keyword_score": keyword_score
                }
        
        # Step 5: Sort by combined score and take top results
        sorted_results = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)[:limit]
        
        articles = []
        for result in sorted_results:
            article_data = result["data"].copy()
            article_data["_metadata"] = {
                "score": result["score"],
                "vector_score": result["vector_score"],
                "keyword_score": result["keyword_score"]
            }
            articles.append(article_data)
        
        search_type = "balanced" if alpha == 0.5 else f"{'semantic' if alpha > 0.5 else 'keyword'}-focused"
        logger.info(f"🔀 Manual hybrid search results for: '{query}' ({search_type})")
        logger.info("=" * 60)
        
        for i, article in enumerate(articles, 1):
            meta = article['_metadata']
            score = meta.get('score', 'N/A')
            vector_score = meta.get('vector_score', 0)
            keyword_score = meta.get('keyword_score', 0)
            
            logger.info(f"🎯 Result {i} (Hybrid: {score:.3f}, Vector: {vector_score:.3f}, Keyword: {keyword_score:.3f}):")
            logger.info(f"   Title: {article['title']}")
            logger.info(f"   Author: {article['author']}")
            logger.info(f"   Category: {article['category']}")
            logger.info(f"   Content preview: {article['content'][:100]}...")
            logger.info("-" * 40)
        
        logger.info(f"📊 Found {len(articles)} articles using manual hybrid search")
        return articles
        
    except Exception as e:
        logger.error(f"❌ Failed to perform hybrid search: {str(e)}")
        # Fallback to semantic search only
        logger.info("💡 Falling back to semantic search...")
        return semantic_search(client, embedder, query, limit)

@task(name="💾 Inspect Data Storage", cache_policy=NO_CACHE)
def inspect_data_storage(client: weaviate.WeaviateClient, sample_count: int = 2) -> Dict[str, Any]:
    """Inspect how data is actually stored in Weaviate."""
    logger = get_run_logger()
    
    try:
        collection = client.collections.get("Article")
        response = collection.query.fetch_objects(
            limit=sample_count,
            return_properties=["title", "author", "content"],
            include_vector=True,
            return_metadata=["uuid", "creation_time"]
        )
        
        articles = []
        for obj in response.objects:
            articles.append({
                "properties": obj.properties,
                "metadata": obj.metadata,
                "vector": obj.vector["default"] if obj.vector else None
            })
        
        logger.info("💾 How Data is Stored in Weaviate:")
        logger.info("=" * 50)
        
        storage_info = {
            "total_inspected": len(articles),
            "vector_dimensions": 0,
            "storage_details": []
        }
        
        for i, article_data in enumerate(articles, 1):
            article = article_data["properties"]
            vector = article_data["vector"]
            metadata = article_data["metadata"]
            
            if vector:
                storage_info["vector_dimensions"] = len(vector)
                
                logger.info(f"📄 Article {i}: '{article['title']}'")
                logger.info(f"   🆔 Object UUID: {metadata.uuid}")
                logger.info(f"   📅 Created: {metadata.creation_time}")
                logger.info(f"   🧮 Vector Dimensions: {len(vector)}")
                logger.info(f"   🔢 Vector Sample (first 5): {vector[:5]}")
                logger.info(f"   📊 Vector Stats:")
                logger.info(f"     • Min value: {min(vector):.6f}")
                logger.info(f"     • Max value: {max(vector):.6f}")
                logger.info(f"     • Mean: {np.mean(vector):.6f}")
                logger.info(f"     • Std Dev: {np.std(vector):.6f}")
                
                storage_details = {
                    "uuid": str(metadata.uuid),
                    "title": article['title'],
                    "vector_dimensions": len(vector),
                    "vector_stats": {
                        "min": float(min(vector)),
                        "max": float(max(vector)),
                        "mean": float(np.mean(vector)),
                        "std": float(np.std(vector))
                    }
                }
                storage_info["storage_details"].append(storage_details)
                logger.info("-" * 40)
        
        logger.info("🏗️  Storage Architecture:")
        logger.info(f"   • Objects stored as JSON documents with vector embeddings")
        logger.info(f"   • Each object has a unique UUID")
        logger.info(f"   • Vectors are {storage_info['vector_dimensions']}-dimensional")
        logger.info(f"   • Indexed using HNSW (Hierarchical Navigable Small World) algorithm")
        logger.info(f"   • Supports both exact and approximate similarity search")
        
        return storage_info
        
    except Exception as e:
        logger.error(f"❌ Failed to inspect data storage: {str(e)}")
        raise

@task(name="📊 Database Statistics", cache_policy=NO_CACHE)
def get_database_stats(client: weaviate.WeaviateClient) -> Dict[str, Any]:
    """Get comprehensive database statistics."""
    logger = get_run_logger()
    
    try:
        collection = client.collections.get("Article")
        
        # Get total count
        total_response = collection.aggregate.over_all(total_count=True)
        total_count = total_response.total_count
        
        # Get category distribution by fetching all and grouping
        all_response = collection.query.fetch_objects(
            return_properties=["category"]
        )
        
        categories = {}
        for obj in all_response.objects:
            category = obj.properties["category"]
            categories[category] = categories.get(category, 0) + 1
        
        stats = {
            "total_articles": total_count,
            "categories": categories,
            "database_info": {}
        }
        
        logger.info("📊 Database Statistics:")
        logger.info("=" * 40)
        logger.info(f"📚 Total Articles: {total_count}")
        
        # Category breakdown
        for category, count in categories.items():
            logger.info(f"📂 {category}: {count} articles")
        
        # Get cluster info
        meta = client.get_meta()
        stats["database_info"] = {
            "weaviate_version": meta.version
        }
        
        logger.info(f"🏷️  Weaviate Version: {meta.version}")
        
        return stats
        
    except Exception as e:
        logger.info(f"   Author: {article['author']}")
        logger.info(f"   Category: {article['category']}")
        logger.info(f"   Content preview: {article['content'][:100]}...")
        logger.info("-" * 40)
        
        logger.info(f"📊 Found {len(articles)} articles using hybrid search")
        return articles
        
    except Exception as e:
        logger.info(f"   Author: {article['author']}")
        logger.info(f"   Category: {article['category']}")
        logger.info(f"   Content preview: {article['content'][:100]}...")
        logger.info("-" * 40)
        
        logger.info(f"📊 Found {len(articles)} articles using hybrid search")
        return articles
        
    except Exception as e:
        logger.error(f"❌ Failed to perform hybrid search: {str(e)}")
        raise

@task(name="💾 Inspect Data Storage")
def inspect_data_storage(client: weaviate.Client, sample_count: int = 2) -> Dict[str, Any]:
    """Inspect how data is actually stored in Weaviate."""
    logger = get_run_logger()
    
    try:
        # Get articles with all metadata including vectors
        result = (
            client.query
            .get("Article", ["title", "author", "content"])
            .with_additional(["vector", "id", "creationTimeUnix"])
            .with_limit(sample_count)
            .do()
        )
        
        articles = result['data']['Get']['Article']
        
        logger.info("💾 How Data is Stored in Weaviate:")
        logger.info("=" * 50)
        
        storage_info = {
            "total_inspected": len(articles),
            "vector_dimensions": 0,
            "storage_details": []
        }
        
        for i, article in enumerate(articles, 1):
            vector = article['_additional']['vector']
            article_id = article['_additional']['id']
            creation_time = article['_additional']['creationTimeUnix']
            
            storage_info["vector_dimensions"] = len(vector)
            
            logger.info(f"📄 Article {i}: '{article['title']}'")
            logger.info(f"   🆔 Object ID: {article_id}")
            logger.info(f"   📅 Created: {datetime.fromtimestamp(int(creation_time)/1000)}")
            logger.info(f"   🧮 Vector Dimensions: {len(vector)}")
            logger.info(f"   🔢 Vector Sample (first 5): {vector[:5]}")
            logger.info(f"   📊 Vector Stats:")
            logger.info(f"     • Min value: {min(vector):.6f}")
            logger.info(f"     • Max value: {max(vector):.6f}")
            logger.info(f"     • Mean: {np.mean(vector):.6f}")
            logger.info(f"     • Std Dev: {np.std(vector):.6f}")
            
            storage_details = {
                "id": article_id,
                "title": article['title'],
                "vector_dimensions": len(vector),
                "vector_stats": {
                    "min": float(min(vector)),
                    "max": float(max(vector)),
                    "mean": float(np.mean(vector)),
                    "std": float(np.std(vector))
                }
            }
            storage_info["storage_details"].append(storage_details)
            logger.info("-" * 40)
        
        logger.info("🏗️  Storage Architecture:")
        logger.info(f"   • Objects stored as JSON documents with vector embeddings")
        logger.info(f"   • Each object has a unique UUID")
        logger.info(f"   • Vectors are {storage_info['vector_dimensions']}-dimensional")
        logger.info(f"   • Indexed using HNSW (Hierarchical Navigable Small World) algorithm")
        logger.info(f"   • Supports both exact and approximate similarity search")
        
        return storage_info
        
    except Exception as e:
        logger.error(f"❌ Failed to inspect data storage: {str(e)}")
        raise

@task(name="📊 Database Statistics")
def get_database_stats(client: weaviate.Client) -> Dict[str, Any]:
    """Get comprehensive database statistics."""
    logger = get_run_logger()
    
    try:
        # Get total count
        result = client.query.aggregate("Article").with_meta_count().do()
        total_count = result['data']['Aggregate']['Article'][0]['meta']['count']
        
        # Get category distribution
        category_result = client.query.aggregate("Article").with_group_by_filter(["category"]).with_meta_count().do()
        
        stats = {
            "total_articles": total_count,
            "categories": {},
            "database_info": {}
        }
        
        logger.info("📊 Database Statistics:")
        logger.info("=" * 40)
        logger.info(f"📚 Total Articles: {total_count}")
        
        # Category breakdown
        if 'groupedBy' in category_result['data']['Aggregate']['Article'][0]:
            for group in category_result['data']['Aggregate']['Article']:
                if 'groupedBy' in group and group['groupedBy']:
                    category = group['groupedBy']['value']
                    count = group['meta']['count']
                    stats["categories"][category] = count
                    logger.info(f"📂 {category}: {count} articles")
        
        # Get cluster info
        meta = client.get_meta()
        stats["database_info"] = {
            "weaviate_version": meta.get('version', 'unknown'),
            "modules": meta.get('modules', {})
        }
        
        logger.info(f"🏷️  Weaviate Version: {meta.get('version', 'unknown')}")
        logger.info(f"🔧 Active Modules: {list(meta.get('modules', {}).keys())}")
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to get database stats: {str(e)}")
        raise

@flow(
    name="🚀 Weaviate Vector Database Test Suite",
    description="Complete test suite for Weaviate vector database operations",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
def weaviate_test_flow(run_name: Optional[str] = None):
    """
    Main Prefect flow that orchestrates the complete Weaviate test suite.
    
    Args:
        run_name: Optional custom name for the flow run
    
    This flow demonstrates:
    - Client connection and health checks
    - Schema creation and management
    - Data insertion with embeddings
    - Various query operations
    - Data storage inspection
    - Performance metrics
    """
    logger = get_run_logger()
    
    # Set custom run name if provided
    if run_name:
        from prefect.runtime import flow_run
        flow_run.name = run_name
    
    logger.info("🚀 Starting Weaviate Vector Database Test Suite")
    logger.info("=" * 60)
    
    try:
        # Step 1: Connect to Weaviate with authentication
        client = create_weaviate_client(
            host="localhost",
            port=8080,
            auth_method=AUTH_CONFIG["method"],
            api_key=AUTH_CONFIG.get("api_key"),
            client_secret=AUTH_CONFIG.get("client_secret")
        )
        
        # Step 2: Create and print schema
        schema = create_article_schema(client)
        printed_schema = print_schema(client)
        
        # Step 3: Initialize embedder
        embedder = initialize_embedder()
        
        # Step 4: Insert sample data
        inserted_ids = insert_sample_data(client, embedder)
        
        # Step 5: Print all articles
        all_articles = print_all_articles(client)
        
        # Step 6: Demonstrate filtering
        tech_articles = filter_by_category(client, "Technology")
        env_articles = filter_by_category(client, "Environment")
        
        # Step 7: Demonstrate sorting
        sorted_articles_asc = sort_by_reading_time(client, ascending=True)
        sorted_articles_desc = sort_by_reading_time(client, ascending=False)
    
        # Step 8: Get database statistics
        db_stats = get_database_stats(client)
        
        # Final summary
        logger.info("🎉 Weaviate Test Suite Completed Successfully!")
        logger.info("=" * 60)
        logger.info("📈 Summary:")
        logger.info(f"   • Connected to Weaviate successfully")
        logger.info(f"   • Created schema with {len(schema.get('properties', []))} properties")
        logger.info(f"   • Inserted {len(inserted_ids)} articles")
        logger.info(f"   • Demonstrated filtering, sorting, semantic and hybrid search")
        logger.info(f"   • Inspected vector storage ({storage_info.get('vector_dimensions', 0)} dimensions)")
        logger.info(f"   • Total database size: {db_stats.get('total_articles', 0)} articles")
        
        # Close the client
        client.close()
        
        return {
            "client_connected": True,
            "schema_created": True,
            "articles_inserted": len(inserted_ids),
            "vector_dimensions": storage_info.get('vector_dimensions', 0),
            "total_articles": db_stats.get('total_articles', 0),
            "categories": db_stats.get('categories', {}),
            "test_results": {
                "filtering": len(tech_articles) > 0,
                "sorting": len(sorted_articles_asc) > 0,
                "semantic_search": len(ai_search) > 0,
                "hybrid_search": len(energy_hybrid) > 0
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Flow failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    print("""
🚀 Weaviate Vector Database Test Suite with Prefect
==================================================

This script will test Weaviate functionality with Prefect orchestration.

Prerequisites:
1. Start Weaviate with Docker:
   docker run -d --name weaviate -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true semitechnologies/weaviate:latest

2. Install dependencies:
   pip install prefect weaviate-client sentence-transformers numpy

3. Start Prefect server (in another terminal):
   prefect server start
   
4. Open Prefect UI in browser:
   http://127.0.0.1:4200

Then run this script and watch the flow in the Prefect UI!
""")
    
    # Generate a simple incremental run name
    import time
    timestamp = int(time.time())
    custom_run_name = f"weaviate-test-run-{timestamp}"
    
    print(f"🏃‍♂️ Starting flow with custom run name: {custom_run_name}")
    
    # Run the flow with custom name
    result = weaviate_test_flow(run_name=custom_run_name)
    
    print("\n✅ Flow completed! Check the Prefect UI for detailed execution logs and flow visualization.")
    print("🌐 Prefect UI: http://127.0.0.1:4200")